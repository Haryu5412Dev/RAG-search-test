from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


INPUT_FILE = "data/PLM_training_manual_clean.txt"
OUTPUT_FILE = "data/chunk.txt"
META_OUTPUT_FILE = "data/clean_with_meta.txt"

DOC_NAME = "KDSIS_Manual"

# Chunk 설계 기준
# - 조항 기준(예: 3.2.1, Ⅱ-3-1 등)
# - Chunk 크기: 500 ~ 800자
# - Overlap: 50 ~ 150자
# - 문장 중간 절단 금지
# - 서로 다른 조항 혼합 금지

MIN_CHUNK = 500
MAX_CHUNK = 800
MIN_OVERLAP = 50
MAX_OVERLAP = 150

CHARS_PER_PAGE = 1800


@dataclass(frozen=True)
class Section:
	clause: str
	text: str


@dataclass(frozen=True)
class Chunk:
	doc: str
	clause: str
	chapter: str
	section: str
	item: str
	page: int
	text: str


_META_LINE_RE = re.compile(
	r"^\s*\[문서명\]\s*(?P<doc>.*?)\s*\[조항\]\s*(?P<clause>[^\]]+?)\s*\[페이지\]\s*(?P<page>\d+)\s*$"
)

_CLAUSE_RE = re.compile(
	r"^\s*(?P<clause>(?:\d+(?:\.\d+)+)|(?:\d+(?:\.\d+)+)|(?:[IVXLCDM]+-\d+(?:-\d+)+)|(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+-\d+(?:-\d+)+))\b"
)

_ROMAN_HYPHEN_RE = re.compile(
	r"^(?P<roman>[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVXLCDM]+)-(?P<a>\d+)(?:-(?P<b>\d+))?(?:-(?P<c>\d+))?$"
)


def parse_hierarchy(item: str) -> tuple[str, str, str]:
	s = item.strip()
	if not s:
		return "UNKNOWN", "UNKNOWN", "UNKNOWN"

	if "." in s and s[0].isdigit():
		parts = [p for p in s.split(".") if p]
		chapter = parts[0]
		section = ".".join(parts[:2]) if len(parts) >= 2 else chapter
		return chapter, section, s

	m = _ROMAN_HYPHEN_RE.match(s)
	if m:
		roman = m.group("roman")
		a = m.group("a")
		chapter = roman
		section = f"{roman}-{a}"
		return chapter, section, s

	return "UNKNOWN", "UNKNOWN", s


def _normalize_newlines(text: str) -> str:
	if text.startswith("\ufeff"):
		text = text.lstrip("\ufeff")
	return text.replace("\r\n", "\n").replace("\r", "\n")


def load_sections(path: str | Path) -> list[Section]:
	raw = Path(path).read_text(encoding="utf-8")
	raw = _normalize_newlines(raw)
	lines = raw.splitlines()

	sections: list[Section] = []
	current_clause: str | None = None
	buffer: list[str] = []

	def flush() -> None:
		nonlocal buffer, current_clause
		if current_clause is None:
			buffer = []
			return
		text = "\n".join(buffer).strip()
		if text:
			sections.append(Section(clause=current_clause, text=text))
		buffer = []

	for line in lines:
		m = _META_LINE_RE.match(line.strip())
		if m:
			flush()
			current_clause = m.group("clause").strip()
			continue

		if current_clause is None:
			m2 = _CLAUSE_RE.match(line)
			if m2:
				current_clause = m2.group("clause").strip()
				rest = line[m2.end() :].strip()
				if rest:
					buffer.append(rest)
			continue

		buffer.append(line)

	flush()

	if not sections:
		text = raw.strip()
		if text:
			sections.append(Section(clause="UNKNOWN", text=text))

	return sections


def split_paragraphs(text: str) -> list[str]:
	return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


def split_sentences(text: str) -> list[str]:
	s = " ".join(text.split())
	if not s:
		return []
	pieces = re.split(r"(?<=[\.!\?。？！])\s+", s)
	pieces = [p.strip() for p in pieces if p.strip()]
	return pieces if pieces else [s]


def split_long_text_by_sentence(text: str, max_len: int) -> list[str]:
	sentences = split_sentences(text)
	out: list[str] = []
	cur: list[str] = []
	cur_len = 0

	def flush() -> None:
		nonlocal cur, cur_len
		if cur:
			out.append(" ".join(cur).strip())
			cur = []
			cur_len = 0

	for sent in sentences:
		if len(sent) > max_len:
			flush()
			words = sent.split(" ")
			wcur: list[str] = []
			wlen = 0
			for w in words:
				extra = (1 if wcur else 0) + len(w)
				if wlen + extra > max_len and wcur:
					out.append(" ".join(wcur))
					wcur = [w]
					wlen = len(w)
				else:
					wcur.append(w)
					wlen += extra
			if wcur:
				out.append(" ".join(wcur))
			continue

		extra = (1 if cur else 0) + len(sent)
		if cur_len + extra > max_len and cur:
			flush()
		cur.append(sent)
		cur_len += extra

	flush()
	return out


def extract_overlap(prev_text: str, min_len: int = MIN_OVERLAP, max_len: int = MAX_OVERLAP) -> str:
	sentences = split_sentences(prev_text)
	if not sentences:
		return ""

	acc: list[str] = []
	total = 0
	for sent in reversed(sentences):
		add_len = len(sent) + (1 if acc else 0)
		if total + add_len > max_len:
			break
		acc.insert(0, sent)
		total += add_len
		if total >= min_len:
			break

	return " ".join(acc).strip() if acc else ""


def chunk_section(section_text: str) -> list[tuple[str, int]]:
	paragraphs = split_paragraphs(section_text)

	expanded: list[str] = []
	for p in paragraphs:
		if len(p) <= MAX_CHUNK:
			expanded.append(p)
		else:
			expanded.extend(split_long_text_by_sentence(p, MAX_CHUNK))

	chunks: list[tuple[str, int]] = []
	cur_parts: list[str] = []
	cur_len = 0

	def cur_text() -> str:
		return "\n\n".join(cur_parts).strip()

	def flush(new_chars: int) -> None:
		nonlocal cur_parts, cur_len
		t = cur_text()
		if t:
			chunks.append((t, new_chars))
		cur_parts = []
		cur_len = 0

	for p in expanded:
		candidate = ("\n\n".join(cur_parts + [p])).strip() if cur_parts else p
		if len(candidate) <= MAX_CHUNK:
			cur_parts.append(p)
			cur_len = len(candidate)
			continue

		if cur_len >= MIN_CHUNK:
			flush(cur_len)
			overlap = extract_overlap(chunks[-1][0])
			if overlap:
				cur_parts = [overlap]
				cur_len = len(overlap)

			candidate2 = ("\n\n".join(cur_parts + [p])).strip() if cur_parts else p
			if len(candidate2) <= MAX_CHUNK:
				cur_parts.append(p)
				cur_len = len(candidate2)
			else:
				for piece in split_long_text_by_sentence(p, max_len=max(50, MAX_CHUNK - cur_len)):
					cand3 = ("\n\n".join(cur_parts + [piece])).strip() if cur_parts else piece
					if len(cand3) > MAX_CHUNK and cur_parts:
						flush(cur_len)
						overlap = extract_overlap(chunks[-1][0])
						cur_parts = [overlap] if overlap else []
						cur_len = len(overlap) if overlap else 0
					cur_parts.append(piece)
					cur_len = len(("\n\n".join(cur_parts)).strip())
		else:
			if cur_parts:
				flush(cur_len)
				overlap = extract_overlap(chunks[-1][0])
				cur_parts = [overlap] if overlap else []
				cur_len = len(overlap) if overlap else 0

			for piece in split_long_text_by_sentence(p, max_len=MAX_CHUNK):
				if not cur_parts:
					cur_parts = [piece]
					cur_len = len(piece)
				else:
					cand4 = ("\n\n".join(cur_parts + [piece])).strip()
					if len(cand4) <= MAX_CHUNK:
						cur_parts.append(piece)
						cur_len = len(cand4)
					else:
						flush(cur_len)
						overlap = extract_overlap(chunks[-1][0])
						cur_parts = [overlap, piece] if overlap else [piece]
						cur_len = len(("\n\n".join(cur_parts)).strip())

	if cur_parts:
		flush(cur_len)

	return chunks


def build_chunks(sections: list[Section]) -> list[Chunk]:
	out: list[Chunk] = []
	cumulative_new_chars = 0

	for sec in sections:
		chapter, section, item = parse_hierarchy(sec.clause)
		pieces = chunk_section(sec.text)
		for text, new_chars in pieces:
			page = (cumulative_new_chars // CHARS_PER_PAGE) + 1
			out.append(
				Chunk(
					doc=DOC_NAME,
					clause=sec.clause,
					chapter=chapter,
					section=section,
					item=item,
					page=int(page),
					text=text,
				)
			)
			cumulative_new_chars += int(new_chars)

	return out



def write_chunk_file(chunks: list[Chunk], path: str | Path) -> None:
	blocks: list[str] = []
	for ch in chunks:
		blocks.append(
			"\n".join(
				[
					f"[문서명] {ch.doc}",
					f"[조항] {ch.clause}",
					f"[페이지] {ch.page}",
					"",
					ch.text.strip(),
				]
			).strip()
		)
	Path(path).write_text("\n\n\n".join(blocks) + "\n", encoding="utf-8")


def write_meta_text_file(chunks: list[Chunk], path: str | Path) -> None:
	blocks: list[str] = []
	for ch in chunks:
		blocks.append(
			"\n".join(
				[
					f"[문서명] {ch.doc}",
					f"[장] {ch.chapter}",
					f"[절] {ch.section}",
					f"[항] {ch.item}",
					f"[페이지] {ch.page}",
					"",
					ch.text.strip(),
				]
			).strip()
		)
	Path(path).write_text("\n\n\n".join(blocks) + "\n", encoding="utf-8")


def validate(chunks: list[Chunk]) -> None:
	lengths = [len(c.text) for c in chunks]
	if not lengths:
		print("전체 Chunk 개수: 0")
		return

	total = len(lengths)
	mn = min(lengths)
	mx = max(lengths)
	avg = sum(lengths) / total
	under = sum(1 for x in lengths if x < MIN_CHUNK)
	over = sum(1 for x in lengths if x > MAX_CHUNK)

	print(f"전체 Chunk 개수: {total}")
	print(f"Chunk 길이(min / max / avg): {mn} / {mx} / {avg:.1f}")
	print(f"500자 미만 Chunk 개수: {under}")
	print(f"800자 초과 Chunk 개수: {over}")


def main() -> None:
	in_path = Path(INPUT_FILE)
	if not in_path.exists():
		raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_FILE}")

	sections = load_sections(in_path)
	chunks = build_chunks(sections)
	write_chunk_file(chunks, OUTPUT_FILE)
	write_meta_text_file(chunks, META_OUTPUT_FILE)
	validate(chunks)


if __name__ == "__main__":
	main()
