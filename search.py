# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CHUNK_FILE = "data/chunk.txt"
TOP_K = 5


@dataclass(frozen=True)
class ChunkItem:
    doc: str
    chapter: str
    section: str
    item: str
    page: int
    text: str

    @property
    def clause(self) -> str:
        return self.item


_META_DOC_RE = re.compile(r"^\[문서명\]\s*(?P<doc>.*)\s*$")
_META_CHAPTER_RE = re.compile(r"^\[장\]\s*(?P<chapter>.*)\s*$")
_META_SECTION_RE = re.compile(r"^\[절\]\s*(?P<section>.*)\s*$")
_META_ITEM_RE = re.compile(r"^\[항\]\s*(?P<item>.*)\s*$")
_META_CLAUSE_RE = re.compile(r"^\[조항\]\s*(?P<clause>.*)\s*$")
_META_PAGE_RE = re.compile(r"^\[페이지\]\s*(?P<page>\d+)\s*$")


def load_chunk_items(path: str | Path = CHUNK_FILE) -> list[ChunkItem]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    blocks = [b.strip() for b in re.split(r"\n{3,}", text) if b.strip()]
    items: list[ChunkItem] = []

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 4:
            continue

        doc_m = _META_DOC_RE.match(lines[0].strip())
        if not doc_m:
            continue
        doc = doc_m.group("doc").strip()

        chapter = ""
        section = ""
        item = ""
        page = None

        idx = 1
        while idx < len(lines):
            line = lines[idx].strip()
            if not line:
                idx += 1
                break

            m = _META_CHAPTER_RE.match(line)
            if m:
                chapter = m.group("chapter").strip()
                idx += 1
                continue
            m = _META_SECTION_RE.match(line)
            if m:
                section = m.group("section").strip()
                idx += 1
                continue
            m = _META_ITEM_RE.match(line)
            if m:
                item = m.group("item").strip()
                idx += 1
                continue
            m = _META_CLAUSE_RE.match(line)
            if m and not item:
                item = m.group("clause").strip()
                idx += 1
                continue
            m = _META_PAGE_RE.match(line)
            if m:
                page = int(m.group("page"))
                idx += 1
                continue

            idx += 1

        if page is None or not item:
            continue

        if not chapter or not section:
            if item and item[0].isdigit() and "." in item:
                parts = [p for p in item.split(".") if p]
                chapter = chapter or parts[0]
                section = section or (".".join(parts[:2]) if len(parts) >= 2 else parts[0])
            else:
                chapter = chapter or "UNKNOWN"
                section = section or "UNKNOWN"

        body_lines = lines[idx:]
        if body_lines and not body_lines[0].strip():
            body_lines = body_lines[1:]
        body = "\n".join(body_lines).strip()
        if not body:
            continue

        items.append(ChunkItem(doc=doc, chapter=chapter, section=section, item=item, page=page, text=body))

    return items


def build_index(items: Iterable[ChunkItem]) -> tuple[TfidfVectorizer, "object", list[ChunkItem]]:
    item_list = list(items)
    texts = [it.text for it in item_list]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000)
    X = vectorizer.fit_transform(texts)
    return vectorizer, X, item_list


def search(
    query: str,
    vectorizer: TfidfVectorizer,
    X,
    items: list[ChunkItem],
    top_k: int = TOP_K,
) -> list[tuple[float, ChunkItem]]:
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    top = sims.argsort()[::-1][:top_k]
    return [(float(sims[idx]), items[int(idx)]) for idx in top]


def main() -> None:
    if not Path(CHUNK_FILE).exists():
        print("chunk.txt 파일이 없습니다.")
        return

    items = load_chunk_items(CHUNK_FILE)
    if not items:
        print("chunk.txt에서 유효한 청크를 찾지 못했습니다.")
        return

    vectorizer, X, items = build_index(items)

    print("문서 검색기 (exit 입력 시 종료)")
    while True:
        q = input("질문 > ")
        if q.lower() == "exit":
            break

        results = search(q, vectorizer, X, items, top_k=TOP_K)
        for i, (score, item) in enumerate(results, 1):
            snippet = " ".join(item.text[:200].split())
            print(f"#{i} score={score:.3f} 조항={item.clause} 페이지={item.page}")
            print(snippet, "\n")


if __name__ == "__main__":
    main()
