from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Chunk:
    page: int
    header: str
    text: str


# 요구 포맷: "[페이지 12] 헤더"
_PAGE_HEADER_INLINE_RE = re.compile(r"^\s*\[\s*페이지\s+(?P<page>\d+)\s*\]\s*(?P<header>.*)\s*$")

# 현재 레포 데이터 폴백: "[페이지] 12" (메타 라인)
_PAGE_META_RE = re.compile(r"^\s*\[\s*페이지\s*\]\s*(?P<page>\d+)\s*$")

# 현재 레포 데이터(한 줄 메타): "[문서명] ... [조항] ... [페이지] 12"
_PAGE_META_INLINE_RE = re.compile(r"\[\s*페이지\s*\]\s*(?P<page>\d+)\b")


def _normalize_newlines(text: str) -> str:
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_chunks(path: str | Path) -> list[dict]:
    """텍스트 파일을 읽어 page/header/text dict 리스트(chunks)를 생성.

    기본 규칙(요구사항):
    - "[페이지 n] 헤더" 라인을 인식
    - 그 아래 본문을 다음 "[페이지 ...]" 전까지 하나의 청크로 묶음

    폴백(현재 레포 데이터 호환):
    - "[페이지] n" 메타 라인을 인식
    - 다음 블록에서 첫 비어있지 않은 줄을 header로, 나머지를 text로 사용
    """

    raw = Path(path).read_text(encoding="utf-8")
    raw = _normalize_newlines(raw)
    lines = raw.splitlines()

    chunks: list[Chunk] = []

    cur_page: int | None = None
    cur_header: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal cur_page, cur_header, buffer
        if cur_page is None:
            buffer = []
            return
        body = "\n".join(buffer).strip()
        header = (cur_header or "").strip()
        if header or body:
            chunks.append(Chunk(page=int(cur_page), header=header, text=body))
        buffer = []

    for line in lines:
        m = _PAGE_HEADER_INLINE_RE.match(line)
        if m:
            flush()
            cur_page = int(m.group("page"))
            cur_header = m.group("header").strip()
            continue

        # 폴백 1: [페이지] n 단독 메타 라인
        m = _PAGE_META_RE.match(line)
        if m:
            flush()
            cur_page = int(m.group("page"))
            cur_header = None
            continue

        # 폴백 2: [문서명] ... [조항] ... [페이지] n (한 줄)
        # -> 새로운 블록 시작으로 보고 flush 후 page만 세팅
        if "[페이지]" in line and "[문서명]" in line:
            m = _PAGE_META_INLINE_RE.search(line)
            if m:
                flush()
                cur_page = int(m.group("page"))
                cur_header = None
                continue

        # 본문 축적
        if cur_page is not None:
            buffer.append(line)

    flush()

    # 폴백 처리: header가 비어있으면 본문 첫 줄을 header로 승격
    normalized: list[Chunk] = []
    for ch in chunks:
        if ch.header.strip():
            normalized.append(ch)
            continue

        body_lines = [ln for ln in ch.text.split("\n")]
        # leading empty lines drop
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        if not body_lines:
            continue

        header = body_lines[0].strip()
        full_body = "\n".join(body_lines).strip()
        rest = "\n".join(body_lines[1:]).strip()
        # 폴백 데이터는 첫 줄에 헤더+본문이 붙어있는 경우가 많아,
        # 검색 품질을 위해 text를 비우지 않도록 전체 본문을 유지한다.
        normalized.append(Chunk(page=ch.page, header=header, text=rest or full_body))

    return [asdict(c) for c in normalized]


def save_chunks_jsonl(chunks: Iterable[dict], path: str | Path) -> None:
    p = Path(path)
    lines = [json.dumps(c, ensure_ascii=False) for c in chunks]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
