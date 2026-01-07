# -*- coding: utf-8 -*-
from __future__ import annotations

import random

import search


CHUNK_FILE = "data/chunk.txt"


def ends_clean(text: str) -> bool:
    t = text.rstrip()
    if not t:
        return True
    return t.endswith((".", "?", "!", "。", "？", "！", "다.", "다"))


def main() -> None:
    items = search.load_chunk_items(CHUNK_FILE)
    if not items:
        print("샘플 검증: chunk가 없습니다")
        return

    random.seed(20260107)
    sample = random.sample(items, k=min(5, len(items)))

    print("샘플 검증(랜덤 5개):")
    for it in sample:
        print(
            f"- 항={it.item} (장={it.chapter}, 절={it.section}, 페이지={it.page}) "
            f"len={len(it.text)} end_ok={ends_clean(it.text)}"
        )


if __name__ == "__main__":
    main()
