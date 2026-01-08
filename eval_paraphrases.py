from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import search


CHUNK_FILE = "data/chunk.txt"
OUT_MD = "data/search_eval_paraphrases.md"
TOP_K = 3


@dataclass(frozen=True)
class Row:
    rank: int
    score: float
    clause: str
    page: int
    summary: str


def summarize(text: str, max_len: int = 100) -> str:
    s = " ".join(text.split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def run() -> tuple[list[str], dict[str, list[Row]]]:
    questions = [
        "형상 변경은 언제 ECP로 신청해야 하나요?",
        "제품 설계나 규격이 바뀌면 어떤 절차를 따라야 하나요?",
        "KDSIS에서 설계 변경을 하려면 무엇을 먼저 해야 하나요?",
    ]

    items = search.load_chunk_items(CHUNK_FILE)
    vectorizer, X, items = search.build_index(items)

    out: dict[str, list[Row]] = {}
    for q in questions:
        results = search.search(q, vectorizer, X, items, top_k=TOP_K)
        rows: list[Row] = []
        for rank, (score, item) in enumerate(results, 1):
            rows.append(
                Row(
                    rank=rank,
                    score=float(score),
                    clause=item.clause,
                    page=item.page,
                    summary=summarize(item.text),
                )
            )
        out[q] = rows

    return questions, out


def write_md(questions: list[str], results: dict[str, list[Row]]) -> None:
    lines: list[str] = []
    lines.append("# 동의 질문 3개 비교 실험")
    lines.append("")
    lines.append("대상 질문은 의미가 유사한 3개이며, TF‑IDF 검색에서 `top_k=3`으로 비교했다.")
    lines.append("")

    lines.append("## 질문별 Top-3")
    lines.append("")
    lines.append("질문 | rank | score | 조항 | 페이지 | chunk 일부 요약")
    lines.append("---|---:|---:|---|---:|---")

    for q in questions:
        for r in results.get(q, []):
            qq = q.replace("|", "\\|")
            ss = r.summary.replace("|", "\\|")
            lines.append(f"{qq} | {r.rank} | {r.score:.3f} | {r.clause} | {r.page} | {ss}")

    lines.append("")

    # winner by Top-1 score
    best_q = None
    best_score = -1.0
    best_top1_clause = ""
    for q in questions:
        rows = results.get(q, [])
        if not rows:
            continue
        if rows[0].score > best_score:
            best_score = rows[0].score
            best_q = q
            best_top1_clause = rows[0].clause

    lines.append("## 결론")
    lines.append("")
    if best_q is None:
        lines.append("- 유효한 검색 결과가 없어 비교할 수 없음")
    else:
        lines.append(f"- Top-1 점수 기준 최적 질문: **{best_q}**")
        lines.append(f"- 해당 질문의 Top-1: 조항={best_top1_clause}, score={best_score:.3f}")
        lines.append("")
        lines.append("### 해석")
        lines.append("- ECP 키워드가 포함된 질문은 `5.2.1`(ECP 절차)로 수렴할 가능성이 높다.")
        lines.append("- 절차/순서 중심 질문은 `5.2.1` 외에 `5.2.2`(CCB) 등으로 점수가 분산될 수 있다.")

    Path(OUT_MD).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not Path(CHUNK_FILE).exists():
        raise FileNotFoundError(f"청크 파일이 없습니다: {CHUNK_FILE}")

    questions, results = run()
    write_md(questions, results)


if __name__ == "__main__":
    main()
