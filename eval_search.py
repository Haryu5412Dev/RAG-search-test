# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import search


CHUNK_FILE = "data/chunk.txt"
EVAL_MD = "data/search_eval.md"
TOP_K = 5


@dataclass(frozen=True)
class QueryResult:
    question: str
    top_score: float
    rows: list[tuple[int, str, str]]  # rank, clause, summary


def summarize(text: str, max_len: int = 80) -> str:
    s = " ".join(text.split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def run_queries() -> list[QueryResult]:
    items = search.load_chunk_items(CHUNK_FILE)
    vectorizer, X, items = search.build_index(items)

    questions = [
        "KDSIS 접속 URL과 외부망 접속(VPN) 절차는 어떻게 되나요?",
        "KDSIS의 2단계 인증(2FA) 구성과 계정 잠금 조건은 무엇인가요?",
        "KDSIS 분류 체계(Taxonomy)는 어떤 구조로 운영되나요?",
        "데이터 정합성 검증에서 오류 유형과 처리 방식은 무엇인가요?",
        "KDSIS에서 TDP(기술자료 묶음) 관리가 의미하는 범위는 무엇인가요?",
    ]

    out: list[QueryResult] = []
    for q in questions:
        results = search.search(q, vectorizer, X, items, top_k=TOP_K)
        rows: list[tuple[int, str, str]] = []
        top_score = results[0][0] if results else 0.0
        for rank, (score, item) in enumerate(results, 1):
            rows.append((rank, item.clause, summarize(item.text)))
        out.append(QueryResult(question=q, top_score=float(top_score), rows=rows))

    return out


def print_table(qrs: list[QueryResult]) -> None:
    print("질문 | rank | 조항 | chunk 일부 요약")
    print("---|---:|---|---")
    for qr in qrs:
        for rank, clause, summary in qr.rows:
            q = qr.question.replace("|", "\\|")
            s = summary.replace("|", "\\|")
            print(f"{q} | {rank} | {clause} | {s}")


def classify(qrs: list[QueryResult]) -> tuple[list[QueryResult], list[QueryResult]]:
    ordered = sorted(qrs, key=lambda x: x.top_score, reverse=True)
    good = ordered[:3]
    bad = ordered[3:]
    return good, bad


def problem_type(q: str, top_score: float) -> str:
    if top_score < 0.10:
        return "메타데이터 부족"
    if len(q.replace(" ", "")) < 14:
        return "모호성"
    return "chunk 크기"


def analysis_text(ptype: str, top_score: float) -> tuple[str, str]:
    if ptype == "메타데이터 부족":
        return (
            f"상위 결과의 유사도 점수({top_score:.3f})가 낮아 핵심 단어 매칭이 약함.",
            "메타데이터(키워드/용어 동의어) 강화 또는 BM25/하이브리드 검색 도입.",
        )
    if ptype == "모호성":
        return (
            "질문이 포괄적이거나 키워드가 부족해 특정 조항으로 수렴하지 않음.",
            "질문 템플릿(주제+대상+조건) 적용 또는 질의 확장(동의어/관련어) 추가.",
        )
    return (
        "청크 내부에 질문의 핵심 내용이 분산되어 점수가 분산될 가능성이 있음.",
        "청크 크기/overlap 튜닝(문단 경계 강화) 또는 조항 내 소제목 단위 분리.",
    )


def write_eval_md(good: list[QueryResult], bad: list[QueryResult], path: str | Path) -> None:
    lines: list[str] = []
    lines.append("# 검색 평가")
    lines.append("")

    lines.append("## 잘 검색된 질문 3개")
    lines.append("")
    for qr in good:
        lines.append(f"### 질문: {qr.question}")
        lines.append("")
        lines.append("- 문제 유형: 없음")
        lines.append(f"- 원인 분석: 상위 결과 유사도 점수({qr.top_score:.3f})가 상대적으로 높고 관련 조항으로 수렴")
        lines.append("- 개선 방향: 동의어 사전/전처리(표기 통일)로 안정성 강화")
        lines.append("")

    lines.append("## 잘 검색되지 않은 질문 2개")
    lines.append("")
    for qr in bad:
        ptype = problem_type(qr.question, qr.top_score)
        cause, improvement = analysis_text(ptype, qr.top_score)
        lines.append(f"### 질문: {qr.question}")
        lines.append("")
        lines.append(f"- 문제 유형: {ptype}")
        lines.append(f"- 원인 분석: {cause}")
        lines.append(f"- 개선 방향: {improvement}")
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not Path(CHUNK_FILE).exists():
        raise FileNotFoundError(f"청크 파일이 없습니다: {CHUNK_FILE}")

    qrs = run_queries()
    print_table(qrs)

    good, bad = classify(qrs)
    write_eval_md(good, bad, EVAL_MD)


if __name__ == "__main__":
    main()
