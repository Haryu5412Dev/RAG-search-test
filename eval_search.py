from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import search


CHUNK_FILE = "data/chunk.txt"
EVAL_MD = "data/search_eval.md"
ANSWERS_MD = "data/search_answers.md"
TOP_K = 5


@dataclass(frozen=True)
class QueryResult:
    question: str
    top_score: float
    rows: list[tuple[int, float, search.ChunkItem, str]]  # rank, score, item, summary


def summarize(text: str, max_len: int = 80) -> str:
    s = " ".join(text.split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def run_queries() -> list[QueryResult]:
    items = search.load_chunk_items(CHUNK_FILE)
    vectorizer, X, items = search.build_index(items)

    questions = [
        "형상 변경(ECP)을 1급으로 신청해야 하는 경우와, 이때 반드시 첨부해야 하는 서류는 무엇인가요?",
        "도면을 수정할 때 체크아웃 → 체크인 절차를 반드시 따라야 하는 이유와, 체크인 시 필수로 입력해야 하는 항목은 무엇인가요?",
        "KDSIS에서 데이터 품질 관리(DQM)에 의해 ‘Obsolete’ 상태로 분류되는 경우는 언제이며, 해당 상태가 승인 프로세스에 미치는 영향은 무엇인가요?",
        "통합 검색 엔진에서 특정 규격명을 정확히 검색해야 할 때 사용해야 하는 검색 방법과 연산자는 무엇인가요?",
        "KDSIS가 DELIS와 형상 정보를 연동하는 목적은 무엇이며, 데이터 전송 실패 시 시스템은 어떤 조치를 수행하나요?",
    ]

    out: list[QueryResult] = []
    for q in questions:
        results = search.search(q, vectorizer, X, items, top_k=TOP_K)
        rows: list[tuple[int, float, search.ChunkItem, str]] = []
        top_score = results[0][0] if results else 0.0
        for rank, (score, item) in enumerate(results, 1):
            rows.append((rank, float(score), item, summarize(item.text)))
        out.append(QueryResult(question=q, top_score=float(top_score), rows=rows))

    return out


def print_table(qrs: list[QueryResult]) -> None:
    print("질문 | rank | score | 조항 | chunk 일부 요약")
    print("---|---:|---:|---|---")
    for qr in qrs:
        for rank, score, item, summary in qr.rows:
            q = qr.question.replace("|", "\\|")
            s = summary.replace("|", "\\|")
            print(f"{q} | {rank} | {score:.3f} | {item.clause} | {s}")


def write_results_table_md(qrs: list[QueryResult], lines: list[str]) -> None:
    lines.append("## 질문별 Top-5 검색 결과")
    lines.append("")
    lines.append("질문 | rank | score | 조항 | chunk 일부 요약")
    lines.append("---|---:|---:|---|---")
    for qr in qrs:
        for rank, score, item, summary in qr.rows:
            q = qr.question.replace("|", "\\|")
            s = summary.replace("|", "\\|")
            lines.append(f"{q} | {rank} | {score:.3f} | {item.clause} | {s}")
    lines.append("")


_NON_WORD_RE = re.compile(r"[^0-9A-Za-z가-힣_]+")


def _keywords(question: str, max_keywords: int = 10) -> list[str]:
    raw = [_NON_WORD_RE.sub("", t) for t in question.split()]
    toks = [t for t in raw if len(t) >= 2]
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_keywords:
            break
    return out


def _split_sentences(text: str) -> list[str]:
    t = " ".join(text.replace("\r", "").split())
    if not t:
        return []
    parts = re.split(r"(?<=[.!?。])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def _clean_bullet_sentence(s: str) -> str:
    t = " ".join(s.split())
    if not t:
        return t

    # 흔한 제목/라벨을 제거해 사람이 읽기 쉽게 만든다.
    # 예) "제6장 ...6.1 통합 검색 ..." / "5.2 형상 변경 절차 - ..."
    t = re.sub(r"^제\s*\d+장\s*", "", t)
    t = re.sub(r"^\d+(?:\.\d+)+\s+", "", t)
    return t.strip()


def extractive_answer_sentences(question: str, evidence_text: str, max_sentences: int = 3) -> list[str]:
    sents = _split_sentences(evidence_text)
    if not sents:
        return []

    keys = _keywords(question)
    picked: list[str] = []

    if keys:
        for s in sents:
            if any(k in s for k in keys):
                picked.append(s)
            if len(picked) >= max_sentences:
                break

    if not picked:
        picked = sents[: min(max_sentences, len(sents))]

    cleaned = [_clean_bullet_sentence(x) for x in picked]
    return [c for c in cleaned if c]


def _clip(text: str, max_chars: int = 360) -> str:
    t = " ".join(text.split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def write_answers_md(qrs: list[QueryResult], path: str | Path) -> None:
    lines: list[str] = []
    lines.append("# 질문별 답변(근거 기반)")
    lines.append("")
    lines.append(
        "아래 답변은 TF‑IDF 검색 결과의 상위 청크(Top‑1/Top‑2)를 기반으로 **원문 발췌/요약(추출형)** 한 것입니다. "
        "정확한 서술은 근거(조항/페이지)와 함께 확인하세요."
    )
    lines.append("")

    for idx, qr in enumerate(qrs, 1):
        lines.append(f"## Q{idx}. {qr.question}")
        lines.append("")
        if not qr.rows:
            lines.append("- 검색 결과 없음")
            lines.append("")
            continue

        rank1_score, rank1_item = qr.rows[0][1], qr.rows[0][2]
        lines.append(
            f"- Top-1 근거: 문서={rank1_item.doc}, 조항={rank1_item.clause}, 페이지={rank1_item.page}, score={rank1_score:.3f}"
        )

        bullets = extractive_answer_sentences(qr.question, rank1_item.text)
        lines.append("- 답변(불릿 요약):")
        if bullets:
            for b in bullets:
                lines.append(f"  - {b}")
        else:
            lines.append("  - (근거 텍스트가 비어 있어 답변을 생성하지 못함)")
        lines.append("")
        lines.append("근거 발췌(Top-1):")
        lines.append("")
        lines.append(f"> { _clip(rank1_item.text) }")

        if len(qr.rows) >= 2:
            rank2_score, rank2_item = qr.rows[1][1], qr.rows[1][2]
            lines.append("")
            lines.append(
                f"추가 근거(Top-2): 문서={rank2_item.doc}, 조항={rank2_item.clause}, 페이지={rank2_item.page}, score={rank2_score:.3f}"
            )
            lines.append("")
            lines.append(f"> { _clip(rank2_item.text) }")

        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


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
    qrs = good + bad
    lines: list[str] = []
    lines.append("# 검색 평가")
    lines.append("")

    lines.append("## 선정 기준")
    lines.append("")
    lines.append(
        "총 5개 질문에 대해 Top-5 검색을 수행한 뒤, 각 질문의 **Top-1 유사도 점수(score)** 기준으로 "
        "상위 3개를 '잘 검색된 질문', 하위 2개를 '잘 검색되지 않은 질문'으로 선택해 원인/개선 방향을 기록한다."
    )
    lines.append("")

    write_results_table_md(qrs, lines)

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
    write_answers_md(qrs, ANSWERS_MD)


if __name__ == "__main__":
    main()
