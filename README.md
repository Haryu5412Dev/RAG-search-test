# KDSIS 매뉴얼 RAG (Chunking + 검색/평가)

KDSIS 매뉴얼 정제 텍스트를 **조항 기준으로 청크 생성**하고, 생성된 청크를 기반으로 **TF‑IDF 검색 + Top‑k 평가/기록**까지 수행하는 실습용 프로젝트입니다.

## 입력/출력

### 입력

- `data/PLM_training_manual_clean.txt`
	- KDSIS 매뉴얼(PDF 추출 후 정제된 전체 텍스트)

### 출력(주요 산출물)

- `data/chunk.txt`
	- 검색 입력용 청크 파일(요구된 예시 포맷 유지)
    - 헤더: `[문서명] / [조항] / [페이지]`
    - 블록 간 구분: 빈 줄 2줄

- `data/clean_with_meta.txt`
	- 제출/추적용 메타데이터 확장 정제 텍스트
	- 헤더: `[문서명] / [장] / [절] / [항] / [페이지]`

- `data/search_eval.md`
	- 검색 결과 해석 및 실패 원인 분석 기록

- `data/search_answers.md`
	- 질문별 답변(근거 기반 요약 + Top 근거 발췌) 기록

## 설치

```bash
pip install -r requirements.txt
```

## 실행 순서

### 1) 청크 생성 + 메타데이터 정제 텍스트 생성

```bash
python build_chunks.py
```

- 생성 파일: `data/chunk.txt`, `data/clean_with_meta.txt`
- 콘솔 출력: 전체 청크 개수/길이 통계

### 2) 청크 예시 검증(랜덤 5개)

```bash
python verify_chunks.py
```

- 본문은 출력하지 않고, 메타데이터/길이/종결(휴리스틱)만 확인합니다.

### 3) 검색기(대화형)

```bash
python search.py
```

- `data/chunk.txt`를 로드하여 Top‑k 결과를 출력합니다.
- 종료: `exit`

### 4) 검색 인덱스 테스트 + 평가 기록 생성

```bash
python eval_search.py
```

- 질문 5개를 하드코딩하여 Top‑k=5 검색을 수행합니다.
- 콘솔 출력: `질문 | rank | 조항 | chunk 일부 요약` 표
- 파일 출력: `data/search_eval.md`

## 구성 파일

- `build_chunks.py`: 조항 기준 청크 생성 + `chunk.txt`/`clean_with_meta.txt` 생성
- `search.py`: `chunk.txt` 파싱/인덱스 구축/검색
- `eval_search.py`: 검색 테스트(Top‑k) + `search_eval.md` 저장
- `verify_chunks.py`: 랜덤 5개 청크 품질 점검

## Chunk 예시 검증

랜덤 5개 청크를 뽑아 메타데이터와 최소 품질(길이/문장 종결)을 확인합니다.

```bash
python verify_chunks.py
```

최근 실행 출력(예):

```text
샘플 검증(랜덤 5개):
- 항=15.2 (장=15, 절=15.2, 페이지=6) len=319 end_ok=True
- 항=3.1 (장=3, 절=3.1, 페이지=1) len=218 end_ok=True
- 항=4.4 (장=4, 절=4.4, 페이지=2) len=200 end_ok=True
- 항=3.4 (장=3, 절=3.4, 페이지=2) len=280 end_ok=True
- 항=19.1 (장=19, 절=19.1, 페이지=6) len=189 end_ok=False
```

`end_ok`는 청크가 문장 부호로 끝나는지 확인하는 휴리스틱입니다.

## 질문-검색 결과 / 실패 분석

`eval_search.py`를 실행하면 질문 5개에 대해 Top‑5 검색을 수행하고 결과/분석을 `data/search_eval.md`에 저장합니다.

```bash
python eval_search.py
```

- 결과 표: 질문별 Top‑5(유사도 점수 포함)
- 실패 분석: 성공 3개 / 실패 2개 분류 + 원인/개선 방향
