# KDSIS 매뉴얼 RAG (Chunking + 검색/평가)

KDSIS 매뉴얼 정제 텍스트를 **조항 기준으로 청크 생성**하고, 생성된 청크를 기반으로 **TF‑IDF 검색 + Top‑k 평가/기록**까지 수행하는 실습용 프로젝트입니다.

## 입력/출력

### 입력

- `data/PLM_training_manual_clean.txt`
	- KDSIS 매뉴얼(PDF 추출 후 정제된 전체 텍스트)

### 출력(주요 산출물)

- `data/chunk.txt`
    -검색 입력용 청크 파일(요구된 예시 포맷 유지)
    - 헤더: `[문서명] / [조항] / [페이지]`
    - 블록 간 구분: 빈 줄 2줄

- `data/clean_with_meta.txt`
	- 제출/추적용 메타데이터 확장 정제 텍스트
	- 헤더: `[문서명] / [장] / [절] / [항] / [페이지]`

- `data/search_eval.md`
	- 검색 결과 해석 및 실패 원인 분석 기록

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
