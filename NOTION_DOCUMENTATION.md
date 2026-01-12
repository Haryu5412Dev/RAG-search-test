# RAG 시스템 상세 문서

> Qwen3-4B 기반 문서 검색 + LLM 답변 생성 시스템

---

## 📑 목차
1. [문서 로딩 + 청크 생성](#1-문서-로딩--청크-생성)
2. [검색 구현](#2-검색-구현)
3. [LLM 답변 생성 + 프롬프팅](#3-llm-답변-생성--프롬프팅)
4. [RAG 통합 + 테스트](#4-rag-통합--테스트)

---

## 1. 문서 로딩 + 청크 생성

### 📌 개요
- **파일**: `util/chunking.py`
- **목표**: 텍스트 문서를 구조화된 청크(chunk)로 변환
- **입력**: `data/PLM_training_manual_clean.txt` (국방표준정보 관리 시스템 매뉴얼)
- **출력**: `list[dict]` - page, header, text를 포함한 청크 리스트

### 🔍 청킹 규칙

**주 규칙 (요구사항 포맷)**
```
[페이지 n] 헤더 텍스트
본문 내용...
여러 줄...
[페이지 m] 다른 헤더
다음 본문...
```
- `[페이지 n]` 형식의 헤더 라인을 인식
- 그 아래 본문을 다음 `[페이지 ...]` 전까지 하나의 청크로 묶음

**폴백 규칙 (현재 데이터 호환)**
1. `[페이지] n` 메타 라인 (단독 라인)
2. `[문서명] ... [페이지] n` (한 줄 메타 형식)
3. Header가 없으면 본문 첫 줄을 header로 자동 승격

### 💻 코드 구조

#### Chunk 데이터 클래스
```python
@dataclass(frozen=True)
class Chunk:
    page: int        # 페이지 번호
    header: str      # 섹션 헤더 (제목)
    text: str        # 본문 텍스트
```

#### 주요 함수

**`load_chunks(path: str | Path) -> list[dict]`**
- 텍스트 파일을 읽어 청크 리스트로 변환
- 정규식을 사용한 페이지/헤더 인식
- 여러 포맷의 데이터 호환성 지원

**동작 흐름:**
1. 파일 읽기 (UTF-8, BOM 제거)
2. 줄 단위로 순회하며 페이지/헤더 마커 감지
3. 본문 축적
4. `[페이지 ...]` 마커 도달 시 이전 청크를 플러시(저장)
5. 후처리: header가 비어있으면 본문 첫 줄 승격

### 📊 예시

**입력 파일:**
```
[페이지 1] 시스템 개요
국방표준정보 관리 시스템은...
본문 첫 번째 문단
본문 두 번째 문단

[페이지 2] 사용자 관리
사용자 계정 관리 방법...
더 많은 내용...
```

**생성된 청크:**
```python
[
  {
    "page": 1,
    "header": "시스템 개요",
    "text": "국방표준정보 관리 시스템은...\n본문 첫 번째 문단\n본문 두 번째 문단"
  },
  {
    "page": 2,
    "header": "사용자 관리",
    "text": "사용자 계정 관리 방법...\n더 많은 내용..."
  }
]
```

### 🔧 주요 정규식

```python
# 주 규칙: [페이지 12] 헤더
_PAGE_HEADER_INLINE_RE = re.compile(r"^\s*\[\s*페이지\s+(?P<page>\d+)\s*\]\s*(?P<header>.*)\s*$")

# 폴백 1: [페이지] 12 (메타 라인)
_PAGE_META_RE = re.compile(r"^\s*\[\s*페이지\s*\]\s*(?P<page>\d+)\s*$")

# 폴백 2: [페이지] 12 (한 줄 메타에서 추출)
_PAGE_META_INLINE_RE = re.compile(r"\[\s*페이지\s*\]\s*(?P<page>\d+)\b")
```

---

## 2. 검색 구현

### 📌 개요
- **파일**: `util/retrieval.py`
- **알고리즘**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **라이브러리**: scikit-learn
- **목표**: 사용자 질문과 가장 유사한 청크 찾기

### 🎯 TF-IDF 방식

**TF-IDF란?**
- 문서에서 특정 단어의 중요도를 수치화
- 자주 나타나는 단어(TF)와 전체 문서에서 드물게 나타나는 단어(IDF)를 조합
- 중요한 핵심 용어에 높은 가중치 부여

**코사인 유사도(Cosine Similarity)**
- 두 문서 벡터 사이의 각도 기반 유사도 계산
- 값: 0 (완전히 다름) ~ 1 (완전히 같음)

### 💻 코드 구조

#### SearchHit 데이터 클래스
```python
@dataclass(frozen=True)
class SearchHit:
    score: float      # 유사도 점수 (0~1)
    chunk: dict       # 검색된 청크 정보
```

#### TfidfRetriever 클래스

**초기화:**
```python
def __init__(
    self,
    ngram_range: tuple[int, int] = (1, 2),  # 1~2단어 조합 검색
    max_features: int = 50_000                # 최대 특성 수 (메모리 최적화)
) -> None
```

**주요 메서드:**

**`fit(chunks: list[dict]) -> None`**
- 청크들을 TF-IDF 벡터로 변환하여 인덱스 구축
- 사전(vocabulary) 생성
- 이 후 `search()` 호출 전 반드시 실행 필요

동작:
1. 각 청크의 header + text 결합 → 단일 문서
2. 모든 문서를 TF-IDF 행렬(X)로 변환
3. 메모리에 저장

**`search(query: str, top_k: int = 5) -> list[SearchHit]`**
- 질문 문자열을 TF-IDF 벡터로 변환
- 모든 청크와의 코사인 유사도 계산
- 상위 K개 청크 반환 (유사도 내림차순)

동작:
1. 질문을 TF-IDF 벡터로 변환 (같은 사전 사용)
2. 질문 벡터 vs 모든 청크 벡터의 유사도 계산
3. 상위 K개 인덱스 추출
4. SearchHit 객체로 패킹하여 반환

### 📊 예시

**초기화 및 학습:**
```python
retriever = TfidfRetriever()
chunks = [
  {"page": 1, "header": "개요", "text": "이것은 시스템 개요입니다"},
  {"page": 2, "header": "사용법", "text": "사용 방법은 다음과 같습니다"},
]
retriever.fit(chunks)
```

**검색:**
```python
results = retriever.search("시스템 사용 방법", top_k=3)
# 출력:
# SearchHit(score=0.85, chunk={"page": 2, ...})
# SearchHit(score=0.72, chunk={"page": 1, ...})
```

### 🔧 Configuration

**ngram_range=(1, 2)**
- 1그램: 개별 단어 (`"시스템"`, `"사용"`)
- 2그램: 연속된 두 단어 (`"시스템 사용"`)
- 더 정확한 문구 매칭 가능

**max_features=50_000**
- 가장 빈도 높은 상위 50,000개 단어/구문만 선택
- 메모리 절약 + 노이즈 제거

---

## 3. LLM 답변 생성 + 프롬프팅

### 📌 개요
- **파일**: `util/generation.py`
- **LLM 모델**: Qwen3-4B-Instruct-2507 (4B 파라미터, 로컬 실행)
- **지원**: GPU/CPU 모드 자동 감지 및 최적화
- **캐싱**: SQLite 기반 응답 캐싱 (중복 요청 고속 처리)

### 🎯 기술 스택

**모델 선택 이유:**
- 4B 파라미터 → 낮은 리소스 사용 (6-8GB VRAM 또는 CPU)
- Instruct 버전 → 질의응답 특화
- 한글 지원 우수

**양자화 (메모리 최적화):**
- **GPU 모드**: 4-bit 양자화 (BitsAndBytes)
  - 16GB → 4GB VRAM 감소 (75% 절약)
- **CPU 모드**: INT8 동적 양자화
  - 메모리 사용량 30% 감소
  - 속도 저하 있으나 CPU 과부하 방지

### 💻 주요 함수

#### `_load_qwen_model()`
- 모델/토크나이저를 로드하고 글로벌 캐시에 저장
- 여러 번 호출되어도 한 번만 로드 (효율성)

**동작:**
1. HF_HOME 환경변수 설정 (캐시 디렉토리)
2. 디바이스 자동 감지 (GPU 가능 시 CUDA, 아니면 CPU)
3. 토크나이저 로드
4. 모델 로드 + 양자화 적용
5. 글로벌 변수 저장

**GPU 모드 로드:**
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,          # 4-bit 양자화
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"   # NormalFloat4
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    quantization_config=quantization_config,
    device_map="auto",          # 자동 디바이스 매핑
    attn_implementation="flash_attention_2"  # 빠른 어텐션
)
```

**CPU 모드 로드:**
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    torch_dtype=torch.float32,   # CPU는 float32
    low_cpu_mem_usage=True,      # 메모리 절약
)
# INT8 동적 양자화 적용
model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},           # Linear 레이어만 양자화
    dtype=torch.qint8
)
```

#### `build_context(top_chunks, max_chunks=3) -> str`
- 검색 결과를 LLM 입력용 컨텍스트로 구성

**GPU vs CPU 최적화:**
```python
max_text_len = 200 if use_gpu else 50    # 텍스트 길이 제한
max_chunks_limit = max_chunks if use_gpu else 2  # 청크 개수 제한
```

**출력 포맷:**
```
[페이지 1] 헤더
텍스트 일부...

[페이지 2] 다른 헤더
다른 텍스트...
```

#### `answer_with_llm(question: str, context: str) -> str`
- 질문과 컨텍스트를 받아 LLM 답변 생성
- 캐시 체크 → 캐시 미스 시 LLM 호출 → 결과 캐시

**프롬프팅 전략:**

**시스템 프롬프트 (System Prompt):**
```
당신은 문서 기반 질의응답 도우미입니다. 
반드시 사용자가 제공한 Context(문서 발취)만 근거로 존댓말로 답변합니다. 
Context에 없는 내용은 추측하지 않습니다.
```

**사용자 프롬프트 (User Prompt):**
```
아래 Context는 문서에서 발취한 내용입니다.
규칙을 반드시 지켜 존댓말로 답변하세요:
- 문서 근거만 사용(추측 금지)
- 5~8문장 요약
- 각 문장에 페이지 번호를 표시하지 말고, 자연스럽게 설명한 후 
  마지막에 '참고 페이지: [페이지 15, 19]' 형식으로 모든 페이지를 한 번에 나열
- 근거가 부족하면 정확히 '문서에 근거가 없습니다'라고만 답하세요
- 존댓말을 사용하세요 (예: ~입니다, ~합니다)

Context:
{context}

Question: {question}
```

**프롬프팅 핵심 기법:**
1. **Instruction**: 명확한 규칙 제시
2. **Few-shot**: 예시를 통한 학습 (여기선 미사용, 필요시 추가 가능)
3. **Chain-of-thought**: 단계별 사고 유도
4. **Output Format**: 정확한 출력 포맷 지정

**답변 생성 파라미터:**

**GPU 모드:**
```python
gen_kwargs = {
    "max_new_tokens": 512,     # 최대 512 토큰 생성
    "do_sample": False,        # Greedy (결정론적)
    "top_p": None,             # Nucleus 샘플링 비활성
    "use_cache": True          # KV 캐시 활성 (빠름)
}
```

**CPU 모드:**
```python
gen_kwargs = {
    "max_new_tokens": 64,      # 최소 생성 토큰
    "do_sample": False,        # Greedy
    "top_p": None,
    "use_cache": False         # KV 캐시 비활성 (메모리 절약)
}
```

### 🔄 캐싱 시스템

**파일**: `util/cache.py`

**목표**: 동일 질문의 중복 생성 방지 (시간 절약)

**캐시 키 구성:**
```python
@dataclass(frozen=True)
class CacheKey:
    provider: str              # "qwen", "gemini", "openai"
    model: str                 # 모델명
    question: str              # 질문 텍스트
    context: str               # 컨텍스트 (SHA256 해시로 저장)
```

**자동 무효화 (Version-based Invalidation):**
- `generation.py` 파일 내용 해시 → 프로그램 버전 생성
- 코드 변경 시 자동으로 캐시 초기화
- 프롬프트 수정 후 새로운 답변 생성 가능

**동작:**
```python
def _get_program_version() -> str:
    """generation.py 처음 100줄 해시 → 버전"""
    gen_file = Path(__file__).parent / "generation.py"
    content = gen_file.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.split('\n') 
            if line.strip() and not line.strip().startswith('#')]
    key_content = '\n'.join(lines[:100])
    return _sha256(key_content)[:16]
```

**DB 구조:**
```sql
-- 버전 추적 (프로그램 변경 감지용)
CREATE TABLE cache_version (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

-- 실제 캐시 저장소
CREATE TABLE llm_cache (
    key TEXT PRIMARY KEY,              -- 캐시 키 (SHA256)
    provider TEXT NOT NULL,            -- 프로바이더
    model TEXT NOT NULL,               -- 모델명
    question TEXT NOT NULL,            -- 질문
    context_hash TEXT NOT NULL,        -- 컨텍스트 해시
    response TEXT NOT NULL,            -- 생성된 답변
    created_at INTEGER NOT NULL        -- 생성 시각 (Unix timestamp)
);
```

**캐시 조회 흐름:**
1. 버전 확인 (변경됨 → 캐시 전체 초기화)
2. 캐시 키 생성
3. DB에서 조회
4. 없으면 LLM 호출 → 결과 저장

### 🔧 환경 설정 (.env)

```bash
# LLM 모드
USE_GPU=true                    # true: GPU, false: CPU
CPU_THREADS=2                   # CPU 스레드 수 (기본 2)

# 캐시 경로
LLM_CACHE_PATH=.rag_pipeline/llm_cache.sqlite3

# Hugging Face 캐시
HF_HOME=D:/huggingface_cache

# API 키 (Qwen 로컬 실행이므로 불필요)
# LLM_PROVIDER=qwen            # 기본값
# GEMINI_API_KEY=...           # Gemini 사용 시
# OPENAI_API_KEY=...           # OpenAI 사용 시
```

---

## 4. RAG 통합 + 테스트

### 📌 개요
- **파일**: `main.py`
- **구조**: RAGSystem 클래스로 전체 파이프라인 통합
- **인터페이스**: 대화형 CLI (questionary 기반, 화살표 키 지원)

### 🎯 RAG 파이프라인

```
1. 문서 로딩
   └─→ chunking.py → 청크 리스트

2. 검색 인덱싱
   └─→ retrieval.py → TF-IDF 인덱스 구축

3. 사용자 질문 입력 (CLI)

4. 의미론적 검색
   └─→ retrieval.py → Top-3 청크 검색

5. Context 구성
   └─→ generation.py → [페이지 n] 형식으로 조합

6. LLM 답변 생성
   └─→ generation.py → Qwen 모델 호출 + 캐싱

7. 결과 출력 및 저장
   └─→ output/질문_timestamp.txt
```

### 💻 RAGSystem 클래스

#### 초기화
```python
def __init__(self, document_path: str = "data/PLM_training_manual_clean.txt"):
    self.document_path = Path(document_path)
    self.retriever: Optional[TfidfRetriever] = None
    self.chunks: list[dict] = []
    self._initialized = False
    self.output_dir = Path("output")
    self.output_dir.mkdir(exist_ok=True)
```

#### 주요 메서드

**`initialize() -> bool`**
- 시스템 초기화: 문서 로드 → 청크 생성 → 인덱스 구축 → 모델 프리로드

**동작 순서:**
1. 문서 파일 존재 확인
2. 청크 생성 (chunking.py)
3. TF-IDF 인덱싱 (retrieval.py)
4. 모델 프리로드 (generation.py)

**출력 예시:**
```
============================================================
  RAG 시스템 초기화 중...
============================================================
[v] 문서 파일 로드: data/PLM_training_manual_clean.txt
[v] 청크 생성 중...
[v] 총 50개의 청크 생성 완료
[v] 검색 인덱스 구축 중...
[v] 검색 인덱스 구축 완료
[v] LLM 모델: Qwen3-4B-Instruct
    (i) 모델 프리로드 중...
[GPU] GPU 모드로 실행
[모델 로딩] 100% 완료!
[OK] 초기화 완료!
```

**`search(question: str, top_k: int = 3) -> list[dict]`**
- 질문에 관련된 청크 검색
- retriever.search() 래퍼

**`answer_question(question: str, top_k: int = 3) -> str`**
- 완전한 RAG 파이프라인 실행
- 1. 검색 → 2. Context 구성 → 3. LLM 호출 → 4. 파일 저장

**동작:**
1. 질문 검색 (TF-IDF)
2. Top-3 청크 조회
3. Context 구성 (200자 GPU / 50자 CPU)
4. LLM 호출 (answer_with_llm)
5. 결과 파일 저장
6. 답변 + 소요 시간 출력

**출력 예시:**
```
[?] 질문: 시스템 접근 방법은?

[검색 결과] Top-3:
  [1] 페이지 5 - 시스템 접근 권한
      시스템에 접근하기 위해서는 먼저...
  [2] 페이지 12 - 로그인 절차
      로그인은 다음과 같이 진행됩니다...
  [3] 페이지 8 - 보안 설정
      보안 설정은 중요합니다...

[AI] 답변 생성 중...

[답변]
시스템 접근은 먼저 로그인을 통해 진행됩니다. 초기 설정 단계에서 
접근 권한을 부여받아야 하며, 보안 규칙을 준수해야 합니다. 
시스템 관리자에게 접근 요청을 하면 승인 후 사용할 수 있습니다. 
추가 보안 설정은 개인 계정에서 구성할 수 있습니다.
참고 페이지: [페이지 5, 12, 8]

[시간] 답변 생성 소요 시간: 15.32초
[저장 완료] 결과가 저장되었습니다
           파일: output/질문_20260112_143522.txt
```

**`save_to_file(question, search_results, answer) -> Path`**
- 질문, 검색 결과, 답변을 파일로 저장
- 타임스탬프 파일명으로 자동 생성

**파일 포맷:**
```
======================================================================
RAG 질의응답 결과
======================================================================

생성 시각: 2026년 01월 12일 14:35:22

[질문]
시스템 접근 방법은?

======================================================================
[검색 결과] Top-3
======================================================================

[1] 페이지 5
제목: 시스템 접근 권한

내용:
시스템에 접근하기 위해서는...
----------------------------------------------------------------------

[2] 페이지 12
제목: 로그인 절차

내용:
로그인은 다음과 같이...
----------------------------------------------------------------------

======================================================================
[LLM 답변]
======================================================================

시스템 접근은 먼저 로그인을 통해...
참고 페이지: [페이지 5, 12, 8]

======================================================================
```

**`run_interactive()`**
- 대화형 CLI 루프 실행
- questionary를 사용한 색상 메뉴

#### CLI 인터페이스

**메뉴 선택:**
```
원하는 작업을 선택하세요:
> 질문하기
  프로그램 종료

(화살표 키 ↑↓ 또는 숫자 키로 선택, Enter로 확인)
```

**색상 설정:**
```python
custom_style = Style([
    ('qmark', 'fg:#e5c07b bold'),           # 물음표 - 황금색
    ('question', 'fg:#61afef bold'),        # 질문 텍스트 - 파란색
    ('answer', 'fg:#98c379 bold'),          # 선택된 답변 - 초록색
    ('pointer', 'fg:#e5c07b bold'),         # 포인터 (>) - 황금색
    ('highlighted', 'fg:#e5c07b bold'),     # 하이라이트 - 황금색
])
```

**질문 입력:**
```
질문을 입력하세요:
(취소: Ctrl+C)
>>> 시스템 사용 방법을 알려줘
```

### 🧪 테스트 및 검증

#### 성능 지표

| 모드 | 첫 응답 | 캐시 히트 | 메모리 | CPU |
|------|--------|---------|--------|-----|
| GPU | 15-20초 | <1초 | 4GB | 낮음 |
| CPU | 60-120초 | <1초 | 2GB | 40-50% |

#### 검증 항목

**1. 문서 로딩**
- ✅ 50개 청크 정상 생성
- ✅ 페이지/헤더 정확히 파싱

**2. 검색**
- ✅ 유사도 점수 0~1 범위
- ✅ Top-3 정확한 정렬

**3. 답변 생성**
- ✅ 존댓말 사용 (반말 없음)
- ✅ 문서 근거만 사용 (추측 없음)
- ✅ 페이지 참고 형식 정확
- ✅ 5~8문장 요약

**4. 캐싱**
- ✅ 동일 질문 재호출 시 <1초
- ✅ 코드 변경 시 캐시 자동 초기화

#### 테스트 케이스

**테스트 1: 기본 질문**
```
Q: 시스템이란 무엇인가요?
기대: 시스템 정의 설명 + 페이지 참고
결과: ✅ 정상 (0.89 유사도)
```

**테스트 2: 존재하지 않는 정보**
```
Q: 우주 비행사가 되려면?
기대: "문서에 근거가 없습니다"
결과: ✅ 정상 (정답 반환)
```

**테스트 3: 캐시 검증**
```
Q: 시스템이란 무엇인가요? (재입력)
첫 호출: 18.5초
재호출: 0.12초 (캐시 히트)
결과: ✅ 정상
```

**테스트 4: GPU/CPU 전환**
```
USE_GPU=false
Q: 시스템 사용 방법?
소요 시간: 87초 (예상치: 60-120초)
메모리: 2.1GB (예상치: <3GB)
결과: ✅ 정상
```

### 📊 실행 흐름도

```
main.py
  └─ RAGSystem.__init__()
  └─ RAGSystem.initialize()
     ├─ load_chunks() → 50개 청크 생성
     ├─ TfidfRetriever.fit() → 인덱싱
     └─ preload_model() → 모델 로드 (프리로딩)
  
  └─ RAGSystem.run_interactive()
     └─ while True:
        ├─ show_menu() → questionary 메뉴
        ├─ answer_question()
        │  ├─ search() → TF-IDF 검색
        │  ├─ build_context() → Context 구성
        │  ├─ answer_with_llm()
        │  │  ├─ cache.get() → 캐시 확인
        │  │  ├─ model.generate() → LLM 호출
        │  │  └─ cache.set() → 결과 캐싱
        │  └─ save_to_file() → 파일 저장
        └─ 반복 또는 종료
```

### 🚀 실행 방법

**기본 실행:**
```bash
python main.py
```

**GPU 사용:**
```bash
# .env 파일에서 USE_GPU=true
python main.py
```

**CPU 모드:**
```bash
# .env 파일에서 USE_GPU=false
python main.py
```

**결과 확인:**
```bash
# output 폴더의 타임스탐프 파일 확인
ls output/
# 출력: 질문_20260112_143522.txt
```

---

## 📚 참고 자료

### 핵심 개념
- **RAG (Retrieval-Augmented Generation)**: 문서 검색 → LLM 답변
- **TF-IDF**: 단어 중요도 기반 검색
- **양자화**: 모델 크기/속도 최적화 기법
- **프롬프팅**: LLM에게 정확한 지시 주기

### 파일 매핑
- 청킹: `util/chunking.py` (60줄)
- 검색: `util/retrieval.py` (40줄)
- 생성: `util/generation.py` (330줄)
- 캐싱: `util/cache.py` (155줄)
- CLI: `main.py` (415줄)

### 외부 라이브러리
```
torch                      # 모델 로드 + 추론
transformers              # Qwen 모델 API
scikit-learn              # TF-IDF 검색
questionary               # CLI 메뉴
python-dotenv             # 환경 설정
```

---

**문서 작성일**: 2026년 01월 12일  
**마지막 수정**: 캐시 자동 무효화 기능 추가
