# RAG Search Test

Qwen3-4B 모델을 사용한 문서 기반 질의응답(RAG) 시스템

## 📁 프로젝트 구조

```
RAG-search-test/
├── main.py                    # 메인 실행 프로그램
├── requirements.txt           # Python 패키지 의존성
├── .env                       # 환경변수 설정 (API 키 등)
├── .env.example               # 환경변수 예시 파일
│
├── data/                      # 문서 데이터
│   └── PLM_training_manual_clean.txt  # 국방표준정보 관리 시스템 매뉴얼
│
└── util/                      # 유틸리티 모듈
    ├── __init__.py
    ├── chunking.py            # 문서 청킹 (페이지/헤더 기반)
    ├── retrieval.py           # TF-IDF 기반 검색
    ├── generation.py          # LLM 답변 생성 (Qwen/Gemini/OpenAI)
    ├── cache.py               # LLM 응답 캐싱
    └── rate_limit.py          # API 호출 제한
```

## 🚀 시작하기

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 실행

```bash
python main.py
```

## ✨ 주요 기능

- ✅ **자동 문서 처리**: PLM 매뉴얼을 자동으로 로드하고 청킹
- ✅ **TF-IDF 검색**: 질문과 관련된 Top-K 문서 검색
- ✅ **Qwen3-4B 모델**: 로컬 LLM으로 요약 답변 생성
- ✅ **대화형 인터페이스**: 반복적인 질문-답변 가능
- ✅ **에러 처리**: API 제한, 모델 오류 등 안전 처리
- ✅ **로딩 표시**: 답변 생성 중 스피너 애니메이션

## 🔧 환경 설정

`.env` 파일에서 LLM 제공자 선택 가능:

```bash
# 기본값: Qwen (로컬 모델)
LLM_PROVIDER=qwen

# 또는 다른 제공자
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your_key_here

# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_key_here
```

## 📊 사용 예시

```
질문 > KDSIS에서 설계 변경을 하려면 무엇을 먼저 해야 하나요?

============================================================
🔍 질문: KDSIS에서 설계 변경을 하려면 무엇을 먼저 해야 하나요?
============================================================

📚 검색 결과 (Top-3):
[검색 #1] 페이지 44
  제목: 제15장 용어 사전 (Glossary)
  내용: ECP (Engineering Change Proposal): 형상변경 제안서...

============================================================
🤖 LLM 답변 생성 중...
============================================================

💡 답변:
KDSIS에서 설계 변경을 시작하려면 먼저 ECP(Engineering Change Proposal) 
문서를 작성하고 제출해야 합니다. [페이지 44]
...
```

## 🛠️ 기술 스택

- **LLM**: Qwen3-4B-Instruct-2507 (Hugging Face Transformers)
- **검색**: scikit-learn (TF-IDF)
- **GPU**: CUDA 12.8 지원
- **캐싱**: JSON 기반 응답 캐시

## 📝 라이선스

MIT License
