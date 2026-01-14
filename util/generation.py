from __future__ import annotations

import os
import sys
import warnings
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
from transformers import BitsAndBytesConfig
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

from .cache import CacheKey, LlmResponseCache
from .rate_limit import RateLimiter

# CPU 모드 전역 최적화 설정
def _configure_cpu_optimization():
    """CPU 모드에서 성능 최적화 설정"""
    use_gpu = os.getenv('USE_GPU', 'true').lower() == 'true'
    
    if not use_gpu:
        # CPU 스레드 수 제한 (시스템 과부하 방지)
        cpu_threads = int(os.getenv('CPU_THREADS', '4'))
        torch.set_num_threads(cpu_threads)
        os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(cpu_threads)
        print(f"[CPU 최적화] 스레드 수: {cpu_threads}")

# 프로그램 시작 시 CPU 최적화 적용
_configure_cpu_optimization()

# Transformers 로깅 레벨 설정 (경고 이상만 표시)
logging.set_verbosity_error()

# Python 경고 메시지 숨김
warnings.filterwarnings("ignore")


def build_context(top_chunks: Iterable, max_chunks: int = 3) -> str:
    """
    검색 결과(SearchHit 또는 dict)를 컨텍스트 문자열로 변환
    """
    parts: list[str] = []
    use_gpu = os.getenv('USE_GPU', 'true').lower() == 'true'
    
    # CPU 모드에서는 더 작은 컨텍스트 사용 (메모리/속도 최적화)
    max_text_len = 200 if use_gpu else 50  # CPU: 100 -> 50자로 감소
    max_chunks_limit = max_chunks if use_gpu else 2  # CPU: 청크 2개로 제한
    
    for item in list(top_chunks)[:max_chunks_limit]:
        # SearchHit 객체인 경우 chunk 속성에서 가져오기
        if hasattr(item, 'chunk'):
            ch = item.chunk
        else:
            ch = item
        
        page = ch.get("page")
        header = (ch.get("header") or "").strip()
        text = (ch.get("text") or "").strip()
        
        # 텍스트 길이 제한
        if len(text) > max_text_len:
            text = text[:max_text_len] + "..."
        
        parts.append(f"[페이지 {page}] {header}\n{text}".strip())
    return "\n\n".join(parts).strip()


_SYSTEM_PROMPT = (
    "당신은 문서 기반 질의응답 도우미입니다. "
    "반드시 사용자가 제공한 Context(문서 발취)만 근거로 존댓말로 답변합니다. "
    "Context에 없는 내용은 추측하지 않습니다."
)

# Qwen 모델 캐싱 (한 번만 로드)
_qwen_model = None
_qwen_tokenizer = None


def _load_qwen_model():
    """Qwen 모델을 로드하고 캐시한다."""
    global _qwen_model, _qwen_tokenizer
    
    if _qwen_model is not None and _qwen_tokenizer is not None:
        return _qwen_model, _qwen_tokenizer
    
    # Hugging Face 캐시 디렉토리 설정
    hf_home = os.getenv('HF_HOME', 'D:/huggingface_cache')
    os.environ['HF_HOME'] = hf_home
    
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    
    # .env에서 GPU 사용 여부 확인
    use_gpu = os.getenv('USE_GPU', 'true').lower() == 'true'
    
    # GPU 비활성화 옵션
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # CUDA 디바이스 설정
    device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
    print(f"[디바이스] {'GPU' if device.type == 'cuda' else 'CPU'} 모드로 실행   ")
    
    # 진행률 표시를 위한 간단한 메시지
    print("[모델 로딩] 0%    ", end="\r", flush=True)
    
    # 토크나이저 로드
    _qwen_tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        verbose=False
    )
    
    print("[모델 로딩] 30%   ", end="\r", flush=True)
    
    # 표준 출력 임시 리다이렉트 (진행바 숨김)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    try:
        if device.type == 'cpu':
            # CPU 모드: INT8 양자화 + 메모리 최적화
            print("[CPU 모드] INT8 양자화 적용 중...", end="\r", flush=True, file=old_stdout)
            _qwen_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # CPU에서는 float32 사용
                low_cpu_mem_usage=True,  # 메모리 최적화
                trust_remote_code=True,
            )
            # 동적 INT8 양자화 적용 (CPU 최적화)
            _qwen_model = torch.quantization.quantize_dynamic(
                _qwen_model,
                {torch.nn.Linear},  # Linear 레이어만 양자화
                dtype=torch.qint8
            )
        else:
            # GPU 모드: 4-bit 양자화
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # 모델 로드 (Flash Attention 시도, 실패 시 일반 모드)
            try:
                _qwen_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
            except ImportError:
                # Flash Attention이 없으면 일반 모드로 로드
                _qwen_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
    finally:
        # 표준 출력 복구
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    print("[모델 로딩] 100% 완료!  ")
    
    return _qwen_model, _qwen_tokenizer


def preload_model():
    """모델을 미리 로드 (프리로딩)"""
    _load_qwen_model()


def answer_with_llm(question: str, context: str) -> str:
    """Top-3 컨텍스트를 근거로 5~8문장 요약 답변 + 출처를 생성.

    요구 프롬프트 규칙:
    - 문서 근거만 사용
    - 5~8문장
    - 출처를 [페이지 n] 형식으로 표시
    - 근거가 없으면 "문서에 근거가 없습니다"
    - API 키는 환경변수에서 읽기
    """

    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if not provider:
        # 기본값을 qwen으로 설정
        provider = "qwen"

    # 모델명은 provider에 따라 달라서 캐시 키에 포함
    if provider == "gemini":
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    elif provider == "qwen":
        model_name = "Qwen/Qwen3-4B-Instruct-2507"
    else:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    cache = LlmResponseCache()
    cache_key = CacheKey(provider=provider, model=model_name, question=question, context=context)
    cached = cache.get(cache_key)
    if cached:
        return cached

    # 캐시 미스일 때만 호출 제한(과사용 방지)
    RateLimiter().check_and_consume(1)

    prompt = (
        "아래 Context는 문서에서 발취한 내용입니다.\n"
        "규칙을 반드시 지켜 존댓말로 답변하세요:\n"
        "- 문서 근거만 사용(추측 금지)\n"
        "- 5~8문장 요약\n"
        "- 각 문장에 페이지 번호를 표시하지 말고, 자연스럽게 설명한 후 마지막에 '참고 페이지: [페이지 15, 19]' 형식으로 모든 페이지를 한 번에 나열\n"
        "- 근거가 부족하면 정확히 '문서에 근거가 없습니다'라고만 답하세요\n"
        "- 존댓말을 사용하세요 (예: ~입니다, ~합니다)\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
    )

    if provider == "qwen":
        try:
            model, tokenizer = _load_qwen_model()
            
            # 메시지 포맷 준비
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            # CPU/GPU 모드에 따른 토큰 수 조정
            use_gpu = os.getenv('USE_GPU', 'true').lower() == 'true'
            max_tokens = 256 if use_gpu else 64  # CPU 모드: 메모리 부담 최소화
            
            # 텍스트 생성 설정
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": False,  # Greedy decoding (더 빠름)
                "top_p": None,
            }
            
            # CPU 모드 추가 최적화
            if not use_gpu:
                gen_kwargs["use_cache"] = False  # KV 캐시 비활성화 (메모리 절약)
            
            generated_ids = model.generate(**model_inputs, **gen_kwargs)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            out = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            if out:
                cache.set(cache_key, out)
            return out
            
        except Exception as e:
            raise RuntimeError(f"Qwen 모델 호출 오류: {type(e).__name__}: {e}") from e

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다.")

        import google.generativeai as genai

        genai.configure(api_key=api_key)

        # system_instruction 지원 버전에서는 시스템 규칙을 분리 전달
        try:
            model = genai.GenerativeModel(model_name=model_name, system_instruction=_SYSTEM_PROMPT)
            resp = model.generate_content(prompt)
        except TypeError:
            model = genai.GenerativeModel(model_name=model_name)
            resp = model.generate_content(f"{_SYSTEM_PROMPT}\n\n{prompt}")
        except Exception as e:
            name = type(e).__name__
            msg = str(e)
            if name in {"ResourceExhausted", "TooManyRequests"} or "429" in msg or "quota" in msg.lower():
                raise RuntimeError("Gemini API quota/요청 제한에 걸렸습니다. 잠시 후 재시도하거나 요금제/쿼터를 확인하세요.") from e
            raise

        out = (getattr(resp, "text", "") or "").strip()
        if out:
            cache.set(cache_key, out)
        return out

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

        # OpenAI 호환 라이브러리(v1) 사용
        from openai import OpenAI

        model = model_name
        base_url = os.getenv("OPENAI_BASE_URL")

        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

        # 일부 모델은 temperature를 지원하지 않으므로 기본은 생략한다.
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as e:
            name = type(e).__name__
            msg = str(e)
            if name in {"BadRequestError"} and ("temperature" in msg or "unsupported_value" in msg):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
            else:
                raise

        out = (resp.choices[0].message.content or "").strip()
        if out:
            cache.set(cache_key, out)
        return out

    raise RuntimeError("LLM_PROVIDER는 'qwen', 'gemini' 또는 'openai' 여야 합니다.")
