from __future__ import annotations

import os
import sys
import warnings
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

from .cache import CacheKey, LlmResponseCache
from .rate_limit import RateLimiter

# Transformers 로깅 레벨 설정 (경고 이상만 표시)
logging.set_verbosity_error()

# Python 경고 메시지 숨김
warnings.filterwarnings("ignore")


def build_context(top_chunks: Iterable[dict], max_chunks: int = 3) -> str:
    parts: list[str] = []
    for ch in list(top_chunks)[:max_chunks]:
        page = ch.get("page")
        header = (ch.get("header") or "").strip()
        text = (ch.get("text") or "").strip()
        parts.append(f"[페이지 {page}] {header}\n{text}".strip())
    return "\n\n".join(parts).strip()


_SYSTEM_PROMPT = (
    "너는 문서 기반 질의응답 도우미다. "
    "반드시 사용자가 제공한 Context(문서 발췌)만 근거로 답변한다. "
    "Context에 없는 내용은 추측하지 않는다."
)

# Qwen 모델 캐싱 (한 번만 로드)
_qwen_model = None
_qwen_tokenizer = None


def _load_qwen_model():
    """Qwen 모델을 로드하고 캐시한다."""
    global _qwen_model, _qwen_tokenizer
    
    if _qwen_model is not None and _qwen_tokenizer is not None:
        return _qwen_model, _qwen_tokenizer
    
    # Hugging Face 캐시를 D 드라이브로 변경
    os.environ['HF_HOME'] = 'D:/huggingface_cache'
    
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    
    # CUDA 디바이스 설정 (로그 숨김)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 진행률 표시를 위한 간단한 메시지
    print("[모델 로딩] 0%...", end="\r", flush=True)
    
    # 토크나이저 로드
    _qwen_tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        verbose=False
    )
    
    print("[모델 로딩] 30%...", end="\r", flush=True)
    
    # 표준 출력 임시 리다이렉트 (진행바 숨김)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    try:
        # 모델 로드
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    finally:
        # 표준 출력 복구
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    print("[모델 로딩] 100% 완료!       ")
    
    return _qwen_model, _qwen_tokenizer


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
        "아래 Context는 문서에서 발췌한 내용이다.\n"
        "규칙을 반드시 지켜 답변하라:\n"
        "- 문서 근거만 사용(추측 금지)\n"
        "- 5~8문장 요약\n"
        "- 출처는 문장 끝 또는 답변 마지막에 [페이지 n] 형식으로 표시\n"
        "- 근거가 부족하면 정확히 '문서에 근거가 없습니다'라고만 답하라\n\n"
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
            
            # 텍스트 생성
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=16384
            )
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
