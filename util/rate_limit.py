from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


def _state_path() -> Path:
    return Path(os.getenv("LLM_RATE_STATE_PATH", ".rag_pipeline/rate_state.json"))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RateLimitConfig:
    per_minute: int
    per_day: int


class RateLimiter:
    """로컬 스크립트용 간단한 요청 횟수 제한.

    - 분 단위(rolling 60s)와 일 단위(UTC day) 카운트를 관리
    - 상태를 파일에 저장해 프로세스 재시작 후에도 과사용을 줄임
    """

    def __init__(self, config: RateLimitConfig | None = None, state_path: str | Path | None = None) -> None:
        per_minute = int(os.getenv("MAX_LLM_REQUESTS_PER_MINUTE", "30"))
        per_day = int(os.getenv("MAX_LLM_REQUESTS_PER_DAY", "200"))
        self._config = config or RateLimitConfig(per_minute=per_minute, per_day=per_day)
        self._path = Path(state_path) if state_path else _state_path()
        _ensure_parent(self._path)

    def _load(self) -> dict:
        if not self._path.exists():
            return {"minute": [], "day": {"day_key": None, "count": 0}}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {"minute": [], "day": {"day_key": None, "count": 0}}

    def _save(self, state: dict) -> None:
        self._path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _day_key(ts: float) -> str:
        # UTC 기준 YYYY-MM-DD
        return time.strftime("%Y-%m-%d", time.gmtime(ts))

    def check_and_consume(self, n: int = 1) -> None:
        now = time.time()
        state = self._load()

        # minute rolling window
        minute: list[float] = [float(x) for x in state.get("minute", [])]
        minute = [t for t in minute if now - t < 60.0]
        if len(minute) + n > self._config.per_minute:
            oldest = min(minute) if minute else now
            wait_s = max(0, int(60 - (now - oldest)))
            raise RuntimeError(
                f"LLM 요청이 너무 많습니다(분당 {self._config.per_minute}회 제한). 약 {wait_s}초 후 다시 시도하세요."
            )

        # day window
        day = state.get("day", {}) or {}
        day_key = self._day_key(now)
        if day.get("day_key") != day_key:
            day = {"day_key": day_key, "count": 0}

        if int(day.get("count", 0)) + n > self._config.per_day:
            raise RuntimeError(f"LLM 요청이 너무 많습니다(일당 {self._config.per_day}회 제한). 내일 다시 시도하세요.")

        # consume
        minute.extend([now] * n)
        day["count"] = int(day.get("count", 0)) + n
        state["minute"] = minute
        state["day"] = day
        self._save(state)
