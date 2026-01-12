from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


def _default_cache_path() -> Path:
    return Path(os.getenv("LLM_CACHE_PATH", ".rag_pipeline/llm_cache.sqlite3"))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_program_version() -> str:
    """프로그램 버전을 생성 (프롬프트 및 주요 설정의 해시)"""
    # generation.py 파일의 내용을 읽어서 해시 생성
    # 프롬프트나 주요 로직이 변경되면 자동으로 버전이 달라짐
    try:
        gen_file = Path(__file__).parent / "generation.py"
        if gen_file.exists():
            content = gen_file.read_text(encoding="utf-8")
            # 주요 설정만 추출 (주석 제외)
            lines = [line.strip() for line in content.split('\n') 
                    if line.strip() and not line.strip().startswith('#')]
            key_content = '\n'.join(lines[:100])  # 처음 100줄 정도만 사용
            return _sha256(key_content)[:16]  # 짧게 16자만 사용
    except Exception:
        pass
    return "v1.0.0"


@dataclass(frozen=True)
class CacheKey:
    provider: str
    model: str
    question: str
    context: str

    def digest(self) -> str:
        # context가 길 수 있어 안정적으로 해시
        payload = f"provider={self.provider}\nmodel={self.model}\nq={self.question}\nctx={self.context}"
        return _sha256(payload)


class LlmResponseCache:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = Path(db_path) if db_path else _default_cache_path()
        # ':memory:'는 커넥션마다 DB가 분리되므로 단일 연결을 유지한다.
        self._in_memory = str(self._path) == ":memory:"
        self._conn: sqlite3.Connection | None = None
        _ensure_parent(self._path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._in_memory:
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:")
                self._conn.execute("PRAGMA synchronous=NORMAL;")
            return self._conn

        conn = sqlite3.connect(str(self._path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            # 버전 테이블 생성
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_version (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            
            # 캐시 테이블 생성
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    question TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            
            # 버전 체크 및 캐시 무효화
            current_version = _get_program_version()
            row = conn.execute("SELECT version FROM cache_version WHERE id = 1").fetchone()
            
            if row is None:
                # 첫 실행: 버전 저장
                conn.execute(
                    "INSERT INTO cache_version (id, version, updated_at) VALUES (1, ?, ?)",
                    (current_version, int(time.time()))
                )
                conn.commit()
            elif row[0] != current_version:
                # 버전 변경: 캐시 삭제
                print(f"[캐시] 프로그램 변경 감지 - 캐시 초기화 중...")
                conn.execute("DELETE FROM llm_cache")
                conn.execute(
                    "UPDATE cache_version SET version = ?, updated_at = ? WHERE id = 1",
                    (current_version, int(time.time()))
                )
                conn.commit()

    def get(self, key: CacheKey) -> str | None:
        digest = key.digest()
        conn = self._connect()
        row = conn.execute("SELECT response FROM llm_cache WHERE key=?", (digest,)).fetchone()
        if not self._in_memory:
            conn.close()
        return row[0] if row else None

    def set(self, key: CacheKey, response: str) -> None:
        digest = key.digest()
        ctx_hash = _sha256(key.context)
        created_at = int(time.time())
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache
            (key, provider, model, question, context_hash, response, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (digest, key.provider, key.model, key.question, ctx_hash, response, created_at),
        )
        conn.commit()
        if not self._in_memory:
            conn.close()

    def clear(self) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM llm_cache")
        conn.commit()
        if not self._in_memory:
            conn.close()
