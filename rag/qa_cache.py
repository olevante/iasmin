import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CacheItem:
    answer: str
    sources: list[str]
    confidence: float


def _normalizar_texto(texto: str) -> str:
    t = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalizar_pergunta(pergunta: str) -> str:
    return _normalizar_texto(pergunta)


def _conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def inicializar_cache(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                question_norm TEXT NOT NULL,
                question_raw TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                index_signature TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_qa_cache_lookup
            ON qa_cache(scope, question_norm, index_signature, created_at DESC)
            """
        )


def buscar_resposta_exata(
    db_path: str,
    scope: str,
    question_norm: str,
    index_signature: str,
) -> Optional[CacheItem]:
    with _conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT answer, sources_json, confidence
            FROM qa_cache
            WHERE scope = ? AND question_norm = ? AND index_signature = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (scope, question_norm, index_signature),
        ).fetchone()
        if row is None:
            return None
        return CacheItem(
            answer=row["answer"],
            sources=json.loads(row["sources_json"]),
            confidence=float(row["confidence"]),
        )


def salvar_resposta(
    db_path: str,
    scope: str,
    question_raw: str,
    answer: str,
    sources: list[str],
    confidence: float,
    index_signature: str,
) -> None:
    question_norm = normalizar_pergunta(question_raw)
    with _conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO qa_cache (
                scope, question_norm, question_raw, answer, sources_json, confidence, index_signature
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scope,
                question_norm,
                question_raw,
                answer,
                json.dumps(sources, ensure_ascii=False),
                float(confidence),
                index_signature,
            ),
        )


def compactar_cache(db_path: str, manter_registros: int = 5000) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            """
            DELETE FROM qa_cache
            WHERE id NOT IN (
                SELECT id FROM qa_cache
                ORDER BY created_at DESC
                LIMIT ?
            )
            """,
            (int(manter_registros),),
        )
