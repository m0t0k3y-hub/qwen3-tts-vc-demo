"""Generation history shared by all users (SQLite metadata + WAV files on disk)."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

# Columns that can be filtered / shown in the list; everything else lives in the `params` JSON.
_COLUMNS = ["id", "created_at", "kind", "size", "attn", "language", "text", "speaker", "instruct",
            "ref_audio", "ref_text", "x_vector_only", "seed", "normalize", "audio_path",
            "synth_sec", "audio_sec", "rtf", "peak_reserved_gib", "load_sec", "client_ip"]

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS generations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    size TEXT NOT NULL,
    attn TEXT,
    language TEXT,
    text TEXT,
    speaker TEXT,
    instruct TEXT,
    ref_audio TEXT,
    ref_text TEXT,
    x_vector_only INTEGER,
    seed INTEGER,
    normalize INTEGER,
    audio_path TEXT NOT NULL,
    synth_sec REAL,
    audio_sec REAL,
    rtf REAL,
    peak_reserved_gib REAL,
    load_sec REAL,
    client_ip TEXT,
    params TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS generations_created_at ON generations(created_at);
"""


class HistoryStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def add(self, record: dict) -> None:
        """`record` has the keys in _COLUMNS (missing ones are NULL) plus `params` (a dict)."""
        cols = [c for c in _COLUMNS if c in record] + ["params"]
        values = [record[c] for c in cols[:-1]] + [json.dumps(record.get("params", {}), ensure_ascii=False)]
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT INTO generations ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", values)

    def get(self, record_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM generations WHERE id = ?", (record_id,)).fetchone()
        return _to_dict(row) if row else None

    def search(self, kind: str | None = None, language: str | None = None, text: str | None = None,
               date_from: str | None = None, date_to: str | None = None, limit: int = 200) -> list[dict]:
        """Newest first. `date_from` / `date_to` are ISO dates (inclusive)."""
        where, args = [], []
        if kind:
            where.append("kind = ?")
            args.append(kind)
        if language:
            where.append("language = ?")
            args.append(language)
        if text:
            where.append("text LIKE ?")
            args.append(f"%{text}%")
        if date_from:
            where.append("created_at >= ?")
            args.append(date_from)
        if date_to:
            where.append("created_at < date(?, '+1 day')")
            args.append(date_to)
        sql = "SELECT * FROM generations"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, [*args, limit]).fetchall()
        return [_to_dict(r) for r in rows]

    def delete(self, record_id: str) -> bool:
        """Delete the record and its WAV file. Reference audio is shared between records and kept."""
        rec = self.get(record_id)
        if rec is None:
            return False
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM generations WHERE id = ?", (record_id,))
        Path(rec["audio_path"]).unlink(missing_ok=True)
        return True


def _to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["params"] = json.loads(d["params"])
    for k in ("x_vector_only", "normalize"):
        if d[k] is not None:
            d[k] = bool(d[k])
    return d
