"""Speaker library shared by all users: saved voices (reference audio + transcript) for voice cloning."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from . import messages as M
from .audio import store_by_hash
from .messages import UserError

SOURCES = {"clone": "クローン (Clone)", "voice_design": "VoiceDesign"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    memo TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    ref_audio TEXT NOT NULL,
    ref_text TEXT,
    x_vector_only INTEGER NOT NULL,
    design TEXT,
    created_at TEXT NOT NULL
);
"""


class SpeakerStore:
    """The reference audio and transcript are the source of truth (F-52); clone prompts are only a cache."""

    def __init__(self, db_path: Path, refs_dir: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.refs_dir = refs_dir
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def add(self, name: str, source: str, ref_audio: str | Path, ref_text: str | None, x_vector_only: bool,
            memo: str = "", design: dict | None = None) -> dict:
        """`design` holds the VoiceDesign description and generation conditions for source='voice_design'."""
        name = name.strip()
        if not name:
            raise UserError(M.NAME_EMPTY)
        if source not in SOURCES:
            raise ValueError(f"unknown source: {source}")
        if not x_vector_only and not (ref_text or "").strip():
            raise UserError(M.NEEDS_TRANSCRIPT)
        if self.get_by_name(name):
            raise UserError(M.NAME_TAKEN, f" {name}")
        path, _ = store_by_hash(ref_audio, self.refs_dir)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO speakers (name, memo, source, ref_audio, ref_text, x_vector_only, design, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (name, memo.strip(), source, str(path), None if x_vector_only else ref_text.strip(),
                 int(x_vector_only), json.dumps(design, ensure_ascii=False) if design else None,
                 datetime.now().isoformat(timespec="seconds")))
        return self.get(cur.lastrowid)

    def get(self, speaker_id: int) -> dict | None:
        return self._one("SELECT * FROM speakers WHERE id = ?", (speaker_id,))

    def get_by_name(self, name: str) -> dict | None:
        return self._one("SELECT * FROM speakers WHERE name = ?", (name.strip(),))

    def find_by_ref(self, ref_audio: str, ref_text: str | None, x_vector_only: bool) -> dict | None:
        """The saved speaker whose reference matches a generation request, if any (for labelling history)."""
        return self._one(
            "SELECT * FROM speakers WHERE ref_audio = ? AND x_vector_only = ? AND (x_vector_only = 1 OR ref_text = ?)"
            " ORDER BY id LIMIT 1", (ref_audio, int(x_vector_only), (ref_text or "").strip()))

    def list(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM speakers ORDER BY name").fetchall()
        return [_to_dict(r) for r in rows]

    def update(self, speaker_id: int, name: str, memo: str) -> dict:
        name = name.strip()
        if not name:
            raise UserError(M.NAME_EMPTY)
        other = self.get_by_name(name)
        if other and other["id"] != speaker_id:
            raise UserError(M.NAME_TAKEN, f" {name}")
        with self._lock, self._conn:
            self._conn.execute("UPDATE speakers SET name = ?, memo = ? WHERE id = ?", (name, memo.strip(), speaker_id))
        return self.get(speaker_id)

    def delete(self, speaker_id: int) -> bool:
        """The reference audio file is kept: history records may point to it."""
        with self._lock, self._conn:
            return self._conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,)).rowcount > 0

    def _one(self, sql: str, args: tuple) -> dict | None:
        with self._lock:
            row = self._conn.execute(sql, args).fetchone()
        return _to_dict(row) if row else None


def _to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["x_vector_only"] = bool(d["x_vector_only"])
    d["design"] = json.loads(d["design"]) if d["design"] else None
    return d
