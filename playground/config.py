"""Playground settings (set from command-line options in app.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    data_dir: Path = REPO_ROOT / "playground_data"
    # Same file as the spike scripts so summarize_metrics.py covers playground runs too.
    metrics_path: Path = REPO_ROOT / "outputs" / "metrics.jsonl"
    vram_budget_gib: float = 18.0
    max_chars: int = 500
    queue_size: int = 20
    target_lufs: float = -16.0
    preload: list[tuple[str, str]] = field(default_factory=lambda: [("base", "1.7B")])

    @property
    def db_path(self) -> Path:
        return self.data_dir / "playground.sqlite3"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def refs_dir(self) -> Path:
        return self.data_dir / "refs"
