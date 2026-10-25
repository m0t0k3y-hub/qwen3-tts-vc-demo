"""Audio post-processing and file helpers."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pyloudnorm as pyln

PEAK_LIMIT_DBFS = -1.0
# pyloudnorm needs at least one 400 ms gating block; shorter clips fall back to peak normalization.
_MIN_LOUDNESS_SEC = 0.5


def loudness_normalize(wav: np.ndarray, sr: int, target_lufs: float = -16.0,
                       peak_dbfs: float = PEAK_LIMIT_DBFS) -> np.ndarray:
    """Scale `wav` to `target_lufs` integrated loudness, then pull it down if the peak exceeds `peak_dbfs`."""
    wav = np.asarray(wav, dtype=np.float32)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak == 0.0:
        return wav
    peak_limit = 10 ** (peak_dbfs / 20)

    gain = peak_limit / peak
    if len(wav) / sr >= _MIN_LOUDNESS_SEC:
        loudness = pyln.Meter(sr).integrated_loudness(wav)
        if np.isfinite(loudness):
            gain = min(10 ** ((target_lufs - loudness) / 20), peak_limit / peak)
    return (wav * gain).astype(np.float32)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def store_by_hash(src: str | Path, dest_dir: Path) -> tuple[Path, str]:
    """Copy `src` into `dest_dir` named by its content hash (deduplicated). Returns (path, sha256)."""
    src = Path(src)
    digest = file_sha256(src)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{digest[:16]}{src.suffix.lower()}"
    if not dest.exists():
        shutil.copyfile(src, dest)
    return dest, digest
