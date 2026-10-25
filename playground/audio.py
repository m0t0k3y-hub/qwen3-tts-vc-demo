"""Audio post-processing and file helpers."""

from __future__ import annotations

import hashlib
import io
import shutil
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

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


def trim_silence(wav: np.ndarray, sr: int, top_db: float = 40.0, pad_sec: float = 0.1) -> np.ndarray:
    """Cut leading / trailing parts quieter than `top_db` below the peak, keeping `pad_sec` of margin."""
    _, (start, end) = librosa.effects.trim(wav, top_db=top_db)
    pad = int(pad_sec * sr)
    return wav[max(start - pad, 0): min(end + pad, len(wav))]


def prepare_reference(src: str | Path, dest_dir: Path, trim: bool = False, top_db: float = 40.0,
                      normalize: bool = False, target_lufs: float = -16.0) -> tuple[Path, float]:
    """Pre-process a reference audio (F-21) and store the result content-addressed in `dest_dir`.

    Returns (path, duration in seconds). With no processing the file is stored as is."""
    if not trim and not normalize:
        path, _ = store_by_hash(src, dest_dir)
        return path, float(librosa.get_duration(path=str(path)))
    wav, sr = librosa.load(str(src), sr=None, mono=True)
    if trim:
        wav = trim_silence(wav, sr, top_db)
    if normalize:
        wav = loudness_normalize(wav, sr, target_lufs)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    data = buf.getvalue()
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{hashlib.sha256(data).hexdigest()[:16]}.wav"
    if not path.exists():
        path.write_bytes(data)
    return path, len(wav) / sr


def concat_with_pauses(wavs: list[np.ndarray], sr: int, pause_sec: float) -> np.ndarray:
    """Join generated sentences with `pause_sec` of silence between them (F-04)."""
    pause = np.zeros(int(pause_sec * sr), dtype=np.float32)
    parts = []
    for i, w in enumerate(wavs):
        if i:
            parts.append(pause)
        parts.append(np.asarray(w, dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
