"""Shared helpers for the Qwen3-TTS spike scripts: model loading, metrics, WAV output."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from tts_core import (GEN_KEYS, MODEL_IDS, GiB, fmt_vram, load_model, resolve_attn,  # noqa: F401 (re-exported)
                      seed_everything, timed, vram_snapshot, warmup_text)

OUTPUT_DIR = Path(__file__).parent / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.jsonl"


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--attn", default="sdpa", choices=["auto", "sdpa", "flash_attention_2", "eager"],
                   help="attention implementation (default sdpa: flash_attention_2 measured ~30%% slower here; "
                        "auto: flash_attention_2 if installed)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="output WAV path (default: outputs/<auto>.wav)")
    p.add_argument("--no-warmup", action="store_true", help="skip the untimed warm-up generation")

    g = p.add_argument_group("generation (unset = model's generate_config.json default)")
    g.add_argument("--no-sample", dest="do_sample", action="store_const", const=False, default=None)
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--top-p", type=float)
    g.add_argument("--repetition-penalty", type=float)
    g.add_argument("--subtalker-no-sample", dest="subtalker_dosample", action="store_const", const=False, default=None)
    g.add_argument("--subtalker-temperature", type=float)
    g.add_argument("--subtalker-top-k", type=int)
    g.add_argument("--subtalker-top-p", type=float)
    g.add_argument("--max-new-tokens", type=int)


def gen_kwargs(args: argparse.Namespace) -> dict:
    return {k: getattr(args, k) for k in GEN_KEYS if getattr(args, k) is not None}


def synthesize(fn, warmup_kwargs: dict | None = None, **kwargs):
    """Run `fn(**kwargs)` (a generate_* method) with timing / peak-VRAM measurement.

    If `warmup_kwargs` is given, an untimed short generation runs first so CUDA kernel
    compilation/caching does not inflate the measured RTF.
    """
    if warmup_kwargs is not None:
        fn(**warmup_kwargs)
    torch.cuda.reset_peak_memory_stats()
    with timed() as t:
        wavs, sr = fn(**kwargs)
    vram = vram_snapshot()
    wav = wavs[0]
    audio_sec = len(wav) / sr
    m = {
        "synth_sec": t["sec"],
        "audio_sec": audio_sec,
        "rtf": t["sec"] / audio_sec if audio_sec > 0 else float("inf"),
        "sample_rate": sr,
        "synth_vram": vram,
    }
    print(f"[synth] {t['sec']:.2f}s for {audio_sec:.2f}s audio -> RTF {m['rtf']:.3f} | {fmt_vram(vram)}")
    return wav, sr, m


def save_wav(wav: np.ndarray, sr: int, out: Path | None, stem: str) -> Path:
    if out is None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        out = OUTPUT_DIR / f"{stem}_{datetime.now():%Y%m%d-%H%M%S}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, wav, sr)
    print(f"[save] {out}")
    return out


def log_metrics(record: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    record = {"time": datetime.now().isoformat(timespec="seconds"), **record}
    with METRICS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
