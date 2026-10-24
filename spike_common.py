"""Shared helpers for the Qwen3-TTS spike scripts: model loading, metrics, WAV output."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

OUTPUT_DIR = Path(__file__).parent / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.jsonl"

MODEL_IDS = {
    ("base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ("base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ("voice_design", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    ("custom_voice", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ("custom_voice", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}

GiB = 2**30

_WARMUP_TEXT = {"japanese": "テストです。", "chinese": "测试。"}


def warmup_text(language: str) -> str:
    return _WARMUP_TEXT.get(language.lower(), "Test.")


def resolve_attn(attn: str) -> str:
    if attn != "auto":
        return attn
    return "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"


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


GEN_KEYS = ["do_sample", "temperature", "top_k", "top_p", "repetition_penalty",
            "subtalker_dosample", "subtalker_temperature", "subtalker_top_k", "subtalker_top_p",
            "max_new_tokens"]


def gen_kwargs(args: argparse.Namespace) -> dict:
    return {k: getattr(args, k) for k in GEN_KEYS if getattr(args, k) is not None}


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vram_snapshot() -> dict:
    """PyTorch allocator stats plus whole-device usage (includes CUDA context and other processes)."""
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": torch.cuda.memory_allocated() / GiB,
        "reserved_gib": torch.cuda.memory_reserved() / GiB,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GiB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GiB,
        "device_used_gib": (total - free) / GiB,
        "device_total_gib": total / GiB,
    }


def fmt_vram(v: dict) -> str:
    return (f"alloc {v['allocated_gib']:.2f} / reserved {v['reserved_gib']:.2f} GiB "
            f"(peak alloc {v['peak_allocated_gib']:.2f}, peak reserved {v['peak_reserved_gib']:.2f}) | "
            f"device used {v['device_used_gib']:.2f} / {v['device_total_gib']:.2f} GiB")


@contextmanager
def timed():
    torch.cuda.synchronize()
    t = {"start": time.perf_counter()}
    yield t
    torch.cuda.synchronize()
    t["sec"] = time.perf_counter() - t["start"]


def load_model(model_id: str, attn: str = "sdpa"):
    from qwen_tts import Qwen3TTSModel

    attn = resolve_attn(attn)
    torch.cuda.reset_peak_memory_stats()
    with timed() as t:
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation=attn,
        )
    vram = vram_snapshot()
    print(f"[load] {model_id} attn={attn}: {t['sec']:.2f}s | {fmt_vram(vram)}")
    return model, {"model_id": model_id, "attn": attn, "load_sec": t["sec"], "load_vram": vram}


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
