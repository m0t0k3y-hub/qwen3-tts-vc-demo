"""Qwen3-TTS helpers shared by the spike CLI scripts and the playground: model ids, loading, metrics, parameters."""

from __future__ import annotations

import importlib.util
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

MODEL_IDS = {
    ("base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ("base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ("voice_design", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    ("custom_voice", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ("custom_voice", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}

ATTN_CHOICES = ["sdpa", "flash_attention_2", "eager"]

# Values returned by get_supported_languages() / get_supported_speakers() (identical for all 12Hz models;
# checked in the spike). Kept static so the UI can be built without loading a model.
LANGUAGES = ["auto", "chinese", "english", "french", "german", "italian", "japanese", "korean",
             "portuguese", "russian", "spanish"]
SPEAKERS = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]

# Generation kwargs that generate_* actually forwards to the talker (others are silently dropped),
# with the defaults from generation_config.json (the same for all five models).
GEN_KEYS = ["do_sample", "temperature", "top_k", "top_p", "repetition_penalty",
            "subtalker_dosample", "subtalker_temperature", "subtalker_top_k", "subtalker_top_p",
            "max_new_tokens"]
GEN_DEFAULTS = {
    "do_sample": True, "temperature": 0.9, "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.05,
    "subtalker_dosample": True, "subtalker_temperature": 0.9, "subtalker_top_k": 50, "subtalker_top_p": 1.0,
    "max_new_tokens": 8192,
}
# Default of the `non_streaming_mode` argument differs per generate_* method.
NON_STREAMING_DEFAULTS = {"base": False, "voice_design": True, "custom_voice": True}

GiB = 2**30

_WARMUP_TEXT = {"japanese": "テストです。", "chinese": "测试。"}


def warmup_text(language: str) -> str:
    return _WARMUP_TEXT.get(language.lower(), "Test.")


def flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def resolve_attn(attn: str) -> str:
    if attn != "auto":
        return attn
    return "flash_attention_2" if flash_attn_available() else "sdpa"


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
