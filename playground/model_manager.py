"""On-demand model loading with LRU eviction under a VRAM budget."""

from __future__ import annotations

import gc
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from tts_core import GiB, MODEL_IDS, load_model

# (kind, size, attn)
ModelKey = tuple[str, str, str]

# Peak reserved VRAM while generating, measured in the spike (README_spike.md). Used until a model
# has been loaded once in this process; after that the measured weight size + GEN_MARGIN_GIB is used.
INITIAL_ESTIMATE_GIB = {
    ("base", "1.7B"): 4.9,
    ("base", "0.6B"): 3.1,
    ("voice_design", "1.7B"): 4.5,
    ("custom_voice", "1.7B"): 4.4,
    ("custom_voice", "0.6B"): 2.6,
    ("asr", "large-v3"): 3.4,  # Whisper large-v3 fp16, measured in phase 3
}
# Activations and KV cache on top of the weights during generation (spike: ~0.5-1.0 GiB).
GEN_MARGIN_GIB = 1.0


@dataclass
class LoadedModel:
    key: ModelKey
    model: Any
    load_sec: float
    weights_gib: float
    last_used: float = field(default_factory=time.time)
    warmed: bool = False


class ModelManager:
    """Keeps loaded models keyed by (kind, size, attn) and evicts the least recently used one
    when loading another would exceed `budget_gib` (sum of per-model estimates).

    `model_ids` maps (kind, size) to a model id; `loaders` overrides the loader per kind (e.g. Whisper)."""

    def __init__(self, budget_gib: float,
                 loader: Callable[[str, str], tuple[Any, dict]] = load_model,
                 allocated_gib: Callable[[], float] = lambda: torch.cuda.memory_allocated() / GiB,
                 model_ids: dict[tuple[str, str], str] = MODEL_IDS,
                 loaders: dict[str, Callable[[str, str], tuple[Any, dict]]] | None = None):
        self.budget_gib = budget_gib
        self.model_ids = model_ids
        self._loader = loader
        self._loaders = loaders or {}
        self._allocated_gib = allocated_gib
        self._models: OrderedDict[ModelKey, LoadedModel] = OrderedDict()  # least recently used first
        self._estimates: dict[tuple[str, str], float] = dict(INITIAL_ESTIMATE_GIB)
        self._lock = threading.RLock()

    def estimate_gib(self, kind: str, size: str) -> float:
        return self._estimates[(kind, size)]

    def get(self, kind: str, size: str, attn: str,
            on_load: Callable[[str], None] | None = None) -> tuple[LoadedModel, bool]:
        """Return the model, loading it if needed. The bool is True if a load happened in this call."""
        if (kind, size) not in self.model_ids:
            raise ValueError(f"unknown model: {kind} {size}")
        key = (kind, size, attn)
        with self._lock:
            if key in self._models:
                entry = self._models[key]
                self._models.move_to_end(key)
                entry.last_used = time.time()
                return entry, False

            self._make_room(self.estimate_gib(kind, size))
            model_id = self.model_ids[(kind, size)]
            if on_load:
                on_load(model_id)
            before = self._allocated_gib()
            model, info = self._loaders.get(kind, self._loader)(model_id, attn)
            weights = max(self._allocated_gib() - before, 0.0)
            if weights > 0:
                self._estimates[(kind, size)] = weights + GEN_MARGIN_GIB
            entry = LoadedModel(key=key, model=model, load_sec=info["load_sec"], weights_gib=weights)
            self._models[key] = entry
            return entry, True

    def _make_room(self, needed_gib: float) -> None:
        while self._models and self._resident_estimate() + needed_gib > self.budget_gib:
            self.unload(next(iter(self._models)))

    def _resident_estimate(self) -> float:
        return sum(self.estimate_gib(k[0], k[1]) for k in self._models)

    def unload(self, key: ModelKey) -> bool:
        with self._lock:
            entry = self._models.pop(key, None)
            if entry is None:
                return False
            print(f"[unload] {self.model_ids[key[:2]]} attn={key[2]}")
            del entry
            _free_cuda()
            return True

    def unload_all(self, keep: ModelKey | None = None) -> None:
        with self._lock:
            for key in [k for k in self._models if k != keep]:
                self.unload(key)

    def loaded(self) -> list[LoadedModel]:
        """Most recently used first."""
        with self._lock:
            return list(reversed(self._models.values()))


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
