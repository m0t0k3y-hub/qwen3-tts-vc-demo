"""Generation service used by the UI: validates a request, runs it on the GPU and records the result."""

from __future__ import annotations

import hashlib
import json
import random
import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import soundfile as sf
import torch

from tts_core import (GEN_DEFAULTS, GEN_KEYS, LANGUAGES, MODEL_IDS, NON_STREAMING_DEFAULTS, SPEAKERS,
                      seed_everything, timed, vram_snapshot, warmup_text)

from .audio import loudness_normalize, store_by_hash
from .config import Settings
from .history import HistoryStore
from .model_manager import LoadedModel, ModelManager
from .speakers import SpeakerStore

KINDS = ["base", "voice_design", "custom_voice"]
_PROMPT_CACHE_SIZE = 32
# Rough speaking rates for the time estimate (characters per second of audio).
_CHARS_PER_SEC_CJK = 7.0
_CHARS_PER_SEC_OTHER = 14.0
_DEFAULT_RTF = 1.5

Progress = Callable[[str], None]


@dataclass
class GenRequest:
    kind: str
    text: str
    size: str = "1.7B"
    attn: str = "sdpa"
    language: str = "auto"
    seed: int | None = None
    gen: dict = field(default_factory=dict)  # subset of GEN_KEYS; missing keys use GEN_DEFAULTS
    non_streaming_mode: bool | None = None  # None: the method's default
    normalize: bool = True
    # base
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only: bool = False
    # voice_design / custom_voice
    instruct: str | None = None
    # custom_voice
    speaker: str | None = None


@dataclass
class GenResult:
    id: str
    audio_path: Path
    sample_rate: int
    seed: int
    synth_sec: float
    audio_sec: float
    rtf: float
    peak_reserved_gib: float
    load_sec: float | None  # None: the model was already loaded
    warnings: list[str]


class Engine:
    def __init__(self, settings: Settings, models: ModelManager, history: HistoryStore,
                 speakers: SpeakerStore | None = None):
        self.settings = settings
        self.models = models
        self.history = history
        self.speakers = speakers
        self._gpu_lock = threading.Lock()
        self._prompts: OrderedDict[tuple, list] = OrderedDict()
        self._last_rtf: float | None = None

    # ---- validation -------------------------------------------------------------------------

    def validate(self, req: GenRequest) -> list[str]:
        """Raise ValueError for requests that cannot run; return warnings for ones that can."""
        warnings = []
        if req.kind not in KINDS or (req.kind, req.size) not in MODEL_IDS:
            raise ValueError(f"このモデルはありません: {req.kind} {req.size}")
        if not req.text.strip():
            raise ValueError("合成するテキストを入力してください。")
        if len(req.text) > self.settings.max_chars:
            raise ValueError(f"テキストが長すぎます（{len(req.text)} 文字。上限は {self.settings.max_chars} 文字）。")
        if req.language.lower() not in LANGUAGES:
            raise ValueError(f"対応していない言語です: {req.language}")
        unknown = set(req.gen) - set(GEN_KEYS)
        if unknown:
            raise ValueError(f"不明な生成パラメータ: {sorted(unknown)}")
        if req.kind == "base":
            if not req.ref_audio:
                raise ValueError("参照音声を指定してください。")
            if not req.x_vector_only and not (req.ref_text or "").strip():
                raise ValueError("参照音声の書き起こしを入力してください（x-vector only モードでは不要）。")
        if req.kind == "custom_voice":
            if (req.speaker or "").lower() not in SPEAKERS:
                raise ValueError(f"話者を選んでください: {req.speaker}")
            if req.size == "0.6B" and (req.instruct or "").strip():
                warnings.append("0.6B の CustomVoice はスタイル指示に対応していないため、指示は無視されました。")
        return warnings

    def estimate_seconds(self, text: str) -> float:
        """Rough wall time of a generation (audio length from the character count x recent RTF)."""
        cjk = sum(1 for c in text if ord(c) >= 0x3000)
        audio_sec = cjk / _CHARS_PER_SEC_CJK + (len(text) - cjk) / _CHARS_PER_SEC_OTHER
        return audio_sec * (self._last_rtf or _DEFAULT_RTF)

    # ---- generation -------------------------------------------------------------------------

    def generate(self, req: GenRequest, client_ip: str | None = None, progress: Progress | None = None,
                 record: bool = True, out_path: Path | None = None) -> GenResult:
        """`record=False` skips history and metrics.jsonl (used for speaker samples); `out_path` overrides
        where the WAV is written."""
        progress = progress or (lambda msg: None)
        warnings = self.validate(req)
        req = GenRequest(**asdict(req))  # don't mutate the caller's request
        req.language = req.language.lower()
        if req.kind == "custom_voice":
            req.speaker = req.speaker.lower()
            if req.size == "0.6B":
                req.instruct = None
        if req.kind == "base":
            ref_path, _ = store_by_hash(req.ref_audio, self.settings.refs_dir)
            req.ref_audio = str(ref_path)
            if req.x_vector_only:
                req.ref_text = None
            voice = self.speakers and self.speakers.find_by_ref(req.ref_audio, req.ref_text, req.x_vector_only)
            req.speaker = voice["name"] if voice else None  # shown in history as the library voice name
        seed = req.seed if req.seed is not None else random.randrange(2**31)
        gen = {**GEN_DEFAULTS, **req.gen}
        non_streaming = (NON_STREAMING_DEFAULTS[req.kind] if req.non_streaming_mode is None
                         else req.non_streaming_mode)

        run = dict(req=req, seed=seed, gen=gen, non_streaming=non_streaming, warnings=warnings,
                   client_ip=client_ip, progress=progress, record=record, out_path=out_path)
        with self._gpu_lock:
            try:
                return self._run(**run)
            except torch.cuda.OutOfMemoryError:
                # Free every other model and retry once (M-05).
                progress("VRAM が不足したため、他のモデルを解放して再試行しています…")
                self.models.unload_all(keep=(req.kind, req.size, req.attn))
                self._prompts.clear()
                return self._run(**run)

    def unload(self, key=None) -> None:
        """Unload one model (or all with key=None) without racing a running generation."""
        with self._gpu_lock:
            if key is None:
                self.models.unload_all()
            else:
                self.models.unload(key)

    def _run(self, req: GenRequest, seed: int, gen: dict, non_streaming: bool, warnings: list[str],
             client_ip: str | None, progress: Progress, record: bool, out_path: Path | None) -> GenResult:
        def on_load(model_id: str):
            progress(f"モデルをロードしています: {model_id}（初回はダウンロードに数分かかります）")

        entry, loaded_now = self.models.get(req.kind, req.size, req.attn, on_load=on_load)
        fn, kwargs = self._call_args(entry, req, non_streaming, progress)

        if not entry.warmed:
            progress("ウォームアップ中…")
            fn(**{**kwargs, "text": warmup_text(req.language)}, **gen)
            entry.warmed = True

        progress(f"生成中…（推定 約 {self.estimate_seconds(req.text):.0f} 秒）")
        seed_everything(seed)
        torch.cuda.reset_peak_memory_stats()
        with timed() as t:
            wavs, sr = fn(**kwargs, **gen)
        vram = vram_snapshot()
        wav = wavs[0]
        audio_sec = len(wav) / sr
        rtf = t["sec"] / audio_sec if audio_sec > 0 else float("inf")
        self._last_rtf = rtf if audio_sec > 0 else self._last_rtf
        if req.normalize:
            wav = loudness_normalize(wav, sr, self.settings.target_lufs)

        record_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        audio_path = out_path or self.settings.audio_dir / f"{record_id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, wav, sr)

        result = GenResult(
            id=record_id, audio_path=audio_path, sample_rate=sr, seed=seed, synth_sec=t["sec"],
            audio_sec=audio_sec, rtf=rtf, peak_reserved_gib=vram["peak_reserved_gib"],
            load_sec=entry.load_sec if loaded_now else None, warnings=warnings)
        if record:
            self._record(req, gen, non_streaming, result, vram, client_ip)
        return result

    def _call_args(self, entry: LoadedModel, req: GenRequest, non_streaming: bool,
                   progress: Progress) -> tuple[Callable, dict]:
        model = entry.model
        common = dict(text=req.text, language=req.language, non_streaming_mode=non_streaming)
        if req.kind == "base":
            key = (entry.key, req.ref_audio, req.ref_text, req.x_vector_only)
            prompt = self._prompts.get(key)
            if prompt is None:
                path = self._prompt_path(req)
                if path.exists():
                    prompt = torch.load(path, map_location=model.device, weights_only=False)
                else:
                    progress("参照音声を解析しています…")
                    prompt = model.create_voice_clone_prompt(
                        ref_audio=req.ref_audio, ref_text=req.ref_text, x_vector_only_mode=req.x_vector_only)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(prompt, path)
                self._prompts[key] = prompt
                while len(self._prompts) > _PROMPT_CACHE_SIZE:
                    self._prompts.popitem(last=False)
            else:
                self._prompts.move_to_end(key)
            return model.generate_voice_clone, {**common, "voice_clone_prompt": prompt}
        if req.kind == "voice_design":
            return model.generate_voice_design, {**common, "instruct": req.instruct or ""}
        return model.generate_custom_voice, {**common, "speaker": req.speaker, "instruct": req.instruct}

    def _prompt_path(self, req: GenRequest) -> Path:
        """Disk cache of a clone prompt. Keyed by model size because prompts may not be shared across sizes (F-52);
        reference files are content-addressed, so the file name identifies the audio."""
        key = json.dumps([req.size, Path(req.ref_audio).name, req.ref_text, req.x_vector_only], ensure_ascii=False)
        return self.settings.prompts_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.pt"

    # ---- speaker library ----------------------------------------------------------------------

    def save_voice(self, record_id: str, name: str, memo: str = "") -> dict:
        """Save the voice of a clone or VoiceDesign generation to the speaker library (F-27, F-32).

        A VoiceDesign result becomes a clone reference: its audio and text are the reference audio and transcript."""
        rec = self.history.get(record_id)
        if rec is None:
            raise ValueError("保存する生成結果が見つかりません。先に生成してください。")
        if rec["kind"] == "base":
            return self.speakers.add(name, "clone", rec["ref_audio"], rec["ref_text"], rec["x_vector_only"], memo)
        if rec["kind"] == "voice_design":
            p = rec["params"]
            design = {"instruct": rec["instruct"], "language": rec["language"], "seed": rec["seed"],
                      "gen": p["gen"], "history_id": rec["id"]}
            return self.speakers.add(name, "voice_design", rec["audio_path"], rec["text"], False, memo, design)
        raise ValueError("CustomVoice の結果は話者ライブラリに保存できません。")

    def _record(self, req: GenRequest, gen: dict, non_streaming: bool, result: GenResult, vram: dict,
                client_ip: str | None) -> None:
        params = {**asdict(req), "gen": gen, "non_streaming_mode": non_streaming, "seed": result.seed,
                  "target_lufs": self.settings.target_lufs if req.normalize else None}
        self.history.add({
            "id": result.id, "created_at": datetime.now().isoformat(timespec="seconds"),
            "kind": req.kind, "size": req.size, "attn": req.attn, "language": req.language, "text": req.text,
            "speaker": req.speaker, "instruct": req.instruct, "ref_audio": req.ref_audio, "ref_text": req.ref_text,
            "x_vector_only": req.x_vector_only if req.kind == "base" else None, "seed": result.seed,
            "normalize": req.normalize, "audio_path": str(result.audio_path), "synth_sec": result.synth_sec,
            "audio_sec": result.audio_sec, "rtf": result.rtf, "peak_reserved_gib": result.peak_reserved_gib,
            "load_sec": result.load_sec, "client_ip": client_ip, "params": params,
        })
        # Same shape as the spike scripts' records so summarize_metrics.py can read it (N-41).
        metrics = {
            "time": datetime.now().isoformat(timespec="seconds"), "script": f"playground_{req.kind}",
            "size": req.size, "attn": req.attn, "language": req.language, "text": req.text,
            "gen_kwargs": gen, "seed": result.seed, "out": str(result.audio_path),
            "model_id": MODEL_IDS[(req.kind, req.size)], "load_sec": result.load_sec or 0.0,
            "synth_sec": result.synth_sec, "audio_sec": result.audio_sec, "rtf": result.rtf,
            "sample_rate": result.sample_rate, "synth_vram": vram,
        }
        self.settings.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.settings.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
