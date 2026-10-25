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

from . import messages as M
from .asr import ASR_KEY, WHISPER_TO_TTS_LANGUAGE
from .audio import concat_with_pauses, loudness_normalize, prepare_reference, store_by_hash
from .config import Settings
from .history import HistoryStore
from .messages import Cancelled, Msg, UserError
from .model_manager import LoadedModel, ModelManager
from .speakers import SpeakerStore
from .text import batch_lines, split_sentences

KINDS = ["base", "voice_design", "custom_voice"]
MAX_BATCH = 8
# Recommended reference length is 5-10 s (F-22); warn a little outside it.
REF_MIN_SEC, REF_MAX_SEC = 4.0, 12.0
_PROMPT_CACHE_SIZE = 32
# Rough speaking rates for the time estimate (characters per second of audio).
_CHARS_PER_SEC_CJK = 7.0
_CHARS_PER_SEC_OTHER = 14.0
_DEFAULT_RTF = 1.5

# progress(message, detail, fraction): `detail` is appended untranslated (numbers, names); `fraction` (0-1)
# is given when the share of work done is known.
Progress = Callable[..., None]
Notice = tuple[Msg, str]


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
    # long text (F-04, F-05): at most one of them
    split: bool = False  # generate sentence by sentence and join with pauses
    pause_sec: float = 0.2
    batch: bool = False  # one text per line, generated in one batched call
    # base
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only: bool = False
    ref_trim_silence: bool = False  # reference pre-processing (F-21)
    ref_silence_db: float = 40.0
    ref_normalize: bool = False
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
    text: str
    synth_sec: float  # for a batch: the whole batched call
    audio_sec: float
    rtf: float
    peak_reserved_gib: float
    load_sec: float | None  # None: the model was already loaded
    warnings: list[Notice]


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
        self._cancelled: set[str] = set()

    # ---- validation -------------------------------------------------------------------------

    def validate(self, req: GenRequest) -> list[Notice]:
        """Raise UserError for requests that cannot run; return warnings for ones that can."""
        warnings = []
        if req.kind not in KINDS or (req.kind, req.size) not in MODEL_IDS:
            raise UserError(M.UNKNOWN_MODEL, f" {req.kind} {req.size}")
        if not req.text.strip():
            raise UserError(M.TEXT_EMPTY)
        if len(req.text) > self.settings.max_chars:
            raise UserError(M.TEXT_TOO_LONG, f" {len(req.text)} / {self.settings.max_chars}")
        if req.split and req.batch:
            raise UserError(M.SPLIT_AND_BATCH)
        if req.batch and len(batch_lines(req.text)) > MAX_BATCH:
            raise UserError(M.TOO_MANY_LINES, f" {len(batch_lines(req.text))} / {MAX_BATCH}")
        if req.language.lower() not in LANGUAGES:
            raise UserError(M.BAD_LANGUAGE, f" {req.language}")
        unknown = set(req.gen) - set(GEN_KEYS)
        if unknown:
            raise UserError(M.BAD_GEN_PARAMS, f" {sorted(unknown)}")
        if req.kind == "base":
            if not req.ref_audio:
                raise UserError(M.NO_REF_AUDIO)
            if not req.x_vector_only and not (req.ref_text or "").strip():
                raise UserError(M.NO_REF_TEXT)
        if req.kind == "custom_voice":
            if (req.speaker or "").lower() not in SPEAKERS:
                raise UserError(M.NO_SPEAKER, f" {req.speaker}")
            if req.size == "0.6B" and (req.instruct or "").strip():
                warnings.append((M.INSTRUCT_IGNORED, ""))
        return warnings

    def estimate_seconds(self, text: str) -> float:
        """Rough wall time of a generation (audio length from the character count x recent RTF)."""
        cjk = sum(1 for c in text if ord(c) >= 0x3000)
        audio_sec = cjk / _CHARS_PER_SEC_CJK + (len(text) - cjk) / _CHARS_PER_SEC_OTHER
        return audio_sec * (self._last_rtf or _DEFAULT_RTF)

    # ---- reference audio (no GPU) -------------------------------------------------------------

    def prepare_reference(self, ref_audio: str, ref_text: str | None, x_vector_only: bool, trim: bool,
                          silence_db: float, normalize: bool) -> tuple[str, float, dict | None, list[Notice]]:
        """Store the original (N-31) and return (processed path, seconds, library voice or None, warnings).

        A reference that is already a library voice is used as is: it was processed when it was saved, and
        processing it again would give a slightly different file (breaking the voice name and prompt cache)."""
        original, _ = store_by_hash(ref_audio, self.settings.refs_dir)
        voice = self.speakers and self.speakers.find_by_ref(str(original), ref_text, x_vector_only)
        if voice:
            trim = normalize = False
        path, sec = prepare_reference(original, self.settings.refs_dir, trim, silence_db, normalize,
                                      self.settings.target_lufs)
        if not voice and self.speakers:
            voice = self.speakers.find_by_ref(str(path), ref_text, x_vector_only)
        warnings = []
        if sec < REF_MIN_SEC:
            warnings.append((M.REF_TOO_SHORT, f" {sec:.1f} s"))
        elif sec > REF_MAX_SEC:
            warnings.append((M.REF_TOO_LONG, f" {sec:.1f} s"))
        return str(path), sec, voice, warnings

    # ---- generation -------------------------------------------------------------------------

    def generate(self, req: GenRequest, client_ip: str | None = None, progress: Progress | None = None,
                 record: bool = True, out_path: Path | None = None, cancel_token: str | None = None) -> GenResult:
        """`record=False` skips history and metrics.jsonl (used for speaker samples); `out_path` overrides
        where the WAV is written. A batch request returns its first result; use generate_batch for all."""
        return self.generate_batch(req, client_ip, progress, record, out_path, cancel_token)[0]

    def generate_batch(self, req: GenRequest, client_ip: str | None = None, progress: Progress | None = None,
                       record: bool = True, out_path: Path | None = None,
                       cancel_token: str | None = None) -> list[GenResult]:
        progress = progress or (lambda m, detail="", fraction=None: None)
        warnings = self.validate(req)
        req = GenRequest(**asdict(req))  # don't mutate the caller's request
        req.language = req.language.lower()
        if req.kind == "custom_voice":
            req.speaker = req.speaker.lower()
            if req.size == "0.6B":
                req.instruct = None
        if req.kind == "base":
            if req.x_vector_only:
                req.ref_text = None
            req.ref_audio, _, voice, ref_warnings = self.prepare_reference(
                req.ref_audio, req.ref_text, req.x_vector_only, req.ref_trim_silence, req.ref_silence_db,
                req.ref_normalize)
            warnings += ref_warnings
            req.speaker = voice["name"] if voice else None  # shown in history as the library voice name
        seed = req.seed if req.seed is not None else random.randrange(2**31)
        gen = {**GEN_DEFAULTS, **req.gen}
        non_streaming = (NON_STREAMING_DEFAULTS[req.kind] if req.non_streaming_mode is None
                         else req.non_streaming_mode)

        if cancel_token:
            self._cancelled.discard(cancel_token)
        run = dict(req=req, seed=seed, gen=gen, non_streaming=non_streaming, warnings=warnings,
                   client_ip=client_ip, progress=progress, record=record, out_path=out_path,
                   cancelled=lambda: cancel_token in self._cancelled)
        with self._gpu_lock:
            try:
                return self._run(**run)
            except torch.cuda.OutOfMemoryError:
                # Free every other model and retry once (M-05).
                progress(M.RETRY_OOM, "")
                self.models.unload_all(keep=(req.kind, req.size, req.attn))
                self._prompts.clear()
                return self._run(**run)
            finally:
                self._cancelled.discard(cancel_token)

    def cancel(self, cancel_token: str) -> None:
        """Ask a running sentence-by-sentence generation to stop before its next sentence."""
        self._cancelled.add(cancel_token)

    def unload(self, key=None) -> None:
        """Unload one model (or all with key=None) without racing a running generation."""
        with self._gpu_lock:
            if key is None:
                self.models.unload_all()
            else:
                self.models.unload(key)

    def transcribe(self, audio_path: str, progress: Progress | None = None) -> tuple[str, str, str | None]:
        """Whisper transcription (F-23). Returns (text, Whisper language code, Qwen3-TTS language or None)."""
        progress = progress or (lambda m, detail="", fraction=None: None)
        if not audio_path:
            raise UserError(M.NO_AUDIO)
        with self._gpu_lock:
            entry, _ = self.models.get(*ASR_KEY, "sdpa", on_load=lambda mid: progress(M.LOADING_MODEL, f" {mid}"))
            progress(M.TRANSCRIBING, "")
            text, code = entry.model.transcribe(audio_path)
        return text, code, WHISPER_TO_TTS_LANGUAGE.get(code)

    def _run(self, req: GenRequest, seed: int, gen: dict, non_streaming: bool, warnings: list[Notice],
             client_ip: str | None, progress: Progress, record: bool, out_path: Path | None,
             cancelled: Callable[[], bool]) -> list[GenResult]:
        entry, loaded_now = self.models.get(req.kind, req.size, req.attn,
                                            on_load=lambda mid: progress(M.LOADING_MODEL, f" {mid}"))
        fn, kwargs = self._call_args(entry, req, non_streaming, progress)

        if not entry.warmed:
            progress(M.WARMING_UP, "")
            fn(**{**kwargs, "text": warmup_text(req.language)}, **gen)
            entry.warmed = True

        texts = batch_lines(req.text) if req.batch else [req.text]
        seed_everything(seed)
        torch.cuda.reset_peak_memory_stats()
        with timed() as t:
            if req.split:
                sentences = split_sentences(req.text)
                pieces = []
                for i, sentence in enumerate(sentences, 1):
                    if cancelled():
                        raise Cancelled()
                    progress(M.GENERATING_SENTENCE, f" {i} / {len(sentences)}", (i - 1) / len(sentences))
                    w, sr = fn(**{**kwargs, "text": sentence}, **gen)
                    pieces.append(w[0])
                wavs = [concat_with_pauses(pieces, sr, req.pause_sec)]
            else:
                progress(M.GENERATING, f" {self.estimate_seconds(req.text):.0f}")
                wavs, sr = fn(**{**kwargs, "text": texts if req.batch else req.text}, **gen)
        vram = vram_snapshot()
        total_audio = sum(len(w) for w in wavs) / sr
        rtf = t["sec"] / total_audio if total_audio > 0 else float("inf")
        if total_audio > 0:
            self._last_rtf = rtf

        results = []
        for i, (text, wav) in enumerate(zip(texts, wavs)):
            if req.normalize:
                wav = loudness_normalize(wav, sr, self.settings.target_lufs)
            record_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
            audio_path = (out_path if out_path and i == 0 else None) or self.settings.audio_dir / f"{record_id}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(audio_path, wav, sr)
            results.append(GenResult(
                id=record_id, audio_path=audio_path, sample_rate=sr, seed=seed, text=text, synth_sec=t["sec"],
                audio_sec=len(wav) / sr, rtf=rtf, peak_reserved_gib=vram["peak_reserved_gib"],
                load_sec=entry.load_sec if loaded_now else None, warnings=warnings))
        if record:
            batch = {"batch_id": results[0].id, "batch_size": len(results)} if req.batch else {}
            for r in results:
                self._record(req, gen, non_streaming, r, vram, client_ip, batch)
        return results

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
                    progress(M.ANALYZING_REF, "")
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

    def save_voice(self, record_id: str, name: str, memo: str = "", ref_audio: str | None = None,
                   ref_text: str | None = None) -> dict:
        """Save the voice of a clone or VoiceDesign generation to the speaker library (F-27, F-32).

        A VoiceDesign result becomes a clone reference: its audio and text are the reference audio and transcript.
        `ref_audio` / `ref_text` replace them, e.g. with a trimmed part of the result (F-33)."""
        rec = self.history.get(record_id) if record_id else None
        if rec is None:
            raise UserError(M.NO_RESULT)
        if rec["kind"] == "base":
            return self.speakers.add(name, "clone", rec["ref_audio"], rec["ref_text"], rec["x_vector_only"], memo)
        if rec["kind"] == "voice_design":
            p = rec["params"]
            design = {"instruct": rec["instruct"], "language": rec["language"], "seed": rec["seed"],
                      "gen": p["gen"], "history_id": rec["id"]}
            return self.speakers.add(name, "voice_design", ref_audio or rec["audio_path"],
                                     rec["text"] if ref_text is None else ref_text, False, memo, design)
        raise UserError(M.CANNOT_SAVE_CUSTOM)

    def _record(self, req: GenRequest, gen: dict, non_streaming: bool, result: GenResult, vram: dict,
                client_ip: str | None, batch: dict) -> None:
        params = {**asdict(req), "text": result.text, "gen": gen, "non_streaming_mode": non_streaming,
                  "seed": result.seed, "target_lufs": self.settings.target_lufs if req.normalize else None, **batch}
        self.history.add({
            "id": result.id, "created_at": datetime.now().isoformat(timespec="seconds"),
            "kind": req.kind, "size": req.size, "attn": req.attn, "language": req.language, "text": result.text,
            "speaker": req.speaker, "instruct": req.instruct, "ref_audio": req.ref_audio, "ref_text": req.ref_text,
            "x_vector_only": req.x_vector_only if req.kind == "base" else None, "seed": result.seed,
            "normalize": req.normalize, "audio_path": str(result.audio_path), "synth_sec": result.synth_sec,
            "audio_sec": result.audio_sec, "rtf": result.rtf, "peak_reserved_gib": result.peak_reserved_gib,
            "load_sec": result.load_sec, "client_ip": client_ip, "params": params,
        })
        # Same shape as the spike scripts' records so summarize_metrics.py can read it (N-41).
        metrics = {
            "time": datetime.now().isoformat(timespec="seconds"), "script": f"playground_{req.kind}",
            "size": req.size, "attn": req.attn, "language": req.language, "text": result.text,
            "gen_kwargs": gen, "seed": result.seed, "out": str(result.audio_path),
            "model_id": MODEL_IDS[(req.kind, req.size)], "load_sec": result.load_sec or 0.0,
            "synth_sec": result.synth_sec, "audio_sec": result.audio_sec, "rtf": result.rtf,
            "sample_rate": result.sample_rate, "synth_vram": vram, **batch,
        }
        self.settings.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.settings.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
