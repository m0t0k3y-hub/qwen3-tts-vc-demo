"""Reference-audio transcription with Whisper large-v3 (F-23).

faster-whisper (CTranslate2) needs the CUDA 12 cuBLAS, which this CUDA 13 environment does not have, so the same
large-v3 weights are run through transformers instead.
"""

from __future__ import annotations

import time

import librosa
import torch

from tts_core import MODEL_IDS

ASR_KEY = ("asr", "large-v3")
ASR_MODEL_ID = "openai/whisper-large-v3"
# Model ids the playground's ModelManager can load: the TTS models plus Whisper.
PLAYGROUND_MODEL_IDS = {**MODEL_IDS, ASR_KEY: ASR_MODEL_ID}

SAMPLE_RATE = 16000
_SHORT_FORM_SEC = 30
# Whisper language codes of the languages Qwen3-TTS supports, mapped to its language names.
WHISPER_TO_TTS_LANGUAGE = {"zh": "chinese", "en": "english", "fr": "french", "de": "german", "it": "italian",
                           "ja": "japanese", "ko": "korean", "pt": "portuguese", "ru": "russian", "es": "spanish"}


class Transcriber:
    def __init__(self, model_id: str = ASR_MODEL_ID, attn: str = "sdpa"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.float16, attn_implementation=attn).to("cuda").eval()

    @torch.inference_mode()
    def transcribe(self, path: str) -> tuple[str, str]:
        """Return (text, Whisper language code such as 'ja')."""
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if len(audio) <= _SHORT_FORM_SEC * SAMPLE_RATE:
            inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            kwargs = {}
        else:  # long-form: sequential 30 s windows, which requires timestamps
            inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", truncation=False,
                                    padding="longest", return_attention_mask=True)
            kwargs = {"attention_mask": inputs.attention_mask.to("cuda"), "return_timestamps": True}
        features = inputs.input_features.to("cuda", torch.float16)

        lang_id = self.model.detect_language(features[:, :, :3000])[0].item()
        code = self.processor.tokenizer.convert_ids_to_tokens(lang_id).strip("<|>")
        ids = self.model.generate(features, task="transcribe", language=code, **kwargs)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return text, code


def load_asr(model_id: str, attn: str = "sdpa") -> tuple[Transcriber, dict]:
    """Loader with the same signature as tts_core.load_model, for ModelManager."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    transcriber = Transcriber(model_id, attn)
    torch.cuda.synchronize()
    sec = time.perf_counter() - start
    print(f"[load] {model_id} attn={attn}: {sec:.2f}s")
    return transcriber, {"load_sec": sec}
