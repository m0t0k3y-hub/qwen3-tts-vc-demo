"""The three model tabs: voice clone, VoiceDesign, CustomVoice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gradio as gr

from . import presets
from .engine import Engine
from .i18n import t
from .messages import UserError
from .samples import SAMPLE_TEXTS, sample_path
from .ui_common import (GPU, LANGUAGE_LABELS, SPEAKER_CHOICES, GenControls, gen_controls, instruct_picker,
                        output_area, progress_fn, save_voice_row, show_warnings, user_error, voice_choices,
                        wire)

LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}
DETECTED = t("判定した言語:", "Detected language:")
REF_SECONDS = t("前処理後の長さ（秒）:", "Length after pre-processing (s):")
NO_MIC = t("録音は、ホスト PC から http://localhost でアクセスした場合のみ使えます（ブラウザの制限）。",
           "Recording only works when the host PC opens the app at http://localhost (a browser restriction).")
SAMPLES_MISSING = t("サンプルは未生成です。「システム」タブか `uv run playground/samples.py` で生成できます。",
                    "Samples have not been generated yet. Use the System tab or `uv run playground/samples.py`.")
NO_INSTRUCT_06B = t("0.6B はスタイル指示に対応していません", "0.6B does not support style instructions")


@dataclass
class CloneTab:
    controls: GenControls
    voice: gr.Dropdown  # speaker library picker
    ref_audio: gr.Audio
    ref_text: gr.Textbox
    x_vector_only: gr.Checkbox
    mic_note: gr.Markdown
    saved: object  # save-to-library event

    def enable_recording(self, request: gr.Request):
        """On page load: offer the microphone only to a browser on the host itself (F-20)."""
        host = (request.headers.get("host") or "").rsplit(":", 1)[0] if request else ""
        if host in LOCAL_HOSTS:
            return gr.Audio(sources=["upload", "microphone"]), gr.Markdown(visible=False)
        return gr.skip(), gr.Markdown(visible=True)


@dataclass
class DesignTab:
    controls: GenControls
    instruct: gr.Textbox
    saved: object


@dataclass
class CustomTab:
    controls: GenControls
    speaker: gr.Dropdown
    instruct: gr.Textbox
    show_samples: Callable  # speaker -> values for sample_outputs (to refresh after samples are generated)
    sample_outputs: list


def clone_tab(engine: Engine) -> CloneTab:
    gr.Markdown(t("数秒の参照音声と、その書き起こしから声をクローンします（Base モデル）。"
                  "参照音声は 5〜10 秒程度、書き起こしは正確なほど良い結果になります。",
                  "Clones a voice from a few seconds of reference audio and its transcript (Base model). "
                  "Use 5-10 s of reference audio; the more accurate the transcript, the better."))
    with gr.Row():
        with gr.Column():
            voice = gr.Dropdown(voice_choices(engine.speakers), value=None,
                                label=t("話者ライブラリから選ぶ", "Pick from the speaker library"),
                                info=t("選ぶと参照音声と書き起こしが入ります", "Fills in the reference audio and transcript"))
            ref_audio = gr.Audio(label=t("参照音声（波形の編集ボタンで切り取れます）",
                                         "Reference audio (trim it with the waveform edit button)"),
                                 sources=["upload"], type="filepath", editable=True)
            mic_note = gr.Markdown(NO_MIC, visible=False)
            with gr.Accordion(t("参照音声の前処理", "Reference pre-processing"), open=False):
                with gr.Row():
                    trim = gr.Checkbox(True, label=t("前後の無音を除去", "Trim leading / trailing silence"))
                    silence_db = gr.Slider(20, 60, 40, step=1, label=t("無音とみなすレベル（ピークから -dB）",
                                                                       "Silence threshold (dB below peak)"))
                    ref_normalize = gr.Checkbox(True, label=t("音量を正規化", "Normalize loudness"))
                preview = gr.Button(t("前処理の結果を確認", "Preview the result"), size="sm")
                processed = gr.Audio(label=t("前処理後の参照音声", "Pre-processed reference"), type="filepath",
                                     interactive=False)
                processed_info = gr.Markdown()
            x_vector_only = gr.Checkbox(False, label=t("x-vector only モード", "x-vector only mode"),
                                        info=t("書き起こしなしで話者埋め込みだけを使う（品質は下がる）",
                                               "Use only the speaker embedding, no transcript (lower quality)"))
            ref_text = gr.Textbox(label=t("参照音声の書き起こし", "Transcript of the reference audio"), lines=2)
            with gr.Row():
                transcribe = gr.Button(t("自動で書き起こす（Whisper）", "Transcribe automatically (Whisper)"),
                                       size="sm")
                detected = gr.Markdown()
        with gr.Column():
            controls = gen_controls("base", engine.settings,
                                    t("参照音声の声で読み上げるテキスト", "Text to read in the reference voice"))
    out = output_area()
    saved, _, _ = save_voice_row(engine, out.last_id,
                                 t("直前の生成に使った参照音声（前処理後）と書き起こしを、名前を付けて保存します。",
                                   "Saves the reference audio (after pre-processing) and transcript of the last "
                                   "generation under a name."))

    def pick(voice_id):
        spk = engine.speakers.get(voice_id) if voice_id is not None else None
        if spk is None:
            return gr.skip(), gr.skip(), gr.skip()
        return spk["ref_audio"], spk["ref_text"] or "", spk["x_vector_only"]

    def do_preview(audio, text, xvec, trim, silence_db, normalize, request: gr.Request):
        if not audio:
            return None, ""
        path, sec, _, warnings = engine.prepare_reference(audio, text, xvec, trim, silence_db, normalize)
        show_warnings(warnings, request)
        return path, str(REF_SECONDS) + f" {sec:.2f}"

    def do_transcribe(audio, request: gr.Request, progress=gr.Progress()):
        try:
            text, code, language = engine.transcribe(audio, progress=progress_fn(progress, request))
        except UserError as e:
            raise user_error(e, request) from e
        name = f"{LANGUAGE_LABELS[language]} ({language.capitalize()})" if language else code
        return text, str(DETECTED) + f" {name}"

    voice.change(pick, voice, [ref_audio, ref_text, x_vector_only], queue=False)
    x_vector_only.change(lambda on: (gr.Textbox(interactive=not on), gr.Button(interactive=not on)),
                         x_vector_only, [ref_text, transcribe], queue=False)
    preview.click(do_preview, [ref_audio, ref_text, x_vector_only, trim, silence_db, ref_normalize],
                  [processed, processed_info])
    transcribe.click(do_transcribe, ref_audio, [ref_text, detected], **GPU)
    wire(engine, controls, {"ref_audio": ref_audio, "ref_text": ref_text, "x_vector_only": x_vector_only,
                            "ref_trim_silence": trim, "ref_silence_db": silence_db, "ref_normalize": ref_normalize},
         out)
    return CloneTab(controls, voice, ref_audio, ref_text, x_vector_only, mic_note, saved)


def design_tab(engine: Engine) -> DesignTab:
    gr.Markdown(t("声の特徴を自然文で説明すると、その声で読み上げます（VoiceDesign 1.7B）。"
                  "気に入った声は話者ライブラリに保存すると、クローンで同じ声を何度でも使えます。",
                  "Describe a voice in natural language and it reads the text in that voice (VoiceDesign 1.7B). "
                  "Save a voice you like to the speaker library to reuse it with voice cloning."))
    with gr.Row():
        with gr.Column():
            instruct = gr.Textbox(label=t("声の説明", "Voice description"), lines=4,
                                  placeholder=t("例: 落ち着いた低めの声の40代男性。ゆっくり丁寧に話す。",
                                                "e.g. A calm, low-pitched man in his 40s. Speaks slowly and politely."),
                                  info=t("年齢、性別、声質、話し方、感情など。空欄でもよい",
                                         "Age, gender, timbre, speaking style, emotion… May be empty"))
            instruct_picker(presets.DESIGN_PRESETS, instruct, t("説明の例を挿入", "Insert an example description"))
        with gr.Column():
            controls = gen_controls("voice_design", engine.settings,
                                    t("デザインした声で読み上げるテキスト", "Text to read in the designed voice"))
    out = output_area()
    saved, ref_audio, ref_text = save_voice_row(
        engine, out.last_id,
        t("直前の生成結果の音声とテキストを、クローン用の参照音声と書き起こしとして保存します。"
          "参照音声は 5〜10 秒程度が適しているので、長い場合は切り取ってください。",
          "Saves the last result's audio and text as a reference audio and transcript for cloning. "
          "5-10 s works best, so trim longer results."),
        editable_ref=True)
    wire(engine, controls, {"instruct": instruct}, out, extra_outputs=[ref_audio, ref_text],
         extra_values=lambda results: [str(results[0].audio_path), results[0].text])
    return DesignTab(controls, instruct, saved)


def custom_tab(engine: Engine) -> CustomTab:
    gr.Markdown(t("9 人のプリセット話者で読み上げます（CustomVoice）。1.7B はスタイル指示にも対応しています。",
                  "Reads the text with one of 9 preset speakers (CustomVoice). 1.7B also takes style instructions."))
    with gr.Row():
        with gr.Column():
            speaker = gr.Dropdown(SPEAKER_CHOICES, value="ono_anna", label=t("話者", "Speaker"))
            with gr.Row():
                samples = {"japanese": gr.Audio(label=t("サンプル（日本語）", "Sample (Japanese)"), type="filepath",
                                                interactive=False),
                           "english": gr.Audio(label=t("サンプル（英語）", "Sample (English)"), type="filepath",
                                               interactive=False)}
            missing = gr.Markdown(SAMPLES_MISSING, visible=False)
            instruct = gr.Textbox(label=t("スタイル指示（1.7B のみ）", "Style instruction (1.7B only)"), lines=2,
                                  placeholder=t("例: 楽しそうに、明るく", "e.g. Calm and confident, like a news anchor."))
            instruct_picker(presets.STYLE_PRESETS, instruct, t("指示の例を挿入", "Insert an example instruction"))
        with gr.Column():
            controls = gen_controls("custom_voice", engine.settings, t("読み上げるテキスト", "Text to read"))
    out = output_area()

    def show_samples(spk):
        paths = [sample_path(engine.settings, spk, lang) for lang in SAMPLE_TEXTS]
        return *[str(p) if p.exists() else None for p in paths], gr.Markdown(visible=not all(p.exists() for p in paths))

    def on_size(size: str):
        if size == "0.6B":
            return gr.Textbox(interactive=False, info=NO_INSTRUCT_06B)
        return gr.Textbox(interactive=True, info=None)

    sample_outputs = [*samples.values(), missing]
    speaker.change(show_samples, speaker, sample_outputs, queue=False)
    controls.size.change(on_size, controls.size, instruct, queue=False)
    wire(engine, controls, {"speaker": speaker, "instruct": instruct}, out)
    return CustomTab(controls, speaker, instruct, show_samples, sample_outputs)
