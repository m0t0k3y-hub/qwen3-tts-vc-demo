"""The three model tabs: voice clone, VoiceDesign, CustomVoice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gradio as gr

from . import presets
from .engine import Engine
from .samples import SAMPLE_TEXTS, sample_path
from .ui_common import (SPEAKER_CHOICES, GenControls, OutputArea, gen_controls, instruct_picker, output_area,
                        save_voice_row, voice_choices, wire)


@dataclass
class CloneTab:
    controls: GenControls
    voice: gr.Dropdown  # speaker library picker
    ref_audio: gr.Audio
    ref_text: gr.Textbox
    x_vector_only: gr.Checkbox
    saved: object  # save-to-library event


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
    gr.Markdown("数秒の参照音声と、その書き起こしから声をクローンします（Base モデル）。"
                "参照音声は 5〜10 秒程度、書き起こしは正確なほど良い結果になります。")
    with gr.Row():
        with gr.Column():
            voice = gr.Dropdown(voice_choices(engine.speakers), value=None, label="話者ライブラリから選ぶ",
                                info="選ぶと参照音声と書き起こしが入ります")
            ref_audio = gr.Audio(label="参照音声", sources=["upload"], type="filepath")
            x_vector_only = gr.Checkbox(False, label="x-vector only モード",
                                        info="書き起こしなしで話者埋め込みだけを使う（品質は下がる）")
            ref_text = gr.Textbox(label="参照音声の書き起こし", lines=2)
        with gr.Column():
            controls = gen_controls("base", engine.settings, "参照音声の声で読み上げるテキスト")
    out = output_area()
    saved = save_voice_row(engine, out.last_id, "直前の生成に使った参照音声と書き起こしを、名前を付けて保存します。")

    def pick(voice_id):
        spk = engine.speakers.get(voice_id) if voice_id is not None else None
        if spk is None:
            return gr.skip(), gr.skip(), gr.skip()
        return spk["ref_audio"], spk["ref_text"] or "", spk["x_vector_only"]

    voice.change(pick, voice, [ref_audio, ref_text, x_vector_only], queue=False)
    x_vector_only.change(lambda on: gr.Textbox(interactive=not on), x_vector_only, ref_text, queue=False)
    wire(engine, controls, {"ref_audio": ref_audio, "ref_text": ref_text, "x_vector_only": x_vector_only}, out)
    return CloneTab(controls, voice, ref_audio, ref_text, x_vector_only, saved)


def design_tab(engine: Engine) -> DesignTab:
    gr.Markdown("声の特徴を自然文で説明すると、その声で読み上げます（VoiceDesign 1.7B）。"
                "気に入った声は話者ライブラリに保存すると、クローンで同じ声を何度でも使えます。")
    with gr.Row():
        with gr.Column():
            instruct = gr.Textbox(label="声の説明", lines=4,
                                  placeholder="例: 落ち着いた低めの声の40代男性。ゆっくり丁寧に話す。",
                                  info="年齢、性別、声質、話し方、感情など。空欄でもよい")
            instruct_picker(presets.DESIGN_PRESETS, instruct, "説明の例を挿入")
        with gr.Column():
            controls = gen_controls("voice_design", engine.settings, "デザインした声で読み上げるテキスト")
    out = output_area()
    saved = save_voice_row(
        engine, out.last_id,
        "直前の生成結果の音声とテキストを、クローン用の参照音声と書き起こしとして保存します。"
        "参照音声は 5〜10 秒程度が適しているので、短すぎる・長すぎる場合はテキストを調整して生成し直してください。")
    wire(engine, controls, {"instruct": instruct}, out)
    return DesignTab(controls, instruct, saved)


def custom_tab(engine: Engine) -> CustomTab:
    gr.Markdown("9 人のプリセット話者で読み上げます（CustomVoice）。1.7B はスタイル指示にも対応しています。")
    with gr.Row():
        with gr.Column():
            speaker = gr.Dropdown(SPEAKER_CHOICES, value="ono_anna", label="話者")
            with gr.Row():
                samples = {lang: gr.Audio(label=f"サンプル（{'日本語' if lang == 'japanese' else '英語'}）",
                                          type="filepath", interactive=False)
                           for lang in SAMPLE_TEXTS}
            missing = gr.Markdown(visible=False)
            instruct = gr.Textbox(label="スタイル指示（1.7B のみ）", lines=2,
                                  placeholder="例: 楽しそうに、明るく / Calm and confident, like a news anchor.")
            instruct_picker(presets.STYLE_PRESETS, instruct, "指示の例を挿入")
        with gr.Column():
            controls = gen_controls("custom_voice", engine.settings, "読み上げるテキスト")
    out = output_area()

    def show_samples(spk):
        paths = [sample_path(engine.settings, spk, lang) for lang in SAMPLE_TEXTS]
        have = all(p.exists() for p in paths)
        note = gr.Markdown("サンプルは未生成です。「システム」タブか `uv run playground/samples.py` で生成できます。",
                           visible=not have)
        return *[str(p) if p.exists() else None for p in paths], note

    def on_size(size: str):
        if size == "0.6B":
            return gr.Textbox(interactive=False, info="0.6B はスタイル指示に対応していません")
        return gr.Textbox(interactive=True, info=None)

    sample_outputs = [*samples.values(), missing]
    speaker.change(show_samples, speaker, sample_outputs, queue=False)
    controls.size.change(on_size, controls.size, instruct, queue=False)
    wire(engine, controls, {"speaker": speaker, "instruct": instruct}, out)
    return CustomTab(controls, speaker, instruct, show_samples, sample_outputs)
