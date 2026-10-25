"""Comparison tab: the same text under up to four conditions, side by side (F-60 .. F-64)."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from tts_core import GEN_DEFAULTS

from .engine import Engine, GenRequest
from .ui_common import (GPU, KIND_LABELS, LANGUAGE_CHOICES, SPEAKER_CHOICES, example_text_picker, format_result,
                        speaker_label, voice_choices)

N_COLUMNS = 4
# The generation parameters offered per column; the others stay at their defaults.
COMPARE_PARAMS = ["temperature", "top_k", "top_p", "repetition_penalty", "subtalker_temperature"]
FIELDS = ["kind", "size", "voice", "speaker", "instruct", "seed", "normalize", *COMPARE_PARAMS]


@dataclass
class CompareTab:
    voices: list[gr.Dropdown]  # speaker library pickers, refreshed when the tab opens


def _column(i: int, engine: Engine) -> tuple[gr.Checkbox | None, dict, gr.Audio, gr.Markdown, gr.Button | None]:
    gr.Markdown(f"### 条件 {i + 1}")
    enabled = None if i == 0 else gr.Checkbox(False, label="この条件を使う")
    copy = None if i == 0 else gr.Button("条件 1 を複製", size="sm")
    c = {
        "kind": gr.Dropdown([(v, k) for k, v in KIND_LABELS.items()], value="custom_voice", label="モデル"),
        "size": gr.Radio(["1.7B", "0.6B"], value="1.7B", label="サイズ"),
        "voice": gr.Dropdown(voice_choices(engine.speakers), value=None, label="話者（ライブラリ）", visible=False),
        "speaker": gr.Dropdown(SPEAKER_CHOICES, value="ono_anna", label="話者"),
        "instruct": gr.Textbox(label="説明 / スタイル指示", lines=2),
        "seed": gr.Number(0, label="seed", precision=0, minimum=0, info="空欄ならランダム"),
        "normalize": gr.Checkbox(True, label="音量を正規化"),
        "temperature": gr.Slider(0.05, 2.0, GEN_DEFAULTS["temperature"], step=0.05, label="temperature"),
        "top_k": gr.Slider(0, 200, GEN_DEFAULTS["top_k"], step=1, label="top_k"),
        "top_p": gr.Slider(0.05, 1.0, GEN_DEFAULTS["top_p"], step=0.01, label="top_p"),
        "repetition_penalty": gr.Slider(1.0, 2.0, GEN_DEFAULTS["repetition_penalty"], step=0.01,
                                        label="repetition_penalty"),
        "subtalker_temperature": gr.Slider(0.05, 2.0, GEN_DEFAULTS["subtalker_temperature"], step=0.05,
                                           label="subtalker_temperature"),
    }
    audio = gr.Audio(label=f"結果 {i + 1}", type="filepath", interactive=False, buttons=["download"])
    md = gr.Markdown()

    def on_kind(kind):
        return (gr.Radio(value="1.7B", interactive=False) if kind == "voice_design" else gr.Radio(interactive=True),
                gr.Dropdown(visible=kind == "base"), gr.Dropdown(visible=kind == "custom_voice"),
                gr.Textbox(visible=kind != "base"))

    c["kind"].change(on_kind, c["kind"], [c["size"], c["voice"], c["speaker"], c["instruct"]], queue=False)
    return enabled, c, audio, md, copy


def compare_tab(engine: Engine) -> CompareTab:
    gr.Markdown("同じテキストを、条件を変えて最大 4 件生成し、聴き比べます。"
                "生成パラメータは主なものだけを変えられます（他は既定値）。条件 2〜4 の結果には、条件 1 との違いを表示します。")
    text = gr.Textbox(label="合成するテキスト", lines=3, info=f"上限 {engine.settings.max_chars} 文字")
    with gr.Row():
        language = gr.Dropdown(LANGUAGE_CHOICES, value="auto", label="言語")
        example_text_picker(text, language)
    with gr.Row():
        run = gr.Button("生成", variant="primary", scale=3)
        sweep = gr.Button("seed だけ変えて 4 件", scale=1)
        cancel = gr.Button("キャンセル", scale=1)

    cols, enabled, audios, mds = [], [], [], []
    with gr.Row(equal_height=False):
        for i in range(N_COLUMNS):
            with gr.Column(min_width=240):
                en, c, audio, md, copy = _column(i, engine)
            cols.append(c)
            enabled.append(en)
            audios.append(audio)
            mds.append(md)
            if copy is not None:
                copy.click(lambda *vals: [*vals, True], [cols[0][f] for f in FIELDS], [c[f] for f in FIELDS] + [en],
                           queue=False)

    def do_sweep(*vals):
        base = dict(zip(FIELDS, vals))
        seed = int(base["seed"] or 0)
        out = [seed]
        for i in range(1, N_COLUMNS):
            out += [*(seed + i if f == "seed" else base[f] for f in FIELDS), True]
        return out

    sweep.click(do_sweep, [cols[0][f] for f in FIELDS],
                [cols[0]["seed"]] + [x for i in range(1, N_COLUMNS) for x in [*(cols[i][f] for f in FIELDS), enabled[i]]],
                queue=False)

    def describe(v: dict) -> dict[str, str]:
        d = {"モデル": KIND_LABELS[v["kind"]], "サイズ": v["size"]}
        if v["kind"] == "base":
            spk = engine.speakers.get(v["voice"]) if v["voice"] is not None else None
            d["話者"] = spk["name"] if spk else "（未選択）"
        if v["kind"] == "custom_voice":
            d["話者"] = speaker_label(v["speaker"])
        if v["kind"] != "base":
            d["指示"] = v["instruct"] or "（なし）"
        d["seed"] = "ランダム" if v["seed"] is None else str(int(v["seed"]))
        d["正規化"] = "あり" if v["normalize"] else "なし"
        d.update({k: f"{v[k]:g}" for k in COMPARE_PARAMS})
        return d

    def build_request(v: dict, text: str, language: str) -> GenRequest:
        req = GenRequest(kind=v["kind"], text=text, size=v["size"], language=language,
                         seed=None if v["seed"] is None else int(v["seed"]), normalize=v["normalize"],
                         gen={k: v[k] for k in COMPARE_PARAMS})
        if v["kind"] == "base":
            spk = engine.speakers.get(v["voice"]) if v["voice"] is not None else None
            if spk is None:
                raise ValueError("クローンの話者を話者ライブラリから選んでください。")
            req.ref_audio, req.ref_text, req.x_vector_only = spk["ref_audio"], spk["ref_text"], spk["x_vector_only"]
        elif v["kind"] == "custom_voice":
            req.speaker, req.instruct = v["speaker"], v["instruct"]
        else:
            req.instruct = v["instruct"]
        return req

    def make_handler(i: int):
        def handler(data: dict, request: gr.Request, progress=gr.Progress()):
            if i > 0 and not data[enabled[i]]:
                return None, ""
            vals = {f: data[cols[i][f]] for f in FIELDS}
            try:
                req = build_request(vals, data[text], data[language])
                result = engine.generate(req, client_ip=request.client.host if request and request.client else None,
                                         progress=lambda msg: progress(None, desc=f"条件 {i + 1}: {msg}"))
            except ValueError as e:
                return None, f"⚠️ {e}"
            md = format_result(result)
            if i > 0:
                ref = describe({f: data[cols[0][f]] for f in FIELDS})
                diff = [f"**{k}**: {ref.get(k, '—')} → {v}" for k, v in describe(vals).items() if ref.get(k) != v]
                md += "\n\n条件 1 との違い: " + ("、".join(diff) if diff else "なし")
            return str(result.audio_path), md

        return handler

    inputs = [{text, language, *cols[0].values(), *cols[i].values()} | ({enabled[i]} if i else set())
              for i in range(N_COLUMNS)]
    event = run.click(lambda: [None, ""] * N_COLUMNS, None, [x for pair in zip(audios, mds) for x in pair],
                      queue=False)
    events = []
    for i in range(N_COLUMNS):
        event = event.then(make_handler(i), inputs[i], [audios[i], mds[i]], **GPU)
        events.append(event)
    cancel.click(None, cancels=events)
    return CompareTab([c["voice"] for c in cols])
