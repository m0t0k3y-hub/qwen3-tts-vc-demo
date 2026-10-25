"""Comparison tab: the same text under up to four conditions, side by side (F-60 .. F-64)."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from tts_core import GEN_DEFAULTS

from .engine import Engine, GenRequest
from .i18n import t, tm
from .messages import UserError, msg
from .ui_common import (GPU, KIND_LABELS, LANGUAGE_CHOICES, SPEAKER_CHOICES, cancel_fn, client_ip,
                        example_text_picker, format_result, parse_seed, progress_fn, seed_box, speaker_label,
                        voice_choices)

N_COLUMNS = 4
# The generation parameters offered per column; the others stay at their defaults.
COMPARE_PARAMS = ["temperature", "top_k", "top_p", "repetition_penalty", "subtalker_temperature"]
FIELDS = ["kind", "size", "voice", "speaker", "instruct", "seed", "normalize", *COMPARE_PARAMS]
PICK_VOICE = msg("クローンの話者を話者ライブラリから選んでください。", "Pick a voice from the speaker library for cloning.")
DIFF = t("条件 1 との違い:", "Differences from condition 1:")
NO_DIFF = t("条件 1 との違い: なし", "Differences from condition 1: none")
CONDITION_LABELS = [t(f"条件 {i + 1}", f"Condition {i + 1}") for i in range(N_COLUMNS)]
RESULT_LABELS = [t(f"結果 {i + 1}", f"Result {i + 1}") for i in range(N_COLUMNS)]


@dataclass
class CompareTab:
    voices: list[gr.Dropdown]  # speaker library pickers, refreshed when the tab opens
    audios: list[gr.Audio]  # result slots, also filled from the history (F-74)
    results: list[gr.Markdown]


def _column(i: int, engine: Engine) -> tuple[gr.Checkbox | None, dict, gr.Audio, gr.Markdown, gr.Button | None]:
    gr.Markdown("### " + CONDITION_LABELS[i])
    enabled = None if i == 0 else gr.Checkbox(False, label=t("この条件を使う", "Use this condition"))
    copy = None if i == 0 else gr.Button(t("条件 1 を複製", "Copy condition 1"), size="sm")
    c = {
        "kind": gr.Dropdown([(v, k) for k, v in KIND_LABELS.items()], value="custom_voice", label=t("モデル", "Model")),
        "size": gr.Radio(["1.7B", "0.6B"], value="1.7B", label=t("サイズ", "Size")),
        "voice": gr.Dropdown(voice_choices(engine.speakers), value=None,
                             label=t("話者（ライブラリ）", "Voice (library)"), visible=False),
        "speaker": gr.Dropdown(SPEAKER_CHOICES, value="ono_anna", label=t("話者", "Speaker")),
        "instruct": gr.Textbox(label=t("説明 / スタイル指示", "Description / style instruction"), lines=2),
        "seed": seed_box("0"),
        "normalize": gr.Checkbox(True, label=t("音量を正規化", "Normalize loudness")),
        "temperature": gr.Slider(0.05, 2.0, GEN_DEFAULTS["temperature"], step=0.05, label="temperature"),
        "top_k": gr.Slider(0, 200, GEN_DEFAULTS["top_k"], step=1, label="top_k"),
        "top_p": gr.Slider(0.05, 1.0, GEN_DEFAULTS["top_p"], step=0.01, label="top_p"),
        "repetition_penalty": gr.Slider(1.0, 2.0, GEN_DEFAULTS["repetition_penalty"], step=0.01,
                                        label="repetition_penalty"),
        "subtalker_temperature": gr.Slider(0.05, 2.0, GEN_DEFAULTS["subtalker_temperature"], step=0.05,
                                           label="subtalker_temperature"),
    }
    audio = gr.Audio(label=RESULT_LABELS[i], type="filepath", interactive=False, buttons=["download"])
    md = gr.Markdown()

    def on_kind(kind):
        return (gr.Radio(value="1.7B", interactive=False) if kind == "voice_design" else gr.Radio(interactive=True),
                gr.Dropdown(visible=kind == "base"), gr.Dropdown(visible=kind == "custom_voice"),
                gr.Textbox(visible=kind != "base"))

    c["kind"].change(on_kind, c["kind"], [c["size"], c["voice"], c["speaker"], c["instruct"]], queue=False)
    return enabled, c, audio, md, copy


def compare_tab(engine: Engine) -> CompareTab:
    gr.Markdown(t("同じテキストを、条件を変えて最大 4 件生成し、聴き比べます。"
                  "生成パラメータは主なものだけを変えられます（他は既定値）。条件 2〜4 の結果には、条件 1 との違いを表示します。",
                  "Generate the same text under up to 4 conditions and compare them. Only the main generation "
                  "parameters can be changed (the rest use defaults). Results 2-4 show how they differ from condition 1."))
    max_chars = engine.settings.max_chars
    text = gr.Textbox(label=t("合成するテキスト", "Text to synthesize"), lines=3,
                      info=t(f"上限 {max_chars} 文字", f"Up to {max_chars} characters"))
    with gr.Row():
        language = gr.Dropdown(LANGUAGE_CHOICES, value="auto", label=t("言語", "Language"))
        example_text_picker(text, language)
    with gr.Row():
        run = gr.Button(t("生成", "Generate"), variant="primary", scale=3)
        sweep = gr.Button(t("seed だけ変えて 4 件", "4 runs, different seeds only"), scale=1)
        cancel = gr.Button(t("キャンセル", "Cancel"), scale=1)

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
        try:
            seed = parse_seed(base["seed"]) or 0
        except UserError:
            seed = 0
        out = [str(seed)]
        for i in range(1, N_COLUMNS):
            out += [*(str(seed + i) if f == "seed" else base[f] for f in FIELDS), True]
        return out

    sweep.click(do_sweep, [cols[0][f] for f in FIELDS],
                [cols[0]["seed"]] + [x for i in range(1, N_COLUMNS) for x in [*(cols[i][f] for f in FIELDS), enabled[i]]],
                queue=False)

    def describe(v: dict) -> dict[str, str]:
        # Shown after a single translated prefix, so the field names stay as neutral parameter-style names.
        d = {"model": KIND_LABELS[v["kind"]], "size": v["size"]}
        if v["kind"] == "base":
            spk = engine.speakers.get(v["voice"]) if v["voice"] is not None else None
            d["voice"] = spk["name"] if spk else "—"
        if v["kind"] == "custom_voice":
            d["voice"] = speaker_label(v["speaker"])
        if v["kind"] != "base":
            d["instruct"] = v["instruct"] or "—"
        d["seed"] = (v["seed"] or "").strip() or "random"
        d["normalize"] = "on" if v["normalize"] else "off"
        d.update({k: f"{v[k]:g}" for k in COMPARE_PARAMS})
        return d

    def build_request(v: dict, text: str, language: str) -> GenRequest:
        req = GenRequest(kind=v["kind"], text=text, size=v["size"], language=language,
                         seed=parse_seed(v["seed"]), normalize=v["normalize"],
                         gen={k: v[k] for k in COMPARE_PARAMS})
        if v["kind"] == "base":
            spk = engine.speakers.get(v["voice"]) if v["voice"] is not None else None
            if spk is None:
                raise UserError(PICK_VOICE)
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
                result = engine.generate(req, client_ip=client_ip(request),
                                         progress=progress_fn(progress, request, prefix=f"#{i + 1} "),
                                         cancel_token=request.session_hash)
            except UserError as e:
                return None, "⚠️ " + tm(e.msg, e.detail)
            md = format_result(result)
            if i > 0:
                ref = describe({f: data[cols[0][f]] for f in FIELDS})
                diff = [f"**{k}**: {ref.get(k, '—')} → {v}" for k, v in describe(vals).items() if ref.get(k) != v]
                md += "\n\n" + (str(DIFF) + " " + ", ".join(diff) if diff else str(NO_DIFF))
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
    cancel.click(cancel_fn(engine), None, None, cancels=events, queue=False)
    return CompareTab([c["voice"] for c in cols], audios, mds)
