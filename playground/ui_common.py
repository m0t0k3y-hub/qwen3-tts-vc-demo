"""UI parts shared by several tabs: labels, the generation form, the result area."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import gradio as gr

from tts_core import ATTN_CHOICES, GEN_DEFAULTS, LANGUAGES, NON_STREAMING_DEFAULTS, SPEAKERS, flash_attn_available

from . import presets
from .config import Settings
from .engine import MAX_BATCH, Engine, GenRequest, GenResult
from .i18n import say, t
from . import messages as M
from .messages import Cancelled, UserError
from .speakers import SOURCES, SpeakerStore

# Choice labels and table headers are not translated by gr.I18n, so they are bilingual.
KIND_LABELS = {"base": "ボイスクローン (Voice clone)", "voice_design": "VoiceDesign", "custom_voice": "CustomVoice"}
LANGUAGE_LABELS = {
    "auto": "自動判定", "chinese": "中国語", "english": "英語", "french": "フランス語", "german": "ドイツ語",
    "italian": "イタリア語", "japanese": "日本語", "korean": "韓国語", "portuguese": "ポルトガル語",
    "russian": "ロシア語", "spanish": "スペイン語",
}
LANGUAGE_CHOICES = [(f"{LANGUAGE_LABELS[k]} ({k.capitalize()})", k) for k in LANGUAGES]
SPEAKER_CHOICES = [("_".join(w.capitalize() for w in s.split("_")), s) for s in SPEAKERS]
MODE_CHOICES = [("通常 / Normal", "normal"), ("文ごとに分割 / Split into sentences", "split"),
                ("バッチ（1 行 1 テキスト）/ Batch (one text per line)", "batch")]
# All GPU work shares one queue slot (N-10).
GPU = dict(concurrency_limit=1, concurrency_id="gpu")

RESULT_HEADER = t("| 合成時間 | 音声長 | RTF | ピーク VRAM | モデルのロード | seed |",
                  "| Synthesis | Audio | RTF | Peak VRAM | Model load | seed |")


def parse_seed(text) -> int | None:
    """Seed fields are text boxes: gr.Number sends 0 for an empty field, which would make "empty = random"
    impossible."""
    text = str(text or "").strip()
    if not text:
        return None
    if not text.isdigit():
        raise UserError(M.BAD_SEED, f" {text}")
    return int(text)


def seed_box(value: str = "", **kwargs) -> gr.Textbox:
    return gr.Textbox(value, label="seed", placeholder=t("空欄ならランダム", "Empty: random"), **kwargs)


def speaker_label(speaker: str) -> str:
    return {value: label for label, value in SPEAKER_CHOICES}.get(speaker, speaker)


def voice_choices(store: SpeakerStore) -> list[tuple[str, int]]:
    return [(f"{s['name']}（{SOURCES[s['source']]}）", s["id"]) for s in store.list()]


def clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def client_ip(request: gr.Request | None) -> str | None:
    return request.client.host if request and request.client else None


def user_error(e: UserError, request: gr.Request | None) -> gr.Error:
    return gr.Error(say(e.msg, request, e.detail))


def show_warnings(warnings: list, request: gr.Request | None) -> None:
    for m, detail in warnings:
        gr.Warning(say(m, request, detail))


def progress_fn(progress: gr.Progress, request: gr.Request | None, prefix: str = "") -> Callable:
    # Gradio shows the description only with a numeric progress value, so unknown progress is shown as 0.
    return lambda m, detail="", fraction=None: progress(fraction or 0.0, desc=prefix + say(m, request, detail))


@dataclass
class GenControls:
    """The generation form shared by the three model tabs (F-01 .. F-13)."""

    kind: str
    text: gr.Textbox
    language: gr.Dropdown
    mode: gr.Radio
    pause: gr.Slider
    seed: gr.Textbox
    normalize: gr.Checkbox
    size: gr.Radio
    attn: gr.Dropdown
    non_streaming: gr.Checkbox
    gen: dict[str, gr.components.Component]

    def inputs(self) -> set:
        return {self.text, self.language, self.mode, self.pause, self.seed, self.normalize, self.size, self.attn,
                self.non_streaming, *self.gen.values()}

    def request(self, data: dict, **extra) -> GenRequest:
        return GenRequest(
            kind=self.kind, text=data[self.text], size=data[self.size], attn=data[self.attn],
            language=data[self.language], seed=parse_seed(data[self.seed]),
            gen={k: data[c] for k, c in self.gen.items()}, non_streaming_mode=data[self.non_streaming],
            normalize=data[self.normalize], split=data[self.mode] == "split", pause_sec=data[self.pause],
            batch=data[self.mode] == "batch", **extra)

    def restore(self, params: dict) -> dict:
        """Form values from a history record's params (F-73). A batch item restores as a normal generation."""
        values = {self.text: params["text"], self.language: params["language"], self.seed: str(params["seed"]),
                  self.normalize: params["normalize"], self.size: params["size"], self.attn: params["attn"],
                  self.non_streaming: params["non_streaming_mode"],
                  self.mode: "split" if params.get("split") else "normal",
                  self.pause: params.get("pause_sec", 0.2)}
        values.update({c: params["gen"].get(k, GEN_DEFAULTS[k]) for k, c in self.gen.items()})
        return values


def example_text_picker(text: gr.Textbox, language: gr.Dropdown) -> gr.Dropdown:
    picker = gr.Dropdown(presets.labels(presets.EXAMPLE_TEXTS), value=None, label=t("例文を挿入", "Insert an example"))

    def insert(label):
        p = presets.find(presets.EXAMPLE_TEXTS, label)
        return (p[2], p[1]) if p else (gr.skip(), gr.skip())

    picker.change(insert, picker, [text, language], queue=False)
    return picker


def instruct_picker(preset_list: list[tuple], target: gr.Textbox, label) -> gr.Dropdown:
    picker = gr.Dropdown(presets.labels(preset_list), value=None, label=label)

    def insert(label):
        p = presets.find(preset_list, label)
        return p[1] if p else gr.skip()

    picker.change(insert, picker, target, queue=False)
    return picker


def gen_controls(kind: str, settings: Settings, text_placeholder) -> GenControls:
    text = gr.Textbox(label=t("合成するテキスト", "Text to synthesize"), lines=4, placeholder=text_placeholder,
                      info=t(f"上限 {settings.max_chars} 文字", f"Up to {settings.max_chars} characters"))
    with gr.Row():
        language = gr.Dropdown(LANGUAGE_CHOICES, value="auto", label=t("言語", "Language"),
                               info=t("分かっている場合は明示したほうが安定します", "Set it if you know it; results are more stable"))
        example_text_picker(text, language)
    with gr.Row():
        mode = gr.Radio(MODE_CHOICES, value="normal", label=t("生成のしかた", "Generation mode"),
                        info=t(f"分割: 長文を 1 文ずつ生成して連結 / バッチ: 最大 {MAX_BATCH} 行をまとめて推論",
                               f"Split: generate a long text sentence by sentence and join / "
                               f"Batch: up to {MAX_BATCH} lines in one batched call"))
        pause = gr.Slider(0.0, 1.5, 0.2, step=0.05, label=t("文の間の無音（秒）", "Pause between sentences (s)"),
                          visible=False)
    with gr.Row():
        seed = seed_box(info=t("空欄なら毎回ランダム（使った値は結果に表示）",
                               "Empty: random each time (the value used is shown)"))
        normalize = gr.Checkbox(True, label=t("音量を正規化", "Normalize loudness"),
                                info=t(f"出力を {settings.target_lufs:g} LUFS に揃える（ピークは -1 dBFS まで）",
                                       f"Bring the output to {settings.target_lufs:g} LUFS (peak limited to -1 dBFS)"))
    mode.change(lambda m: gr.Slider(visible=m == "split"), mode, pause, queue=False)

    with gr.Accordion(t("詳細設定", "Advanced settings"), open=False):
        sizes = ["1.7B"] if kind == "voice_design" else ["1.7B", "0.6B"]
        attn_choices = [a for a in ATTN_CHOICES if a != "flash_attention_2" or flash_attn_available()]
        with gr.Row():
            size = gr.Radio(sizes, value="1.7B", label=t("モデルサイズ", "Model size"), interactive=len(sizes) > 1,
                            info=t("0.6B は VRAM が約半分。速度はほぼ同じ", "0.6B uses about half the VRAM at about the same speed"))
            attn = gr.Dropdown(attn_choices, value="sdpa", label=t("attention 実装", "Attention implementation"),
                               info=t("変更するとモデルを読み込み直します。sdpa が最速でした",
                                      "Changing it reloads the model. sdpa was the fastest"))
            non_streaming = gr.Checkbox(
                NON_STREAMING_DEFAULTS[kind], label="non_streaming_mode",
                info=t("False はテキストのストリーミング入力を模擬するだけで、出力は変わりません",
                       "False only simulates streaming text input; the output does not stream"))
        gr.Markdown(t("**talker**（第1コードブック）", "**talker** (first codebook)"))
        with gr.Row():
            do_sample = gr.Checkbox(GEN_DEFAULTS["do_sample"], label="do_sample",
                                    info=t("オフでグリーディ", "Off: greedy decoding"))
            temperature = gr.Slider(0.05, 2.0, GEN_DEFAULTS["temperature"], step=0.05, label="temperature",
                                    info=t("高いほど抑揚が大きく、不安定に", "Higher: more expressive, less stable"))
            top_k = gr.Slider(0, 200, GEN_DEFAULTS["top_k"], step=1, label="top_k", info=t("0 で無効", "0 disables it"))
            top_p = gr.Slider(0.05, 1.0, GEN_DEFAULTS["top_p"], step=0.01, label="top_p")
        with gr.Row():
            repetition_penalty = gr.Slider(1.0, 2.0, GEN_DEFAULTS["repetition_penalty"], step=0.01,
                                           label="repetition_penalty",
                                           info=t("同じ音の繰り返しを抑える", "Suppresses repeated sounds"))
            max_new_tokens = gr.Number(GEN_DEFAULTS["max_new_tokens"], label="max_new_tokens", precision=0,
                                       minimum=16, maximum=8192,
                                       info=t("12 Hz なので 12 トークン ≒ 1 秒", "12 Hz: 12 tokens ≈ 1 second"))
        gr.Markdown(t("**sub-talker**（残り 15 コードブック）", "**sub-talker** (the other 15 codebooks)"))
        with gr.Row():
            subtalker_dosample = gr.Checkbox(GEN_DEFAULTS["subtalker_dosample"], label="subtalker_dosample")
            subtalker_temperature = gr.Slider(0.05, 2.0, GEN_DEFAULTS["subtalker_temperature"], step=0.05,
                                              label="subtalker_temperature")
            subtalker_top_k = gr.Slider(0, 200, GEN_DEFAULTS["subtalker_top_k"], step=1, label="subtalker_top_k")
            subtalker_top_p = gr.Slider(0.05, 1.0, GEN_DEFAULTS["subtalker_top_p"], step=0.01,
                                        label="subtalker_top_p")
        reset = gr.Button(t("既定値に戻す", "Reset to defaults"), size="sm")

    gen = dict(do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
               repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
               subtalker_dosample=subtalker_dosample, subtalker_temperature=subtalker_temperature,
               subtalker_top_k=subtalker_top_k, subtalker_top_p=subtalker_top_p)
    reset.click(lambda: [GEN_DEFAULTS[k] for k in gen] + [NON_STREAMING_DEFAULTS[kind]],
                outputs=[*gen.values(), non_streaming], queue=False)
    return GenControls(kind, text, language, mode, pause, seed, normalize, size, attn, non_streaming, gen)


def format_result(r: GenResult, batch_size: int = 1) -> str:
    load = f"{r.load_sec:.1f} s" if r.load_sec is not None else "—"
    audio = f"{r.audio_sec:.2f} s" if batch_size == 1 else f"{r.audio_sec:.2f} s (+{batch_size - 1})"
    return (f"{RESULT_HEADER}\n|---:|---:|---:|---:|---:|---:|\n"
            f"| {r.synth_sec:.2f} s | {audio} | {r.rtf:.2f} | {r.peak_reserved_gib:.2f} GiB | {load} | {r.seed} |")


@dataclass
class OutputArea:
    run: gr.Button
    cancel: gr.Button
    audio: gr.Audio
    metrics: gr.Markdown
    batch_box: gr.Column
    batch_audios: list[gr.Audio]
    last_id: gr.State = field(default_factory=lambda: gr.State(None))  # history id of the latest result


def output_area() -> OutputArea:
    with gr.Row():
        run = gr.Button(t("生成", "Generate"), variant="primary", scale=3)
        cancel = gr.Button(t("キャンセル", "Cancel"), scale=1)
    audio = gr.Audio(label=t("生成結果", "Result"), type="filepath", interactive=False, buttons=["download"])
    metrics = gr.Markdown()
    with gr.Column(visible=False) as batch_box:
        gr.Markdown(t("**バッチの結果**（テキストごと）", "**Batch results** (one per text)"))
        batch_audios = [gr.Audio(type="filepath", interactive=False, buttons=["download"], visible=False)
                        for _ in range(MAX_BATCH)]
    return OutputArea(run, cancel, audio, metrics, batch_box, batch_audios)


def wire(engine: Engine, controls: GenControls, extra_inputs: dict[str, gr.components.Component],
         out: OutputArea, extra_outputs: list | None = None,
         extra_values: Callable[[list[GenResult]], list] | None = None) -> None:
    """Connect the generate / cancel buttons. `extra_values(results)` fills `extra_outputs` after a run."""
    extra_outputs = extra_outputs or []

    def handler(data: dict, request: gr.Request, progress=gr.Progress()):
        try:
            req = controls.request(data, **{k: data[c] for k, c in extra_inputs.items()})
            results = engine.generate_batch(req, client_ip=client_ip(request),
                                            progress=progress_fn(progress, request),
                                            cancel_token=request.session_hash)
        except UserError as e:
            raise user_error(e, request) from e
        except Cancelled:
            return [gr.skip()] * (4 + MAX_BATCH + len(extra_outputs))
        show_warnings(results[0].warnings, request)
        batch = [gr.Audio(str(r.audio_path), label=clip(r.text, 60), visible=True)
                 for r in results] if req.batch else []
        batch += [gr.Audio(None, visible=False)] * (MAX_BATCH - len(batch))
        return [str(results[0].audio_path), format_result(results[0], len(results)), results[0].id,
                gr.Column(visible=req.batch), *batch, *(extra_values(results) if extra_values else [])]

    event = out.run.click(handler, inputs=controls.inputs() | set(extra_inputs.values()),
                          outputs=[out.audio, out.metrics, out.last_id, out.batch_box, *out.batch_audios,
                                   *extra_outputs], **GPU)
    out.cancel.click(cancel_fn(engine), None, None, cancels=[event], queue=False)


def cancel_fn(engine: Engine) -> Callable:
    """Cancel button handler: also stops a running sentence-by-sentence generation at the next sentence."""
    def cancel(request: gr.Request):
        engine.cancel(request.session_hash)
    return cancel


def save_voice_row(engine: Engine, last_id: gr.State, note, editable_ref: bool = False):
    """'Save this voice to the library' controls under a result (F-27, F-32).

    With `editable_ref`, the result audio and its text are shown for trimming before saving (F-33); the caller
    fills them after each generation. Returns (save event, ref audio, ref text) so the caller can chain on it."""
    ref_audio = ref_text = None
    with gr.Accordion(t("この声を話者ライブラリに保存", "Save this voice to the speaker library"), open=False):
        gr.Markdown(note)
        if editable_ref:
            ref_audio = gr.Audio(label=t("参照音声として保存する音声（波形の編集ボタンで切り取れます）",
                                         "Audio to save as the reference (trim it with the waveform edit button)"),
                                 type="filepath", sources=["upload"], editable=True)
            ref_text = gr.Textbox(label=t("その書き起こし（切り取ったら合わせて直してください）",
                                          "Its transcript (edit it to match if you trim)"), lines=2)
        with gr.Row():
            name = gr.Textbox(label=t("名前", "Name"), scale=2)
            memo = gr.Textbox(label=t("メモ", "Memo"), scale=3)
            save = gr.Button(t("保存", "Save"), scale=1)

    def on_save(request: gr.Request, record_id, name, memo, audio=None, text=None):
        if not record_id:
            raise gr.Error(say(M.GENERATE_FIRST, request))
        try:
            spk = engine.save_voice(record_id, name, memo, ref_audio=audio, ref_text=text)
        except UserError as e:
            raise user_error(e, request) from e
        gr.Info(say(M.SAVED, request, f" {spk['name']}"))
        return "", ""

    inputs = [last_id, name, memo] + ([ref_audio, ref_text] if editable_ref else [])
    return save.click(on_save, inputs, [name, memo], queue=False), ref_audio, ref_text
