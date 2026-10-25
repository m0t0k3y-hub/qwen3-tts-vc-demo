"""UI parts shared by several tabs: labels, the generation form, the result area."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from tts_core import ATTN_CHOICES, GEN_DEFAULTS, LANGUAGES, NON_STREAMING_DEFAULTS, SPEAKERS, flash_attn_available

from . import presets
from .config import Settings
from .engine import Engine, GenRequest, GenResult
from .speakers import SOURCES, SpeakerStore

KIND_LABELS = {"base": "ボイスクローン", "voice_design": "VoiceDesign", "custom_voice": "CustomVoice"}
LANGUAGE_LABELS = {
    "auto": "自動判定", "chinese": "中国語", "english": "英語", "french": "フランス語", "german": "ドイツ語",
    "italian": "イタリア語", "japanese": "日本語", "korean": "韓国語", "portuguese": "ポルトガル語",
    "russian": "ロシア語", "spanish": "スペイン語",
}
LANGUAGE_CHOICES = [(f"{LANGUAGE_LABELS[k]} ({k.capitalize()})", k) for k in LANGUAGES]
SPEAKER_CHOICES = [("_".join(w.capitalize() for w in s.split("_")), s) for s in SPEAKERS]
# All GPU work shares one queue slot (N-10).
GPU = dict(concurrency_limit=1, concurrency_id="gpu")


def speaker_label(speaker: str) -> str:
    return {value: label for label, value in SPEAKER_CHOICES}.get(speaker, speaker)


def voice_choices(store: SpeakerStore) -> list[tuple[str, int]]:
    return [(f"{s['name']}（{SOURCES[s['source']]}）", s["id"]) for s in store.list()]


def clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


@dataclass
class GenControls:
    """The generation form shared by the three model tabs (F-01 .. F-13)."""

    kind: str
    text: gr.Textbox
    language: gr.Dropdown
    seed: gr.Number
    normalize: gr.Checkbox
    size: gr.Radio
    attn: gr.Dropdown
    non_streaming: gr.Checkbox
    gen: dict[str, gr.components.Component]

    def inputs(self) -> set:
        return {self.text, self.language, self.seed, self.normalize, self.size, self.attn, self.non_streaming,
                *self.gen.values()}

    def request(self, data: dict, **extra) -> GenRequest:
        seed = data[self.seed]
        return GenRequest(
            kind=self.kind, text=data[self.text], size=data[self.size], attn=data[self.attn],
            language=data[self.language], seed=None if seed is None else int(seed),
            gen={k: data[c] for k, c in self.gen.items()}, non_streaming_mode=data[self.non_streaming],
            normalize=data[self.normalize], **extra)

    def restore(self, params: dict) -> dict:
        """Form values from a history record's params (F-73)."""
        values = {self.text: params["text"], self.language: params["language"], self.seed: params["seed"],
                  self.normalize: params["normalize"], self.size: params["size"], self.attn: params["attn"],
                  self.non_streaming: params["non_streaming_mode"]}
        values.update({c: params["gen"].get(k, GEN_DEFAULTS[k]) for k, c in self.gen.items()})
        return values


def example_text_picker(text: gr.Textbox, language: gr.Dropdown) -> gr.Dropdown:
    picker = gr.Dropdown(presets.labels(presets.EXAMPLE_TEXTS), value=None, label="例文を挿入")

    def insert(label):
        p = presets.find(presets.EXAMPLE_TEXTS, label)
        return (p[2], p[1]) if p else (gr.skip(), gr.skip())

    picker.change(insert, picker, [text, language], queue=False)
    return picker


def instruct_picker(preset_list: list[tuple], target: gr.Textbox, label: str) -> gr.Dropdown:
    picker = gr.Dropdown(presets.labels(preset_list), value=None, label=label)

    def insert(label):
        p = presets.find(preset_list, label)
        return p[1] if p else gr.skip()

    picker.change(insert, picker, target, queue=False)
    return picker


def gen_controls(kind: str, settings: Settings, text_placeholder: str) -> GenControls:
    text = gr.Textbox(label="合成するテキスト", lines=4, placeholder=text_placeholder,
                      info=f"上限 {settings.max_chars} 文字")
    with gr.Row():
        language = gr.Dropdown(LANGUAGE_CHOICES, value="auto", label="言語",
                               info="分かっている場合は明示したほうが安定します")
        example_text_picker(text, language)
    with gr.Row():
        seed = gr.Number(None, label="seed", precision=0, minimum=0,
                         info="空欄なら毎回ランダム（使った値は結果に表示）")
        normalize = gr.Checkbox(True, label="音量を正規化",
                                info=f"出力を {settings.target_lufs:g} LUFS に揃える（ピークは -1 dBFS まで）")

    with gr.Accordion("詳細設定", open=False):
        sizes = ["1.7B"] if kind == "voice_design" else ["1.7B", "0.6B"]
        attn_choices = [a for a in ATTN_CHOICES if a != "flash_attention_2" or flash_attn_available()]
        with gr.Row():
            size = gr.Radio(sizes, value="1.7B", label="モデルサイズ", interactive=len(sizes) > 1,
                            info="0.6B は VRAM が約半分。速度はほぼ同じ")
            attn = gr.Dropdown(attn_choices, value="sdpa", label="attention 実装",
                               info="変更するとモデルを読み込み直します。sdpa が最速でした")
            non_streaming = gr.Checkbox(NON_STREAMING_DEFAULTS[kind], label="non_streaming_mode",
                                        info="False はテキストのストリーミング入力を模擬するだけで、出力は変わりません")
        gr.Markdown("**talker**（第1コードブック）")
        with gr.Row():
            do_sample = gr.Checkbox(GEN_DEFAULTS["do_sample"], label="do_sample", info="オフでグリーディ")
            temperature = gr.Slider(0.05, 2.0, GEN_DEFAULTS["temperature"], step=0.05, label="temperature",
                                    info="高いほど抑揚が大きく、不安定に")
            top_k = gr.Slider(0, 200, GEN_DEFAULTS["top_k"], step=1, label="top_k", info="0 で無効")
            top_p = gr.Slider(0.05, 1.0, GEN_DEFAULTS["top_p"], step=0.01, label="top_p")
        with gr.Row():
            repetition_penalty = gr.Slider(1.0, 2.0, GEN_DEFAULTS["repetition_penalty"], step=0.01,
                                           label="repetition_penalty", info="同じ音の繰り返しを抑える")
            max_new_tokens = gr.Number(GEN_DEFAULTS["max_new_tokens"], label="max_new_tokens", precision=0,
                                       minimum=16, maximum=8192, info="12 Hz なので 12 トークン ≒ 1 秒")
        gr.Markdown("**sub-talker**（残り 15 コードブック）")
        with gr.Row():
            subtalker_dosample = gr.Checkbox(GEN_DEFAULTS["subtalker_dosample"], label="subtalker_dosample")
            subtalker_temperature = gr.Slider(0.05, 2.0, GEN_DEFAULTS["subtalker_temperature"], step=0.05,
                                              label="subtalker_temperature")
            subtalker_top_k = gr.Slider(0, 200, GEN_DEFAULTS["subtalker_top_k"], step=1, label="subtalker_top_k")
            subtalker_top_p = gr.Slider(0.05, 1.0, GEN_DEFAULTS["subtalker_top_p"], step=0.01,
                                        label="subtalker_top_p")
        reset = gr.Button("既定値に戻す", size="sm")

    gen = dict(do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
               repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
               subtalker_dosample=subtalker_dosample, subtalker_temperature=subtalker_temperature,
               subtalker_top_k=subtalker_top_k, subtalker_top_p=subtalker_top_p)
    reset.click(lambda: [GEN_DEFAULTS[k] for k in gen] + [NON_STREAMING_DEFAULTS[kind]],
                outputs=[*gen.values(), non_streaming], queue=False)
    return GenControls(kind, text, language, seed, normalize, size, attn, non_streaming, gen)


def format_result(r: GenResult) -> str:
    load = f"{r.load_sec:.1f} 秒" if r.load_sec is not None else "（ロード済み）"
    return ("| 合成時間 | 音声長 | RTF | ピーク VRAM | モデルのロード | seed |\n"
            "|---:|---:|---:|---:|---:|---:|\n"
            f"| {r.synth_sec:.2f} 秒 | {r.audio_sec:.2f} 秒 | {r.rtf:.2f} | {r.peak_reserved_gib:.2f} GiB | "
            f"{load} | {r.seed} |")


def run_generation(engine: Engine, req: GenRequest, request: gr.Request | None,
                   progress: gr.Progress) -> GenResult:
    try:
        result = engine.generate(req, client_ip=request.client.host if request and request.client else None,
                                 progress=lambda msg: progress(None, desc=msg))
    except ValueError as e:
        raise gr.Error(str(e)) from e
    for w in result.warnings:
        gr.Warning(w)
    return result


@dataclass
class OutputArea:
    run: gr.Button
    cancel: gr.Button
    audio: gr.Audio
    metrics: gr.Markdown
    last_id: gr.State  # history id of the latest result in this tab


def output_area() -> OutputArea:
    with gr.Row():
        run = gr.Button("生成", variant="primary", scale=3)
        cancel = gr.Button("キャンセル", scale=1)
    audio = gr.Audio(label="生成結果", type="filepath", interactive=False, buttons=["download"])
    metrics = gr.Markdown()
    return OutputArea(run, cancel, audio, metrics, gr.State(None))


def wire(engine: Engine, controls: GenControls, extra_inputs: dict[str, gr.components.Component],
         out: OutputArea) -> None:
    def handler(data: dict, request: gr.Request, progress=gr.Progress()):
        req = controls.request(data, **{k: data[c] for k, c in extra_inputs.items()})
        result = run_generation(engine, req, request, progress)
        return str(result.audio_path), format_result(result), result.id

    event = out.run.click(handler, inputs=controls.inputs() | set(extra_inputs.values()),
                          outputs=[out.audio, out.metrics, out.last_id], **GPU)
    out.cancel.click(None, cancels=[event])


def save_voice_row(engine: Engine, last_id: gr.State, note: str):
    """'Save this voice to the library' controls under a result (F-27, F-32). Returns the save event so the
    caller can refresh voice pickers after it with `.then`."""
    with gr.Accordion("この声を話者ライブラリに保存", open=False):
        gr.Markdown(note)
        with gr.Row():
            name = gr.Textbox(label="名前", scale=2)
            memo = gr.Textbox(label="メモ", scale=3)
            save = gr.Button("保存", scale=1)

    def on_save(record_id, name, memo):
        try:
            spk = engine.save_voice(record_id, name, memo) if record_id else None
        except ValueError as e:
            raise gr.Error(str(e)) from e
        if spk is None:
            raise gr.Error("先に生成してください。")
        gr.Info(f"「{spk['name']}」を話者ライブラリに保存しました。")
        return "", ""

    return save.click(on_save, [last_id, name, memo], [name, memo], queue=False)
