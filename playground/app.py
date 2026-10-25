"""Qwen3-TTS playground (Gradio).

Example:
  uv run playground/app.py                       # http://localhost:7860 (also reachable from the LAN)
  uv run playground/app.py --preload none --vram-budget 12
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from tts_core import ATTN_CHOICES, GEN_DEFAULTS, LANGUAGES, NON_STREAMING_DEFAULTS, SPEAKERS, flash_attn_available
from playground.config import Settings
from playground.engine import Engine, GenRequest, GenResult
from playground.history import HistoryStore
from playground.model_manager import ModelManager

KIND_LABELS = {"base": "ボイスクローン", "voice_design": "VoiceDesign", "custom_voice": "CustomVoice"}
LANGUAGE_LABELS = {
    "auto": "自動判定", "chinese": "中国語", "english": "英語", "french": "フランス語", "german": "ドイツ語",
    "italian": "イタリア語", "japanese": "日本語", "korean": "韓国語", "portuguese": "ポルトガル語",
    "russian": "ロシア語", "spanish": "スペイン語",
}
LANGUAGE_CHOICES = [(f"{LANGUAGE_LABELS[k]} ({k.capitalize()})", k) for k in LANGUAGES]
SPEAKER_CHOICES = [("_".join(w.capitalize() for w in s.split("_")), s) for s in SPEAKERS]
GPU = dict(concurrency_limit=1, concurrency_id="gpu")


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
    reset: gr.Button

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


def gen_controls(kind: str, settings: Settings, text_placeholder: str) -> GenControls:
    text = gr.Textbox(label="合成するテキスト", lines=4, placeholder=text_placeholder,
                      info=f"上限 {settings.max_chars} 文字")
    with gr.Row():
        language = gr.Dropdown(LANGUAGE_CHOICES, value="auto", label="言語",
                               info="分かっている場合は明示したほうが安定します")
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
    controls = GenControls(kind, text, language, seed, normalize, size, attn, non_streaming, gen, reset)
    reset.click(lambda: [GEN_DEFAULTS[k] for k in gen] + [NON_STREAMING_DEFAULTS[kind]],
                outputs=[*gen.values(), non_streaming], queue=False)
    return controls


def format_result(r: GenResult) -> str:
    load = f"{r.load_sec:.1f} 秒" if r.load_sec is not None else "（ロード済み）"
    return ("| 合成時間 | 音声長 | RTF | ピーク VRAM | モデルのロード | seed |\n"
            "|---:|---:|---:|---:|---:|---:|\n"
            f"| {r.synth_sec:.2f} 秒 | {r.audio_sec:.2f} 秒 | {r.rtf:.2f} | {r.peak_reserved_gib:.2f} GiB | "
            f"{load} | {r.seed} |")


def run_generation(engine: Engine, req: GenRequest, request: gr.Request | None,
                   progress: gr.Progress) -> tuple[str, str]:
    try:
        result = engine.generate(req, client_ip=request.client.host if request and request.client else None,
                                 progress=lambda msg: progress(None, desc=msg))
    except ValueError as e:
        raise gr.Error(str(e)) from e
    for w in result.warnings:
        gr.Warning(w)
    return str(result.audio_path), format_result(result)


def output_area() -> tuple[gr.Button, gr.Button, gr.Audio, gr.Markdown]:
    with gr.Row():
        run = gr.Button("生成", variant="primary", scale=3)
        cancel = gr.Button("キャンセル", scale=1)
    audio = gr.Audio(label="生成結果", type="filepath", interactive=False, buttons=["download"])
    metrics = gr.Markdown()
    return run, cancel, audio, metrics


def wire(engine: Engine, controls: GenControls, extra_inputs: dict[str, gr.components.Component],
         run: gr.Button, cancel: gr.Button, audio: gr.Audio, metrics: gr.Markdown) -> None:
    def handler(data: dict, request: gr.Request, progress=gr.Progress()):
        req = controls.request(data, **{k: data[c] for k, c in extra_inputs.items()})
        return run_generation(engine, req, request, progress)

    event = run.click(handler, inputs=controls.inputs() | set(extra_inputs.values()), outputs=[audio, metrics],
                      **GPU)
    cancel.click(None, cancels=[event])


# ---- tabs ---------------------------------------------------------------------------------------

def clone_tab(engine: Engine, settings: Settings) -> None:
    gr.Markdown("数秒の参照音声と、その書き起こしから声をクローンします（Base モデル）。"
                "参照音声は 5〜10 秒程度、書き起こしは正確なほど良い結果になります。")
    with gr.Row():
        with gr.Column():
            ref_audio = gr.Audio(label="参照音声", sources=["upload"], type="filepath")
            x_vector_only = gr.Checkbox(False, label="x-vector only モード",
                                        info="書き起こしなしで話者埋め込みだけを使う（品質は下がる）")
            ref_text = gr.Textbox(label="参照音声の書き起こし", lines=2)
        with gr.Column():
            controls = gen_controls("base", settings, "参照音声の声で読み上げるテキスト")
    run, cancel, audio, metrics = output_area()

    x_vector_only.change(lambda on: gr.Textbox(interactive=not on), x_vector_only, ref_text, queue=False)
    wire(engine, controls, {"ref_audio": ref_audio, "ref_text": ref_text, "x_vector_only": x_vector_only},
         run, cancel, audio, metrics)


def design_tab(engine: Engine, settings: Settings) -> None:
    gr.Markdown("声の特徴を自然文で説明すると、その声で読み上げます（VoiceDesign 1.7B）。")
    with gr.Row():
        with gr.Column():
            instruct = gr.Textbox(label="声の説明", lines=4,
                                  placeholder="例: 落ち着いた低めの声の40代男性。ゆっくり丁寧に話す。",
                                  info="年齢、性別、声質、話し方、感情など。空欄でもよい")
        with gr.Column():
            controls = gen_controls("voice_design", settings, "デザインした声で読み上げるテキスト")
    run, cancel, audio, metrics = output_area()
    wire(engine, controls, {"instruct": instruct}, run, cancel, audio, metrics)


def custom_tab(engine: Engine, settings: Settings) -> None:
    gr.Markdown("9 人のプリセット話者で読み上げます（CustomVoice）。1.7B はスタイル指示にも対応しています。")
    with gr.Row():
        with gr.Column():
            speaker = gr.Dropdown(SPEAKER_CHOICES, value="ono_anna", label="話者")
            instruct = gr.Textbox(label="スタイル指示（1.7B のみ）", lines=2,
                                  placeholder="例: 楽しそうに、明るく / Calm and confident, like a news anchor.")
        with gr.Column():
            controls = gen_controls("custom_voice", settings, "読み上げるテキスト")
    run, cancel, audio, metrics = output_area()

    def on_size(size: str):
        if size == "0.6B":
            return gr.Textbox(interactive=False, info="0.6B はスタイル指示に対応していません")
        return gr.Textbox(interactive=True, info=None)

    controls.size.change(on_size, controls.size, instruct, queue=False)
    wire(engine, controls, {"speaker": speaker, "instruct": instruct}, run, cancel, audio, metrics)


HISTORY_HEADERS = ["日時", "モデル", "サイズ", "言語", "テキスト", "話者 / 説明", "seed", "音声 秒", "RTF", "アクセス元"]


def history_tab(engine: Engine):
    """Returns (refresh fn, its inputs, table, ids) so the caller can refresh the list on tab open."""
    store = engine.history
    with gr.Row():
        kind = gr.Dropdown([("すべて", "")] + [(v, k) for k, v in KIND_LABELS.items()], value="", label="モデル")
        language = gr.Dropdown([("すべて", "")] + LANGUAGE_CHOICES, value="", label="言語")
        text = gr.Textbox(label="テキストを含む")
        date_from = gr.Textbox(label="開始日", placeholder="2026-10-01")
        date_to = gr.Textbox(label="終了日", placeholder="2026-10-31")
    search = gr.Button("検索")
    ids = gr.State([])
    table = gr.Dataframe(headers=HISTORY_HEADERS, interactive=False, wrap=True, max_height=420)
    selected = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(label="選択した結果", type="filepath", interactive=False, buttons=["download"])
            delete = gr.Button("この結果を削除", variant="stop", interactive=False)
        with gr.Column(scale=1):
            detail = gr.JSON(label="条件")

    def rows(kind, language, text, date_from, date_to):
        records = store.search(kind=kind or None, language=language or None, text=text or None,
                               date_from=date_from.strip() or None, date_to=date_to.strip() or None)
        data = [[r["created_at"].replace("T", " "), KIND_LABELS[r["kind"]], r["size"], r["language"],
                 _clip(r["text"], 40), _clip(r["speaker"] or r["instruct"] or "", 30), r["seed"],
                 round(r["audio_sec"], 2), round(r["rtf"], 2), r["client_ip"] or ""] for r in records]
        return data, [r["id"] for r in records]

    def select(ids, evt: gr.SelectData):
        rec = store.get(ids[evt.index[0]])
        if rec is None:
            return None, None, None, gr.Button(interactive=False)
        return rec["id"], rec["audio_path"], rec["params"], gr.Button(interactive=True)

    def remove(record_id, *filters):
        if record_id:
            store.delete(record_id)
        return (*rows(*filters), None, None, None, gr.Button(interactive=False))

    filters = [kind, language, text, date_from, date_to]
    search.click(rows, filters, [table, ids], queue=False)
    table.select(select, ids, [selected, audio, detail, delete], queue=False)
    delete.click(remove, [selected, *filters], [table, ids, selected, audio, detail, delete], queue=False)
    return rows, filters, table, ids


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def build_app(engine: Engine) -> gr.Blocks:
    settings = engine.settings
    with gr.Blocks(title="Qwen3-TTS Playground") as demo:
        gr.Markdown("# Qwen3-TTS Playground\n"
                    "生成は GPU 1 枚で 1 件ずつ順に処理します。音声の長さの約 1.5 倍の時間がかかります。")
        with gr.Tabs():
            with gr.Tab("ボイスクローン"):
                clone_tab(engine, settings)
            with gr.Tab("VoiceDesign"):
                design_tab(engine, settings)
            with gr.Tab("CustomVoice"):
                custom_tab(engine, settings)
            with gr.Tab("履歴") as history:
                rows, filters, table, ids = history_tab(engine)
        history.select(rows, filters, [table, ids], queue=False)
        demo.load(rows, filters, [table, ids], queue=False)
    demo.queue(max_size=settings.queue_size, default_concurrency_limit=8)
    return demo


def parse_preload(values: list[str]) -> list[tuple[str, str]]:
    if values == ["none"]:
        return []
    out = []
    for v in values:
        kind, _, size = v.partition(":")
        out.append((kind, size or "1.7B"))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0", help="listen address (0.0.0.0: reachable from the LAN)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--data-dir", type=Path, default=Settings.data_dir)
    p.add_argument("--vram-budget", type=float, default=Settings.vram_budget_gib,
                   help="GiB; least recently used models are unloaded to stay under it")
    p.add_argument("--max-chars", type=int, default=Settings.max_chars)
    p.add_argument("--queue-size", type=int, default=Settings.queue_size)
    p.add_argument("--target-lufs", type=float, default=Settings.target_lufs)
    p.add_argument("--preload", nargs="+", default=["base:1.7B"],
                   help="models to load at startup, e.g. base:1.7B custom_voice:0.6B (or 'none')")
    args = p.parse_args()

    settings = Settings(data_dir=args.data_dir, vram_budget_gib=args.vram_budget, max_chars=args.max_chars,
                        queue_size=args.queue_size, target_lufs=args.target_lufs,
                        preload=parse_preload(args.preload))
    models = ModelManager(settings.vram_budget_gib)
    engine = Engine(settings, models, HistoryStore(settings.db_path))
    for kind, size in settings.preload:
        models.get(kind, size, "sdpa")

    build_app(engine).launch(server_name=args.host, server_port=args.port,
                             allowed_paths=[str(settings.data_dir)])


if __name__ == "__main__":
    main()
