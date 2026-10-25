"""System tab: loaded models and VRAM, aggregated metrics, environment, speaker samples (F-80 .. F-83, F-41, F-75)."""

from __future__ import annotations

import importlib.metadata
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import gradio as gr
import pandas as pd
import torch

from tts_core import MODEL_IDS, flash_attn_available, vram_snapshot

from .engine import Engine
from .samples import generate_samples
from .ui_common import GPU, KIND_LABELS

# Categorical slots 1-5 of the reference palette, in a fixed order per model so a model keeps its color
# whatever subset of models the history contains.
MODEL_COLORS = dict(zip([f"{KIND_LABELS[k]} {s}" for k, s in MODEL_IDS],
                        ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))


@dataclass
class SystemTab:
    refresh: Callable
    refresh_outputs: list
    samples_done: object  # event fired after speaker samples are generated


def _model_label(kind: str, size: str) -> str:
    return f"{KIND_LABELS[kind]} {size}"


def _env_markdown(engine: Engine) -> str:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "（なし）"
    rows = [("GPU", gpu), ("torch", torch.__version__), ("CUDA（torch）", torch.version.cuda or "—"),
            ("qwen-tts", importlib.metadata.version("qwen-tts")), ("gradio", gr.__version__),
            ("flash-attn", "導入済み" if flash_attn_available() else "未導入（sdpa / eager のみ）"),
            ("VRAM の上限", f"{engine.settings.vram_budget_gib:g} GiB"),
            ("データの保存先", str(engine.settings.data_dir))]
    return "| 項目 | 値 |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


def _dir_size_gib(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30 if path.exists() else 0.0


def system_tab(engine: Engine) -> SystemTab:
    gr.Markdown("### ロード中のモデル")
    vram = gr.Markdown()
    models = gr.Dataframe(headers=["モデル", "attention", "重み GiB", "ロード 秒", "最後に使った時刻"],
                          interactive=False)
    with gr.Row():
        target = gr.Dropdown([], label="解放するモデル", scale=3)
        unload = gr.Button("解放", scale=1)
        unload_all = gr.Button("すべて解放", variant="stop", scale=1)
        refresh = gr.Button("更新", scale=1)

    gr.Markdown("### 計測値の集計（履歴全体）")
    stats = gr.Dataframe(headers=["モデル", "件数", "RTF 平均", "RTF 中央値", "合成時間 平均 秒", "ピーク VRAM 最大 GiB"],
                         interactive=False)
    scatter = gr.ScatterPlot(x="音声長（秒）", y="RTF", color="モデル", color_map=MODEL_COLORS,
                             tooltip=["モデル", "音声長（秒）", "RTF", "合成時間（秒）"],
                             title="音声の長さと RTF", height=360,
                             caption="RTF = 合成時間 / 音声長。1 未満なら実時間より速い")
    disk = gr.Markdown()

    gr.Markdown("### CustomVoice の話者サンプル")
    with gr.Row():
        make_samples = gr.Button("未生成のサンプルを生成（9 話者 × 日英）")
        samples_status = gr.Markdown()

    gr.Markdown("### 環境")
    gr.Markdown(_env_markdown(engine))

    def collect():
        v = vram_snapshot() if torch.cuda.is_available() else None
        loaded = engine.models.loaded()
        vram_md = (f"デバイス全体 **{v['device_used_gib']:.2f} / {v['device_total_gib']:.2f} GiB** 使用中"
                   f"（このアプリの確保分 {v['reserved_gib']:.2f} GiB）。" if v else "GPU がありません。")
        model_rows = [[_model_label(e.key[0], e.key[1]), e.key[2], round(e.weights_gib, 2), round(e.load_sec, 1),
                       datetime.fromtimestamp(e.last_used).strftime("%H:%M:%S")] for e in loaded]
        choices = [(f"{_model_label(e.key[0], e.key[1])}（{e.key[2]}）", "|".join(e.key)) for e in loaded]

        records = engine.history.search(limit=1_000_000)
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(_model_label(r["kind"], r["size"]), []).append(r)
        stat_rows = [[label, len(rs), round(statistics.fmean(r["rtf"] for r in rs), 2),
                      round(statistics.median(r["rtf"] for r in rs), 2),
                      round(statistics.fmean(r["synth_sec"] for r in rs), 2),
                      round(max(r["peak_reserved_gib"] or 0 for r in rs), 2)]
                     for label, rs in sorted(groups.items(), key=lambda kv: list(MODEL_COLORS).index(kv[0]))]
        df = pd.DataFrame([{"モデル": _model_label(r["kind"], r["size"]), "音声長（秒）": round(r["audio_sec"], 2),
                            "RTF": round(r["rtf"], 3), "合成時間（秒）": round(r["synth_sec"], 2)} for r in records],
                          columns=["モデル", "音声長（秒）", "RTF", "合成時間（秒）"])
        disk_md = (f"履歴 {len(records)} 件。データの使用量: "
                   f"**{_dir_size_gib(engine.settings.data_dir):.2f} GiB**（自動削除はしません）")
        return vram_md, model_rows, gr.Dropdown(choices=choices, value=None), stat_rows, df, disk_md

    outputs = [vram, models, target, stats, scatter, disk]

    def do_unload(key):
        if key:
            engine.unload(tuple(key.split("|")))
        return collect()

    def do_unload_all():
        engine.unload()
        return collect()

    def do_samples(progress=gr.Progress()):
        n = generate_samples(engine, progress=lambda msg: progress(None, desc=msg))
        return f"{n} 件を生成しました。" if n else "すべて生成済みです。"

    refresh.click(collect, None, outputs, queue=False)
    unload.click(do_unload, target, outputs, **GPU)
    unload_all.click(do_unload_all, None, outputs, **GPU)
    samples_done = make_samples.click(do_samples, None, samples_status, **GPU)
    return SystemTab(collect, outputs, samples_done)
