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
from .i18n import t
from .samples import generate_samples
from .ui_common import GPU, KIND_LABELS, progress_fn

# Categorical slots 1-5 of the reference palette, in a fixed order per model so a model keeps its color
# whatever subset of models the history contains.
MODEL_COLORS = dict(zip([f"{KIND_LABELS[k]} {s}" for k, s in MODEL_IDS],
                        ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MODEL_LABELS = {**KIND_LABELS, "asr": "Whisper"}
VRAM_USED = t("デバイス全体の VRAM 使用量（GiB）/ このアプリの確保分:", "Device VRAM in use (GiB) / reserved by this app:")
NO_GPU = t("GPU がありません。", "No GPU.")
DISK = t("履歴の件数 / データの使用量（GiB、自動削除はしません）:", "History records / data size (GiB, never deleted automatically):")
SAMPLES_MADE = t("生成したサンプル数:", "Samples generated:")


@dataclass
class SystemTab:
    refresh: Callable
    refresh_outputs: list
    samples_done: object  # event fired after speaker samples are generated


def _model_label(kind: str, size: str) -> str:
    return f"{MODEL_LABELS[kind]} {size}"


def _env_markdown(engine: Engine) -> str:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "（なし）"
    rows = [("GPU", gpu), ("torch", torch.__version__), ("CUDA (torch)", torch.version.cuda or "—"),
            ("qwen-tts", importlib.metadata.version("qwen-tts")), ("gradio", gr.__version__),
            ("flash-attn", "installed" if flash_attn_available() else "not installed (sdpa / eager only)"),
            ("VRAM budget", f"{engine.settings.vram_budget_gib:g} GiB"),
            ("data_dir", str(engine.settings.data_dir))]
    return "| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


def _dir_size_gib(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30 if path.exists() else 0.0


def system_tab(engine: Engine) -> SystemTab:
    gr.Markdown(t("### ロード中のモデル", "### Loaded models"))
    vram = gr.Markdown()
    models = gr.Dataframe(headers=["モデル / Model", "attention", "重み / Weights GiB", "ロード / Load s",
                                   "最終使用 / Last used"], interactive=False)
    with gr.Row():
        target = gr.Dropdown([], label=t("解放するモデル", "Model to unload"), scale=3)
        unload = gr.Button(t("解放", "Unload"), scale=1)
        unload_all = gr.Button(t("すべて解放", "Unload all"), variant="stop", scale=1)
        refresh = gr.Button(t("更新", "Refresh"), scale=1)

    gr.Markdown(t("### 計測値の集計（履歴全体）", "### Measurements (whole history)"))
    stats = gr.Dataframe(headers=["モデル / Model", "件数 / Runs", "RTF 平均 / mean", "RTF 中央値 / median",
                                  "合成 平均 / Synthesis mean s", "ピーク VRAM 最大 / Peak max GiB"],
                         interactive=False)
    gr.Markdown(t("**音声の長さと RTF**（RTF = 合成時間 / 音声長。1 未満なら実時間より速い）",
                  "**Audio length and RTF** (RTF = synthesis time / audio length. Below 1 is faster than real time)"))
    # Plot titles are not translated by gr.I18n, and column names are data keys: both stay language-neutral.
    scatter = gr.ScatterPlot(x="audio_sec", y="RTF", color="model", color_map=MODEL_COLORS,
                             tooltip=["model", "audio_sec", "RTF", "synth_sec"],
                             x_title="音声の長さ / Audio length (s)", color_title="モデル / Model", height=360,
                             show_label=False)
    disk = gr.Markdown()

    gr.Markdown(t("### CustomVoice の話者サンプル", "### CustomVoice speaker samples"))
    with gr.Row():
        make_samples = gr.Button(t("未生成のサンプルを生成（9 話者 × 日英）",
                                   "Generate missing samples (9 speakers x Japanese / English)"))
        samples_status = gr.Markdown()

    gr.Markdown(t("### 環境", "### Environment"))
    gr.Markdown(_env_markdown(engine))

    def collect():
        v = vram_snapshot() if torch.cuda.is_available() else None
        loaded = engine.models.loaded()
        vram_md = (str(VRAM_USED) + f" **{v['device_used_gib']:.2f} / {v['device_total_gib']:.2f}** / "
                   f"{v['reserved_gib']:.2f}" if v else str(NO_GPU))
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
        df = pd.DataFrame([{"model": _model_label(r["kind"], r["size"]), "audio_sec": round(r["audio_sec"], 2),
                            "RTF": round(r["rtf"], 3), "synth_sec": round(r["synth_sec"], 2)} for r in records],
                          columns=["model", "audio_sec", "RTF", "synth_sec"])
        disk_md = str(DISK) + f" {len(records)} / {_dir_size_gib(engine.settings.data_dir):.2f}"
        return vram_md, model_rows, gr.Dropdown(choices=choices, value=None), stat_rows, df, disk_md

    outputs = [vram, models, target, stats, scatter, disk]

    def do_unload(key):
        if key:
            engine.unload(tuple(key.split("|")))
        return collect()

    def do_unload_all():
        engine.unload()
        return collect()

    def do_samples(request: gr.Request, progress=gr.Progress()):
        n = generate_samples(engine, progress=progress_fn(progress, request))
        return str(SAMPLES_MADE) + f" {n}"

    refresh.click(collect, None, outputs, queue=False)
    unload.click(do_unload, target, outputs, **GPU)
    unload_all.click(do_unload_all, None, outputs, **GPU)
    samples_done = make_samples.click(do_samples, None, samples_status, **GPU)
    return SystemTab(collect, outputs, samples_done)
