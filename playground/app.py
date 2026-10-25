"""Qwen3-TTS playground (Gradio).

Example:
  uv run playground/app.py                       # http://localhost:7860 (also reachable from the LAN)
  uv run playground/app.py --preload none --vram-budget 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from playground.config import Settings
from playground.engine import Engine
from playground.history import HistoryStore
from playground.model_manager import ModelManager
from playground.speakers import SpeakerStore
from playground.ui_common import voice_choices
from playground.ui_compare import compare_tab
from playground.ui_generate import CloneTab, CustomTab, DesignTab, clone_tab, custom_tab, design_tab
from playground.ui_history import history_tab
from playground.ui_library import library_tab
from playground.ui_system import system_tab

TAB_OF_KIND = {"base": "clone", "voice_design": "design", "custom_voice": "custom"}


def build_app(engine: Engine) -> gr.Blocks:
    settings = engine.settings
    with gr.Blocks(title="Qwen3-TTS Playground") as demo:
        gr.Markdown("# Qwen3-TTS Playground\n"
                    "生成は GPU 1 枚で 1 件ずつ順に処理します。音声の長さの約 1.5 倍の時間がかかります。")
        with gr.Tabs() as tabs:
            with gr.Tab("ボイスクローン", id="clone") as clone_t:
                clone = clone_tab(engine)
            with gr.Tab("VoiceDesign", id="design"):
                design = design_tab(engine)
            with gr.Tab("CustomVoice", id="custom") as custom_t:
                custom = custom_tab(engine)
            with gr.Tab("話者ライブラリ", id="library") as library_t:
                library = library_tab(engine)
            with gr.Tab("比較", id="compare") as compare_t:
                compare = compare_tab(engine)
            with gr.Tab("履歴", id="history") as history_t:
                history = history_tab(engine)
            with gr.Tab("システム", id="system") as system_t:
                system = system_tab(engine)

        # Lists shared across tabs are re-read when a tab is opened, since other users may have changed them.
        def refresh_voices(n: int):
            def refresh():
                d = gr.Dropdown(choices=voice_choices(engine.speakers))
                return d if n == 1 else [d] * n
            return refresh

        clone_t.select(refresh_voices(1), None, clone.voice, queue=False)
        for saved in (clone.saved, design.saved):
            saved.then(refresh_voices(1), None, clone.voice, queue=False)
        compare_t.select(refresh_voices(len(compare.voices)), None, compare.voices, queue=False)
        library_t.select(library.refresh, None, library.refresh_outputs, queue=False)
        history_t.select(history.refresh, history.filters, history.refresh_outputs, queue=False)
        system_t.select(system.refresh, None, system.refresh_outputs, queue=False)
        custom_t.select(custom.show_samples, custom.speaker, custom.sample_outputs, queue=False)
        system.samples_done.then(custom.show_samples, custom.speaker, custom.sample_outputs, queue=False)
        demo.load(custom.show_samples, custom.speaker, custom.sample_outputs, queue=False)

        # Library -> clone tab with the voice filled in (F-54).
        def use_voice(speaker_id):
            spk = engine.speakers.get(speaker_id) if speaker_id is not None else None
            if spk is None:
                raise gr.Error("話者を選んでください。")
            return (gr.Tabs(selected="clone"), gr.Dropdown(choices=voice_choices(engine.speakers), value=spk["id"]),
                    spk["ref_audio"], spk["ref_text"] or "", spk["x_vector_only"])

        library.use.click(use_voice, library.selected,
                          [tabs, clone.voice, clone.ref_audio, clone.ref_text, clone.x_vector_only], queue=False)

        # History -> the model tab with the form restored (F-73).
        restore_outputs = {tabs, clone.voice, clone.ref_audio, clone.ref_text, clone.x_vector_only, design.instruct,
                           custom.speaker, custom.instruct, *clone.controls.inputs(), *design.controls.inputs(),
                           *custom.controls.inputs()}
        history.open_button.click(lambda record_id: restore(engine, record_id, tabs, clone, design, custom),
                                  history.selected, restore_outputs, queue=False)
    demo.queue(max_size=settings.queue_size, default_concurrency_limit=8)
    return demo


def restore(engine: Engine, record_id: str, tabs: gr.Tabs, clone: CloneTab, design: DesignTab,
            custom: CustomTab) -> dict:
    rec = engine.history.get(record_id) if record_id else None
    if rec is None:
        raise gr.Error("履歴が見つかりません。")
    p = rec["params"]
    tab = {"base": clone, "voice_design": design, "custom_voice": custom}[rec["kind"]]
    values = tab.controls.restore(p)
    if rec["kind"] == "base":
        ref = p["ref_audio"] if p["ref_audio"] and Path(p["ref_audio"]).exists() else None
        if ref is None:
            gr.Warning("参照音声のファイルが見つかりません。参照音声を指定し直してください。")
        voice = engine.speakers.find_by_ref(p["ref_audio"], p["ref_text"], p["x_vector_only"]) if ref else None
        values.update({clone.voice: gr.Dropdown(choices=voice_choices(engine.speakers),
                                                value=voice["id"] if voice else None),
                       clone.ref_audio: ref, clone.ref_text: p["ref_text"] or "",
                       clone.x_vector_only: p["x_vector_only"]})
    elif rec["kind"] == "voice_design":
        values[design.instruct] = p["instruct"] or ""
    else:
        values.update({custom.speaker: p["speaker"], custom.instruct: p["instruct"] or ""})
    values[tabs] = gr.Tabs(selected=TAB_OF_KIND[rec["kind"]])
    return values


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
    engine = Engine(settings, models, HistoryStore(settings.db_path), SpeakerStore(settings.db_path, settings.refs_dir))
    for kind, size in settings.preload:
        models.get(kind, size, "sdpa")

    build_app(engine).launch(server_name=args.host, server_port=args.port,
                             allowed_paths=[str(settings.data_dir)])


if __name__ == "__main__":
    main()
