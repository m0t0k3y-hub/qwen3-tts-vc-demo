"""Qwen3-TTS playground (Gradio).

Example:
  uv run playground/app.py                       # http://localhost:7860 (also reachable from the LAN)
  uv run playground/app.py --preload none --vram-budget 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from playground import i18n
from playground.asr import ASR_KEY, PLAYGROUND_MODEL_IDS, load_asr
from playground.config import Settings
from playground.engine import Engine
from playground.history import HistoryStore
from playground import messages as M
from playground.i18n import say, t
from playground.model_manager import ModelManager
from playground.speakers import SpeakerStore
from playground.ui_common import voice_choices
from playground.ui_compare import compare_tab
from playground.ui_generate import CloneTab, CustomTab, DesignTab, clone_tab, custom_tab, design_tab
from playground.ui_history import history_tab, record_summary
from playground.ui_library import library_tab
from playground.ui_system import system_tab

TAB_OF_KIND = {"base": "clone", "voice_design": "design", "custom_voice": "custom"}


def build_app(engine: Engine) -> gr.Blocks:
    settings = engine.settings
    with gr.Blocks(title="Qwen3-TTS Playground") as demo:
        gr.Markdown("# Qwen3-TTS Playground")
        gr.Markdown(t("生成は GPU 1 枚で 1 件ずつ順に処理します。音声の長さの約 1.5 倍の時間がかかります。"
                      "表示言語は画面下の「設定」で切り替えられます。",
                      "Generations run one at a time on a single GPU and take about 1.5x the audio length. "
                      "Change the display language from Settings at the bottom of the page."))
        with gr.Tabs() as tabs:
            with gr.Tab(t("ボイスクローン", "Voice clone"), id="clone") as clone_t:
                clone = clone_tab(engine)
            with gr.Tab("VoiceDesign", id="design"):
                design = design_tab(engine)
            with gr.Tab("CustomVoice", id="custom") as custom_t:
                custom = custom_tab(engine)
            with gr.Tab(t("話者ライブラリ", "Speaker library"), id="library") as library_t:
                library = library_tab(engine)
            with gr.Tab(t("比較", "Compare"), id="compare") as compare_t:
                compare = compare_tab(engine)
            with gr.Tab(t("履歴", "History"), id="history") as history_t:
                history = history_tab(engine)
            with gr.Tab(t("システム", "System"), id="system") as system_t:
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
        demo.load(clone.enable_recording, None, [clone.ref_audio, clone.mic_note], queue=False)

        # Library -> clone tab with the voice filled in (F-54).
        def use_voice(speaker_id, request: gr.Request):
            spk = engine.speakers.get(speaker_id) if speaker_id is not None else None
            if spk is None:
                raise gr.Error(say(M.PICK_SPEAKER, request))
            return (gr.Tabs(selected="clone"), gr.Dropdown(choices=voice_choices(engine.speakers), value=spk["id"]),
                    spk["ref_audio"], spk["ref_text"] or "", spk["x_vector_only"])

        library.use.click(use_voice, library.selected,
                          [tabs, clone.voice, clone.ref_audio, clone.ref_text, clone.x_vector_only], queue=False)

        # History -> the model tab with the form restored (F-73).
        restore_outputs = {tabs, clone.voice, clone.ref_audio, clone.ref_text, clone.x_vector_only, design.instruct,
                           custom.speaker, custom.instruct, *clone.controls.inputs(), *design.controls.inputs(),
                           *custom.controls.inputs()}
        def open_record(record_id, request: gr.Request):
            return restore(engine, record_id, request, tabs, clone, design, custom)

        # History -> compare slots (F-74).
        def show_in_compare(record_ids):
            records = [engine.history.get(i) for i in record_ids]
            records = [r for r in records if r is not None][: len(compare.audios)]
            values = []
            for i in range(len(compare.audios)):
                r = records[i] if i < len(records) else None
                values += [r["audio_path"] if r else None, record_summary(r) if r else ""]
            return [gr.Tabs(selected="compare"), *values]

        history.compare_button.click(show_in_compare, history.compare_ids,
                                     [tabs, *[x for pair in zip(compare.audios, compare.results) for x in pair]],
                                     queue=False)

        history.open_button.click(open_record,
                                  history.selected, restore_outputs, queue=False)
    demo.queue(max_size=settings.queue_size, default_concurrency_limit=8)
    return demo


def restore(engine: Engine, record_id: str, request: gr.Request | None, tabs: gr.Tabs, clone: CloneTab,
            design: DesignTab, custom: CustomTab) -> dict:
    rec = engine.history.get(record_id) if record_id else None
    if rec is None:
        raise gr.Error(say(M.NO_HISTORY, request))
    p = rec["params"]
    tab = {"base": clone, "voice_design": design, "custom_voice": custom}[rec["kind"]]
    values = tab.controls.restore(p)
    if rec["kind"] == "base":
        ref = p["ref_audio"] if p["ref_audio"] and Path(p["ref_audio"]).exists() else None
        if ref is None:
            gr.Warning(say(M.REF_MISSING, request))
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
    models = ModelManager(settings.vram_budget_gib, model_ids=PLAYGROUND_MODEL_IDS, loaders={ASR_KEY[0]: load_asr})
    engine = Engine(settings, models, HistoryStore(settings.db_path), SpeakerStore(settings.db_path, settings.refs_dir))
    for kind, size in settings.preload:
        models.get(kind, size, "sdpa")

    demo = build_app(engine)  # registers the UI texts; i18n.build() must come after it
    demo.launch(server_name=args.host, server_port=args.port, allowed_paths=[str(settings.data_dir)],
                i18n=i18n.build())


if __name__ == "__main__":
    main()
