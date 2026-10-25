"""History tab (F-70 .. F-73)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gradio as gr

from .engine import Engine
from .ui_common import KIND_LABELS, LANGUAGE_CHOICES, clip, speaker_label

HEADERS = ["日時", "モデル", "サイズ", "言語", "テキスト", "話者 / 説明", "seed", "音声 秒", "RTF", "アクセス元"]


@dataclass
class HistoryTab:
    refresh: Callable  # (*filters) -> (rows, ids)
    filters: list
    refresh_outputs: list
    open_button: gr.Button  # "open with these conditions": the caller restores the form
    selected: gr.State  # history record id


def history_tab(engine: Engine) -> HistoryTab:
    store = engine.history
    with gr.Row():
        kind = gr.Dropdown([("すべて", "")] + [(v, k) for k, v in KIND_LABELS.items()], value="", label="モデル")
        language = gr.Dropdown([("すべて", "")] + LANGUAGE_CHOICES, value="", label="言語")
        text = gr.Textbox(label="テキストを含む")
        date_from = gr.Textbox(label="開始日", placeholder="2026-10-01")
        date_to = gr.Textbox(label="終了日", placeholder="2026-10-31")
    search = gr.Button("検索")
    ids = gr.State([])
    table = gr.Dataframe(headers=HEADERS, interactive=False, wrap=True, max_height=420)
    selected = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(label="選択した結果", type="filepath", interactive=False, buttons=["download"])
            with gr.Row():
                open_button = gr.Button("この条件で開く", variant="primary", interactive=False)
                delete = gr.Button("この結果を削除", variant="stop", interactive=False)
        with gr.Column(scale=1):
            detail = gr.JSON(label="条件")

    def rows(kind, language, text, date_from, date_to):
        records = store.search(kind=kind or None, language=language or None, text=text or None,
                               date_from=date_from.strip() or None, date_to=date_to.strip() or None)
        data = [[r["created_at"].replace("T", " "), KIND_LABELS[r["kind"]], r["size"], r["language"],
                 clip(r["text"], 40), clip(_who(r), 30), r["seed"],
                 round(r["audio_sec"], 2), round(r["rtf"], 2), r["client_ip"] or ""] for r in records]
        return data, [r["id"] for r in records]

    def select(ids, evt: gr.SelectData):
        rec = store.get(ids[evt.index[0]])
        if rec is None:
            off = gr.Button(interactive=False)
            return None, None, None, off, off
        on = gr.Button(interactive=True)
        return rec["id"], rec["audio_path"], rec["params"], on, on

    def remove(record_id, *filters):
        if record_id:
            store.delete(record_id)
        off = gr.Button(interactive=False)
        return (*rows(*filters), None, None, None, off, off)

    filters = [kind, language, text, date_from, date_to]
    search.click(rows, filters, [table, ids], queue=False)
    table.select(select, ids, [selected, audio, detail, open_button, delete], queue=False)
    delete.click(remove, [selected, *filters], [table, ids, selected, audio, detail, open_button, delete],
                 queue=False)
    return HistoryTab(rows, filters, [table, ids], open_button, selected)


def _who(r: dict) -> str:
    if r["kind"] == "custom_voice":
        return speaker_label(r["speaker"])
    if r["kind"] == "base":
        return r["speaker"] or "（ライブラリ外の参照音声）"
    return r["instruct"] or ""
