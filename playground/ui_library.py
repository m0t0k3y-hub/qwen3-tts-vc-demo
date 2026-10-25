"""Speaker library tab (F-50 .. F-55)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gradio as gr

from .engine import Engine
from .speakers import SOURCES
from .ui_common import clip

HEADERS = ["名前", "作成元", "メモ", "書き起こし", "作成日時"]


@dataclass
class LibraryTab:
    refresh: Callable  # () -> (rows, ids)
    refresh_outputs: list
    use: gr.Button  # "read with this voice": the caller switches to the clone tab
    selected: gr.State  # speaker id


def library_tab(engine: Engine) -> LibraryTab:
    store = engine.speakers
    gr.Markdown("クローンした声と VoiceDesign で作った声を、全員で共有して使い回せます。"
                "声はボイスクローンタブやVoiceDesign タブの生成結果の下から保存できます。")
    ids = gr.State([])
    table = gr.Dataframe(headers=HEADERS, interactive=False, wrap=True, max_height=360)
    selected = gr.State(None)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(label="参照音声", type="filepath", interactive=False)
            ref_text = gr.Textbox(label="書き起こし", interactive=False, lines=2)
            design = gr.JSON(label="VoiceDesign の条件", visible=False)
        with gr.Column():
            name = gr.Textbox(label="名前")
            memo = gr.Textbox(label="メモ", lines=2)
            with gr.Row():
                save = gr.Button("名前・メモを更新", interactive=False)
                delete = gr.Button("削除", variant="stop", interactive=False)
            use = gr.Button("この声で読み上げる", variant="primary", interactive=False)

    detail = [selected, audio, ref_text, design, name, memo, save, delete, use]

    def rows():
        speakers = store.list()
        data = [[s["name"], SOURCES[s["source"]], clip(s["memo"], 40),
                 "（x-vector only）" if s["x_vector_only"] else clip(s["ref_text"] or "", 40),
                 s["created_at"].replace("T", " ")] for s in speakers]
        return data, [s["id"] for s in speakers]

    def show(spk: dict | None):
        if spk is None:
            off = gr.Button(interactive=False)
            return None, None, "", gr.JSON(visible=False), "", "", off, off, off
        on = gr.Button(interactive=True)
        return (spk["id"], spk["ref_audio"], spk["ref_text"] or "（x-vector only）",
                gr.JSON(spk["design"], visible=spk["design"] is not None), spk["name"], spk["memo"], on, on, on)

    def select(ids, evt: gr.SelectData):
        return show(store.get(ids[evt.index[0]]))

    def update(speaker_id, name, memo):
        try:
            spk = store.update(speaker_id, name, memo)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        gr.Info("更新しました。")
        return *rows(), *show(spk)

    def remove(speaker_id):
        store.delete(speaker_id)
        gr.Info("削除しました。")
        return *rows(), *show(None)

    table.select(select, ids, detail, queue=False)
    save.click(update, [selected, name, memo], [table, ids, *detail], queue=False)
    delete.click(remove, selected, [table, ids, *detail], queue=False)
    return LibraryTab(rows, [table, ids], use, selected)
