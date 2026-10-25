"""Speaker library tab (F-50 .. F-55)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gradio as gr

from .engine import Engine
from . import messages as M
from .i18n import say, t
from .messages import UserError
from .speakers import SOURCES
from .ui_common import clip, user_error

HEADERS = ["名前 / Name", "作成元 / Source", "メモ / Memo", "書き起こし / Transcript", "作成日時 / Created"]
XVEC_ONLY = "（x-vector only）"


@dataclass
class LibraryTab:
    refresh: Callable  # () -> (rows, ids)
    refresh_outputs: list
    use: gr.Button  # "read with this voice": the caller switches to the clone tab
    selected: gr.State  # speaker id


def library_tab(engine: Engine) -> LibraryTab:
    store = engine.speakers
    gr.Markdown(t("クローンした声と VoiceDesign で作った声を、全員で共有して使い回せます。"
                  "声はボイスクローンタブや VoiceDesign タブの生成結果の下から保存できます。",
                  "Voices cloned or made with VoiceDesign, shared by everyone. Save a voice from under a result "
                  "in the voice clone or VoiceDesign tab."))
    ids = gr.State([])
    table = gr.Dataframe(headers=HEADERS, interactive=False, wrap=True, max_height=360)
    selected = gr.State(None)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(label=t("参照音声", "Reference audio"), type="filepath", interactive=False)
            ref_text = gr.Textbox(label=t("書き起こし", "Transcript"), interactive=False, lines=2)
            design = gr.JSON(label=t("VoiceDesign の条件", "VoiceDesign conditions"), visible=False)
        with gr.Column():
            name = gr.Textbox(label=t("名前", "Name"))
            memo = gr.Textbox(label=t("メモ", "Memo"), lines=2)
            with gr.Row():
                save = gr.Button(t("名前・メモを更新", "Update name / memo"), interactive=False)
                delete = gr.Button(t("削除", "Delete"), variant="stop", interactive=False)
            use = gr.Button(t("この声で読み上げる", "Read with this voice"), variant="primary", interactive=False)

    detail = [selected, audio, ref_text, design, name, memo, save, delete, use]

    def rows():
        speakers = store.list()
        data = [[s["name"], SOURCES[s["source"]], clip(s["memo"], 40),
                 XVEC_ONLY if s["x_vector_only"] else clip(s["ref_text"] or "", 40),
                 s["created_at"].replace("T", " ")] for s in speakers]
        return data, [s["id"] for s in speakers]

    def show(spk: dict | None):
        if spk is None:
            off = gr.Button(interactive=False)
            return None, None, "", gr.JSON(visible=False), "", "", off, off, off
        on = gr.Button(interactive=True)
        return (spk["id"], spk["ref_audio"], spk["ref_text"] or XVEC_ONLY,
                gr.JSON(spk["design"], visible=spk["design"] is not None), spk["name"], spk["memo"], on, on, on)

    def select(ids, evt: gr.SelectData):
        return show(store.get(ids[evt.index[0]]))

    def update(speaker_id, name, memo, request: gr.Request):
        try:
            spk = store.update(speaker_id, name, memo)
        except UserError as e:
            raise user_error(e, request) from e
        gr.Info(say(M.UPDATED, request))
        return *rows(), *show(spk)

    def remove(speaker_id, request: gr.Request):
        store.delete(speaker_id)
        gr.Info(say(M.DELETED, request))
        return *rows(), *show(None)

    table.select(select, ids, detail, queue=False)
    save.click(update, [selected, name, memo], [table, ids, *detail], queue=False)
    delete.click(remove, selected, [table, ids, *detail], queue=False)
    return LibraryTab(rows, [table, ids], use, selected)
