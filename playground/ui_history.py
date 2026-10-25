"""History tab (F-70 .. F-73)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gradio as gr

from .engine import Engine
from .i18n import t
from .ui_common import KIND_LABELS, LANGUAGE_CHOICES, clip, speaker_label

HEADERS = ["日時 / Time", "モデル / Model", "サイズ / Size", "言語 / Language", "テキスト / Text",
           "話者・説明 / Voice", "seed", "音声 s / Audio s", "RTF", "アクセス元 / Client"]
ALL = "すべて / All"
OUTSIDE_LIBRARY = "（ライブラリ外 / not in library）"
COMPARE_MAX = 4
PICKED = t("比較に選んだ件数:", "Picked for comparison:")


@dataclass
class HistoryTab:
    refresh: Callable  # (*filters) -> (rows, ids)
    filters: list
    refresh_outputs: list
    open_button: gr.Button  # "open with these conditions": the caller restores the form
    selected: gr.State  # history record id
    compare_button: gr.Button  # "show in the compare tab": the caller fills the compare slots (F-74)
    compare_ids: gr.State  # record ids picked for comparison (up to COMPARE_MAX)


def history_tab(engine: Engine) -> HistoryTab:
    store = engine.history
    with gr.Row():
        kind = gr.Dropdown([(ALL, "")] + [(v, k) for k, v in KIND_LABELS.items()], value="",
                           label=t("モデル", "Model"))
        language = gr.Dropdown([(ALL, "")] + LANGUAGE_CHOICES, value="", label=t("言語", "Language"))
        text = gr.Textbox(label=t("テキストを含む", "Text contains"))
        date_from = gr.Textbox(label=t("開始日", "From"), placeholder="2026-10-01")
        date_to = gr.Textbox(label=t("終了日", "To"), placeholder="2026-10-31")
    search = gr.Button(t("検索", "Search"))
    ids = gr.State([])
    table = gr.Dataframe(headers=HEADERS, interactive=False, wrap=True, max_height=420)
    selected = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(label=t("選択した結果", "Selected result"), type="filepath", interactive=False,
                             buttons=["download"])
            with gr.Row():
                open_button = gr.Button(t("この条件で開く", "Open with these conditions"), variant="primary",
                                        interactive=False)
                delete = gr.Button(t("この結果を削除", "Delete this result"), variant="stop", interactive=False)
            with gr.Row():
                add = gr.Button(t("比較に追加", "Add to comparison"), interactive=False)
                clear = gr.Button(t("選択をクリア", "Clear picks"))
                compare_button = gr.Button(t(f"比較タブに並べる（{COMPARE_MAX} 件まで）",
                                             f"Show in the Compare tab (up to {COMPARE_MAX})"), interactive=False)
            picked = gr.Markdown()
        with gr.Column(scale=1):
            detail = gr.JSON(label=t("条件", "Conditions"))

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
    compare_ids = gr.State([])

    def picked_view(picks):
        return str(PICKED) + f" {len(picks)} / {COMPARE_MAX}", gr.Button(interactive=len(picks) >= 2)

    def do_add(picks, record_id):
        if record_id and record_id not in picks:
            picks = (picks + [record_id])[-COMPARE_MAX:]
        return picks, *picked_view(picks)

    table.select(select, ids, [selected, audio, detail, open_button, delete], queue=False)
    table.select(lambda: gr.Button(interactive=True), None, add, queue=False)
    add.click(do_add, [compare_ids, selected], [compare_ids, picked, compare_button], queue=False)
    clear.click(lambda: ([], *picked_view([])), None, [compare_ids, picked, compare_button], queue=False)
    delete.click(remove, [selected, *filters], [table, ids, selected, audio, detail, open_button, delete],
                 queue=False)
    return HistoryTab(rows, filters, [table, ids], open_button, selected, compare_button, compare_ids)


def record_summary(r: dict) -> str:
    """One history record as Markdown for a compare slot: its conditions, then the metrics."""
    p = r["params"]
    items = {"model": f"{KIND_LABELS[r['kind']]} {r['size']}", "voice": _who(r), "seed": r["seed"],
             **{k: p["gen"].get(k) for k in ("temperature", "top_k", "top_p", "repetition_penalty")}}
    conditions = ", ".join(f"**{k}**: {v}" for k, v in items.items() if v not in (None, ""))
    return (f"{conditions}\n\n> {clip(r['text'], 80)}\n\n"
            f"`{r['created_at'].replace('T', ' ')}` · {r['audio_sec']:.2f} s · RTF {r['rtf']:.2f}")


def _who(r: dict) -> str:
    if r["kind"] == "custom_voice":
        return speaker_label(r["speaker"])
    if r["kind"] == "base":
        return r["speaker"] or OUTSIDE_LIBRARY
    return r["instruct"] or ""
