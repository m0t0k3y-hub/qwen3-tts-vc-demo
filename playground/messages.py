"""User-facing messages from the non-UI modules, in Japanese and English (F-90).

Every message is a module-level constant so the UI can register all of them with gr.I18n before launch.
Variable parts (numbers, names) are passed separately as `detail` and appended untranslated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Msg:
    ja: str
    en: str


MESSAGES: list[Msg] = []


def msg(ja: str, en: str) -> Msg:
    m = Msg(ja, en)
    MESSAGES.append(m)
    return m


class UserError(ValueError):
    """A request the user can fix. str() is the Japanese text, for logs and tests."""

    def __init__(self, m: Msg, detail: str = ""):
        super().__init__(m.ja + detail)
        self.msg = m
        self.detail = detail


class Cancelled(Exception):
    pass


# ---- validation ------------------------------------------------------------------------------
UNKNOWN_MODEL = msg("このモデルはありません:", "No such model:")
TEXT_EMPTY = msg("合成するテキストを入力してください。", "Enter the text to synthesize.")
TEXT_TOO_LONG = msg("テキストが長すぎます。文字数 / 上限:", "The text is too long. Characters / limit:")
BAD_LANGUAGE = msg("対応していない言語です:", "Unsupported language:")
BAD_GEN_PARAMS = msg("不明な生成パラメータ:", "Unknown generation parameters:")
NO_REF_AUDIO = msg("参照音声を指定してください。", "Provide a reference audio.")
NO_REF_TEXT = msg("参照音声の書き起こしを入力してください（x-vector only モードでは不要）。",
                  "Enter the transcript of the reference audio (not needed in x-vector only mode).")
NO_SPEAKER = msg("話者を選んでください:", "Choose a speaker:")
SPLIT_AND_BATCH = msg("文分割とバッチは同時には使えません。", "Sentence splitting and batch mode cannot be combined.")
TOO_MANY_LINES = msg("バッチの行数が多すぎます。行数 / 上限:", "Too many lines for a batch. Lines / limit:")
REF_TOO_SHORT = msg("参照音声が短すぎます（5〜10 秒程度を推奨）。長さ:",
                    "The reference audio is short (5-10 s recommended). Length:")
REF_TOO_LONG = msg("参照音声が長すぎます（5〜10 秒程度を推奨）。長さ:",
                   "The reference audio is long (5-10 s recommended). Length:")
NO_AUDIO = msg("音声を指定してください。", "Provide an audio.")
BAD_SEED = msg("seed は 0 以上の整数で指定してください（空欄ならランダム）:",
               "The seed must be a non-negative integer (empty for random):")

# ---- warnings --------------------------------------------------------------------------------
INSTRUCT_IGNORED = msg("0.6B の CustomVoice はスタイル指示に対応していないため、指示は無視されました。",
                       "The 0.6B CustomVoice model does not support style instructions; the instruction was ignored.")

# ---- progress --------------------------------------------------------------------------------
LOADING_MODEL = msg("モデルをロードしています（初回はダウンロードに数分かかります）:",
                    "Loading the model (the first time includes a download of a few minutes):")
ANALYZING_REF = msg("参照音声を解析しています…", "Analyzing the reference audio…")
WARMING_UP = msg("ウォームアップ中…", "Warming up…")
GENERATING = msg("生成中… 推定秒数:", "Generating… estimated seconds:")
GENERATING_SENTENCE = msg("文ごとに生成中… 文:", "Generating sentence by sentence… sentence:")
RETRY_OOM = msg("VRAM が不足したため、他のモデルを解放して再試行しています…",
                "Out of VRAM; unloading the other models and retrying…")
TRANSCRIBING = msg("書き起こし中…", "Transcribing…")
MAKING_SAMPLES = msg("話者サンプルを生成中:", "Generating speaker samples:")

# ---- UI notifications ------------------------------------------------------------------------
SAVED = msg("話者ライブラリに保存しました:", "Saved to the speaker library:")
GENERATE_FIRST = msg("先に生成してください。", "Generate first.")
UPDATED = msg("更新しました。", "Updated.")
DELETED = msg("削除しました。", "Deleted.")
PICK_SPEAKER = msg("話者を選んでください。", "Choose a speaker.")
NO_HISTORY = msg("履歴が見つかりません。", "History record not found.")
REF_MISSING = msg("参照音声のファイルが見つかりません。参照音声を指定し直してください。",
                  "The reference audio file is missing. Provide the reference audio again.")

# ---- speaker library -------------------------------------------------------------------------
NAME_EMPTY = msg("話者の名前を入力してください。", "Enter a name for the speaker.")
NAME_TAKEN = msg("この名前の話者は既にあります。別の名前にしてください:", "A speaker with this name already exists:")
NEEDS_TRANSCRIPT = msg("書き起こしがない声は、x-vector only モードでのみ保存できます。",
                       "A voice without a transcript can only be saved in x-vector only mode.")
NO_RESULT = msg("保存する生成結果が見つかりません。先に生成してください。",
                "No generation result to save. Generate first.")
CANNOT_SAVE_CUSTOM = msg("CustomVoice の結果は話者ライブラリに保存できません。",
                         "CustomVoice results cannot be saved to the speaker library.")
