"""UI translations, Japanese and English (F-90).

The English text is written next to the Japanese at each call site: t("話者", "Speaker"). The browser language
(or Gradio's Settings panel at the bottom of the page) picks the language.

Limits of gr.I18n found in testing (Gradio 6.17):
- Translated: labels, info, placeholders, button values, tab / accordion labels, Markdown (also returned values).
- Not translated: Dropdown / Radio choice labels and Dataframe headers, so those are bilingual plain text.
- A string may contain only one translated part; more than one shows raw markers. Variable parts (numbers,
  names) are appended to a single translated part as plain text.
- Translations are sent when the page loads, so every key must be registered before launch: call t() at import
  or while building the UI, never only inside an event handler.
- Toasts (gr.Info / Warning / Error) and progress texts are not translated either. They are rendered on the
  server in the browser's language (Accept-Language), which is also what Gradio uses for the page by default.
"""

from __future__ import annotations

import hashlib

import gradio as gr
from gradio.i18n import I18nData

from .messages import MESSAGES, Msg

_JA: dict[str, str] = {}
_EN: dict[str, str] = {}


def _key(ja: str) -> str:
    # Hashed: svelte-i18n may treat dots in a key as nested paths, and many texts contain dots.
    return "t" + hashlib.sha1(ja.encode()).hexdigest()[:12]


def t(ja: str, en: str) -> I18nData:
    key = _key(ja)
    if _EN.get(key, en) != en:
        raise ValueError(f"conflicting English text for {ja!r}: {_EN[key]!r} vs {en!r}")
    _JA[key], _EN[key] = ja, en
    return I18nData(key)


def tm(m: Msg, detail: str = "") -> str:
    """A message (with an untranslated `detail` suffix) for Markdown, which translates it in the browser."""
    return str(t(m.ja, m.en)) + detail


def request_lang(request: gr.Request | None) -> str:
    accept = (request.headers.get("accept-language") or "") if request else ""
    return "ja" if accept.lower().startswith("ja") else "en"


def say(m: Msg, request: gr.Request | None, detail: str = "") -> str:
    """Plain text of a message in the requesting browser's language, for toasts and progress."""
    return (m.ja if request_lang(request) == "ja" else m.en) + detail


def build() -> gr.I18n:
    for m in MESSAGES:
        t(m.ja, m.en)
    return gr.I18n(ja=dict(_JA), en=dict(_EN))
