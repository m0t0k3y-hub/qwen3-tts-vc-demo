"""Pre-generated CustomVoice speaker samples for the preview players (F-41).

Example:
  uv run playground/samples.py            # generate the missing samples (9 speakers x Japanese / English)
  uv run playground/samples.py --force    # regenerate all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tts_core import SPEAKERS

from playground import messages as M
from playground.config import Settings
from playground.engine import Engine, GenRequest, Progress

SAMPLE_TEXTS = {
    "japanese": "こんにちは。私の声はこんな感じです。どうぞよろしくお願いします。",
    "english": "Hello! This is what my voice sounds like. Nice to meet you.",
}
SAMPLE_SEED = 0


def sample_path(settings: Settings, speaker: str, language: str) -> Path:
    return settings.samples_dir / f"{speaker}_{language}.wav"


def generate_samples(engine: Engine, force: bool = False,
                     progress: Progress = lambda m, detail="", fraction=None: print(m.en + detail)) -> int:
    """Generate missing samples with CustomVoice 1.7B (not recorded in history). Returns how many were made."""
    todo = [(s, lang) for s in SPEAKERS for lang in SAMPLE_TEXTS
            if force or not sample_path(engine.settings, s, lang).exists()]
    for i, (speaker, lang) in enumerate(todo, 1):
        progress(M.MAKING_SAMPLES, f" {i} / {len(todo)} ({speaker}, {lang})", (i - 1) / len(todo))
        req = GenRequest(kind="custom_voice", speaker=speaker, language=lang, text=SAMPLE_TEXTS[lang],
                         seed=SAMPLE_SEED)
        engine.generate(req, record=False, out_path=sample_path(engine.settings, speaker, lang))
    return len(todo)


def main():
    from playground.history import HistoryStore
    from playground.model_manager import ModelManager

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Settings.data_dir)
    p.add_argument("--force", action="store_true", help="regenerate existing samples too")
    args = p.parse_args()

    settings = Settings(data_dir=args.data_dir)
    engine = Engine(settings, ModelManager(settings.vram_budget_gib), HistoryStore(settings.db_path))
    n = generate_samples(engine, force=args.force)
    print(f"{n} samples generated in {settings.samples_dir}")


if __name__ == "__main__":
    main()
