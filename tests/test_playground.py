from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest

from playground.audio import loudness_normalize, store_by_hash
from playground.config import Settings
from playground.engine import Engine, GenRequest
from playground.history import HistoryStore
from playground.model_manager import ModelManager

SR = 24000


def _tone(sec: float, amp: float) -> np.ndarray:
    t = np.arange(int(SR * sec)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# ---- audio ----------------------------------------------------------------------------------

def test_loudness_normalize_reaches_target():
    out = loudness_normalize(_tone(3.0, 0.05), SR, target_lufs=-20.0)
    assert pyln.Meter(SR).integrated_loudness(out) == pytest.approx(-20.0, abs=0.5)


def test_loudness_normalize_limits_peak():
    out = loudness_normalize(_tone(3.0, 0.05), SR, target_lufs=0.0)
    assert np.max(np.abs(out)) <= 10 ** (-1 / 20) + 1e-6


def test_loudness_normalize_short_and_silent():
    short = loudness_normalize(_tone(0.2, 0.1), SR)
    assert np.max(np.abs(short)) == pytest.approx(10 ** (-1 / 20), rel=1e-3)
    silent = np.zeros(SR, dtype=np.float32)
    assert np.array_equal(loudness_normalize(silent, SR), silent)


def test_store_by_hash_deduplicates(tmp_path):
    src = tmp_path / "a.WAV"
    src.write_bytes(b"same")
    p1, sha1 = store_by_hash(src, tmp_path / "refs")
    other = tmp_path / "b.wav"
    other.write_bytes(b"same")
    p2, sha2 = store_by_hash(other, tmp_path / "refs")
    assert p1 == p2 and sha1 == sha2 and p1.suffix == ".wav"


# ---- history --------------------------------------------------------------------------------

def _record(i: int, **kw) -> dict:
    return {"id": f"r{i}", "created_at": f"2026-10-{20 + i:02d}T10:00:00", "kind": "base", "size": "1.7B",
            "language": "japanese", "text": f"テキスト{i}", "audio_path": f"/nonexistent/{i}.wav",
            "x_vector_only": False, "normalize": True, "params": {"seed": i}, **kw}


def test_history_search_and_delete(tmp_path):
    h = HistoryStore(tmp_path / "db.sqlite3")
    wav = tmp_path / "r2.wav"
    wav.write_bytes(b"x")
    h.add(_record(1))
    h.add(_record(2, kind="custom_voice", language="english", text="hello", audio_path=str(wav)))
    h.add(_record(3))

    assert [r["id"] for r in h.search()] == ["r3", "r2", "r1"]
    assert [r["id"] for r in h.search(kind="base")] == ["r3", "r1"]
    assert [r["id"] for r in h.search(language="english")] == ["r2"]
    assert [r["id"] for r in h.search(text="キスト1")] == ["r1"]
    assert [r["id"] for r in h.search(date_from="2026-10-22", date_to="2026-10-22")] == ["r2"]
    assert h.get("r1")["params"] == {"seed": 1} and h.get("r1")["normalize"] is True

    assert h.delete("r2") and not wav.exists()
    assert h.get("r2") is None and not h.delete("r2")


# ---- model manager --------------------------------------------------------------------------

class FakeLoader:
    def __init__(self):
        self.loads = []

    def __call__(self, model_id, attn):
        self.loads.append(model_id)
        return {"id": model_id}, {"load_sec": 0.1}


def _manager(budget: float) -> tuple[ModelManager, FakeLoader]:
    loader = FakeLoader()
    return ModelManager(budget, loader=loader, allocated_gib=lambda: 0.0), loader


def test_manager_reuses_loaded_model():
    m, loader = _manager(18)
    _, loaded1 = m.get("base", "1.7B", "sdpa")
    _, loaded2 = m.get("base", "1.7B", "sdpa")
    assert (loaded1, loaded2) == (True, False) and len(loader.loads) == 1
    _, loaded3 = m.get("base", "1.7B", "eager")  # attn is part of the key
    assert loaded3 and len(loader.loads) == 2


def test_manager_evicts_least_recently_used():
    m, _ = _manager(10)  # estimates: base 1.7B 4.9, voice_design 4.5, custom 1.7B 4.4
    m.get("base", "1.7B", "sdpa")
    m.get("voice_design", "1.7B", "sdpa")
    m.get("base", "1.7B", "sdpa")  # base becomes most recently used
    m.get("custom_voice", "1.7B", "sdpa")  # 4.9 + 4.5 + 4.4 > 10 -> evict voice_design
    assert [e.key[0] for e in m.loaded()] == ["custom_voice", "base"]


def test_manager_unload_all_keeps_one():
    m, _ = _manager(18)
    m.get("base", "0.6B", "sdpa")
    m.get("custom_voice", "0.6B", "sdpa")
    m.unload_all(keep=("base", "0.6B", "sdpa"))
    assert [e.key for e in m.loaded()] == [("base", "0.6B", "sdpa")]


def test_manager_rejects_unknown_model():
    m, _ = _manager(18)
    with pytest.raises(ValueError):
        m.get("voice_design", "0.6B", "sdpa")


# ---- engine validation ----------------------------------------------------------------------

@pytest.fixture
def engine(tmp_path):
    settings = Settings(data_dir=tmp_path, metrics_path=tmp_path / "metrics.jsonl", max_chars=10)
    return Engine(settings, ModelManager(18, loader=None), HistoryStore(settings.db_path))


@pytest.mark.parametrize("req, msg", [
    (GenRequest(kind="voice_design", size="0.6B", text="a"), "モデル"),
    (GenRequest(kind="custom_voice", speaker="ryan", text="  "), "テキスト"),
    (GenRequest(kind="custom_voice", speaker="ryan", text="a" * 11), "長すぎ"),
    (GenRequest(kind="custom_voice", speaker="ryan", text="a", language="klingon"), "言語"),
    (GenRequest(kind="custom_voice", speaker="ryan", text="a", gen={"min_p": 0.1}), "min_p"),
    (GenRequest(kind="custom_voice", speaker="nobody", text="a"), "話者"),
    (GenRequest(kind="base", text="a"), "参照音声"),
    (GenRequest(kind="base", text="a", ref_audio="x.wav"), "書き起こし"),
])
def test_validate_rejects(engine, req, msg):
    with pytest.raises(ValueError, match=msg):
        engine.validate(req)


def test_validate_accepts_and_warns(engine):
    assert engine.validate(GenRequest(kind="base", text="a", ref_audio="x.wav", x_vector_only=True)) == []
    assert engine.validate(GenRequest(kind="custom_voice", speaker="Ono_Anna", language="Japanese", text="a")) == []
    warnings = engine.validate(GenRequest(kind="custom_voice", size="0.6B", speaker="ryan", text="a",
                                          instruct="happily"))
    assert len(warnings) == 1


def test_estimate_seconds(engine):
    assert engine.estimate_seconds("あ" * 70) == pytest.approx(10 * 1.5)
    assert engine.estimate_seconds("a" * 140) == pytest.approx(10 * 1.5)


# ---- speaker library ------------------------------------------------------------------------

@pytest.fixture
def speakers(tmp_path):
    from playground.speakers import SpeakerStore
    return SpeakerStore(tmp_path / "db.sqlite3", tmp_path / "refs")


def _ref(tmp_path, name="ref.wav", content=b"audio"):
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_speakers_add_list_find(speakers, tmp_path):
    a = speakers.add("Alice", "clone", _ref(tmp_path), " こんにちは ", False, memo="memo")
    assert a["ref_text"] == "こんにちは" and a["ref_audio"].startswith(str(tmp_path / "refs"))
    b = speakers.add("Bob", "voice_design", _ref(tmp_path, "b.wav", b"b"), None, True, design={"instruct": "x"})
    assert b["ref_text"] is None and b["design"] == {"instruct": "x"}
    assert [s["name"] for s in speakers.list()] == ["Alice", "Bob"]
    assert speakers.find_by_ref(a["ref_audio"], "こんにちは", False)["id"] == a["id"]
    assert speakers.find_by_ref(a["ref_audio"], "別の文", False) is None
    assert speakers.find_by_ref(b["ref_audio"], None, True)["id"] == b["id"]


def test_speakers_validation(speakers, tmp_path):
    speakers.add("Alice", "clone", _ref(tmp_path), "text", False)
    with pytest.raises(ValueError, match="既に"):
        speakers.add(" Alice ", "clone", _ref(tmp_path), "text", False)
    with pytest.raises(ValueError, match="名前"):
        speakers.add(" ", "clone", _ref(tmp_path), "text", False)
    with pytest.raises(ValueError, match="x-vector"):
        speakers.add("Carol", "clone", _ref(tmp_path), "", False)


def test_speakers_update_delete(speakers, tmp_path):
    a = speakers.add("Alice", "clone", _ref(tmp_path), "text", False)
    speakers.add("Bob", "clone", _ref(tmp_path, "b.wav", b"b"), "text", False)
    with pytest.raises(ValueError, match="既に"):
        speakers.update(a["id"], "Bob", "")
    assert speakers.update(a["id"], "Alicia", "new")["memo"] == "new"
    assert speakers.delete(a["id"]) and speakers.get(a["id"]) is None
    assert Path(a["ref_audio"]).exists()  # reference audio stays for history


def test_save_voice_from_history(tmp_path):
    from playground.speakers import SpeakerStore
    settings = Settings(data_dir=tmp_path, metrics_path=tmp_path / "metrics.jsonl")
    history = HistoryStore(settings.db_path)
    engine = Engine(settings, ModelManager(18, loader=None), history,
                    SpeakerStore(settings.db_path, settings.refs_dir))
    out = _ref(tmp_path, "design.wav", b"designed")
    history.add({**_record(1, kind="voice_design", instruct="低い声", seed=5, audio_path=str(out),
                           x_vector_only=None), "params": {"gen": {"temperature": 0.9}}})
    spk = engine.save_voice("r1", "Narrator")
    assert spk["source"] == "voice_design" and spk["ref_text"] == "テキスト1"
    assert spk["design"]["instruct"] == "低い声" and spk["design"]["seed"] == 5
    ref = _ref(tmp_path, "mine.wav", b"mine")
    history.add(_record(2, ref_audio=str(ref), ref_text="参照", x_vector_only=False))
    assert engine.save_voice("r2", "Me")["source"] == "clone"
    history.add(_record(3, kind="custom_voice"))
    with pytest.raises(ValueError, match="CustomVoice"):
        engine.save_voice("r3", "X")
    with pytest.raises(ValueError, match="見つかりません"):
        engine.save_voice("nope", "X")


def test_prompt_path_depends_on_size_and_reference(engine):
    a = GenRequest(kind="base", text="a", ref_audio="/x/abc.wav", ref_text="t")
    b = GenRequest(kind="base", text="b", ref_audio="/y/abc.wav", ref_text="t")  # same content-addressed file
    assert engine._prompt_path(a) == engine._prompt_path(b)
    assert engine._prompt_path(a) != engine._prompt_path(GenRequest(**{**a.__dict__, "size": "0.6B"}))
    assert engine._prompt_path(a) != engine._prompt_path(GenRequest(**{**a.__dict__, "ref_text": "u"}))


# ---- reference pre-processing ---------------------------------------------------------------

def test_prepare_reference_trims_and_normalizes(tmp_path):
    from playground.audio import prepare_reference
    import soundfile as sf
    silence = np.zeros(SR, dtype=np.float32)
    src = tmp_path / "ref.wav"
    sf.write(src, np.concatenate([silence, _tone(2.0, 0.05), silence]), SR)

    as_is, sec = prepare_reference(src, tmp_path / "refs")
    assert sec == pytest.approx(4.0, abs=0.01)
    trimmed, sec = prepare_reference(src, tmp_path / "refs", trim=True, normalize=True, target_lufs=-20)
    assert sec == pytest.approx(2.2, abs=0.15)  # 2 s of tone + 0.1 s margin each side (frame-level precision)
    wav, _ = sf.read(trimmed)
    assert pyln.Meter(SR).integrated_loudness(wav) == pytest.approx(-20, abs=1.0)
    again, _ = prepare_reference(src, tmp_path / "refs", trim=True, normalize=True, target_lufs=-20)
    assert again == trimmed  # content-addressed


def test_split_sentences_and_batch_lines():
    from playground.text import batch_lines, split_sentences
    assert split_sentences("一つ目。二つ目！\n三つ目 3.5 です? Last one. ok") == \
        ["一つ目。", "二つ目！", "三つ目 3.5 です?", "Last one.", "ok"]
    assert batch_lines(" a \n\n b\n") == ["a", "b"]


def test_messages_have_unique_keys():
    from playground.messages import MESSAGES
    assert len({m.ja for m in MESSAGES}) == len(MESSAGES)


# ---- engine with a fake model ---------------------------------------------------------------

class FakeTTS:
    """Returns 0.1 s of audio per character; records the texts it was called with."""

    device = "cpu"

    def __init__(self, on_call=None):
        self.calls = []
        self.on_call = on_call

    def generate_custom_voice(self, text, language, speaker, instruct, non_streaming_mode, **gen):
        self.calls.append(text)
        if self.on_call:
            self.on_call(len(self.calls))
        texts = text if isinstance(text, list) else [text]
        return [np.full(int(SR * 0.1 * len(t)), 0.1, dtype=np.float32) for t in texts], SR


needs_cuda = pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="timing uses torch.cuda")


@pytest.fixture
def fake_engine(tmp_path):
    fake = FakeTTS()
    settings = Settings(data_dir=tmp_path, metrics_path=tmp_path / "metrics.jsonl")
    models = ModelManager(18, loader=lambda model_id, attn: (fake, {"load_sec": 0.0}), allocated_gib=lambda: 0.0)
    return Engine(settings, models, HistoryStore(settings.db_path)), fake


@needs_cuda
def test_split_generates_per_sentence_and_joins(fake_engine):
    import soundfile as sf
    engine, fake = fake_engine
    r = engine.generate(GenRequest(kind="custom_voice", speaker="ryan", text="あ。いい。ううう。", split=True,
                                   pause_sec=0.5, normalize=False))
    assert fake.calls[1:] == ["あ。", "いい。", "ううう。"]  # calls[0] is the warm-up
    assert r.audio_sec == pytest.approx((2 + 3 + 4) * 0.1 + 2 * 0.5)
    assert len(sf.read(r.audio_path)[0]) == int(round(r.audio_sec * SR))
    assert [h["text"] for h in engine.history.search()] == ["あ。いい。ううう。"]


@needs_cuda
def test_batch_makes_one_record_per_line(fake_engine):
    engine, fake = fake_engine
    results = engine.generate_batch(GenRequest(kind="custom_voice", speaker="ryan", text="ab\n\ncdef\n", batch=True))
    assert fake.calls[-1] == ["ab", "cdef"]
    assert [r.text for r in results] == ["ab", "cdef"]
    assert [r.audio_sec for r in results] == pytest.approx([0.2, 0.4])
    records = engine.history.search()
    assert {r["params"]["batch_id"] for r in records} == {results[0].id} and len(records) == 2


@needs_cuda
def test_cancel_stops_before_next_sentence(fake_engine):
    from playground.messages import Cancelled
    engine, fake = fake_engine
    fake.on_call = lambda n: engine.cancel("session") if n == 2 else None  # during the first sentence
    with pytest.raises(Cancelled):
        engine.generate(GenRequest(kind="custom_voice", speaker="ryan", text="一。二。三。", split=True),
                        cancel_token="session")
    assert len(fake.calls) == 2 and engine.history.search() == []
    engine.generate(GenRequest(kind="custom_voice", speaker="ryan", text="一。二。", split=True),
                    cancel_token="session")  # the flag does not leak into the next run


def test_validate_long_text_modes(engine):
    with pytest.raises(ValueError, match="同時"):
        engine.validate(GenRequest(kind="custom_voice", speaker="ryan", text="a", split=True, batch=True))
    engine.settings.max_chars = 100
    with pytest.raises(ValueError, match="行数"):
        engine.validate(GenRequest(kind="custom_voice", speaker="ryan", text="\n".join("a" * 9), batch=True))


def test_parse_seed():
    from playground.ui_common import parse_seed
    assert parse_seed("") is None and parse_seed(None) is None and parse_seed("  ") is None
    assert parse_seed(" 42 ") == 42 and parse_seed(0) is None  # 0 from an old gr.Number would be ambiguous
    with pytest.raises(ValueError, match="seed"):
        parse_seed("-1")
    with pytest.raises(ValueError, match="seed"):
        parse_seed("abc")
