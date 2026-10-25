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
