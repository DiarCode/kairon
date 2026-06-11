"""Smoke tests for the model store and config."""

from __future__ import annotations

import numpy as np

from kairon.config import KaironSettings
from kairon.models.contracts import FeatureMatrix
from kairon.models.linear import LinearConfig, LogisticRegressionModel
from kairon.store import ModelStore


def test_settings_loads_with_defaults() -> None:
    s = KaironSettings()
    assert s.log_level == "INFO"
    assert s.api_port == 8000
    assert s.max_position_equity_fraction == 0.20


def test_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("KAIRON_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("KAIRON_API_PORT", "9000")
    s = KaironSettings()
    assert s.log_level == "DEBUG"
    assert s.api_port == 9000


def test_settings_validates_log_level(monkeypatch) -> None:
    monkeypatch.setenv("KAIRON_LOG_LEVEL", "NOPE")
    import pytest
    with pytest.raises(Exception):
        KaironSettings()


def test_settings_overrides() -> None:
    s = KaironSettings(log_level="DEBUG")
    o = s.overrides()
    assert "log_level" in o
    assert o["log_level"] == "DEBUG"


# ---------------------------------------------------------------------------
# W1.4 — live venue, API keys, history_days
# ---------------------------------------------------------------------------
def test_settings_w1_4_defaults() -> None:
    """KAIRON_LIVE_VENUE and KAIRON_HISTORY_DAYS have safe defaults."""
    s = KaironSettings()
    assert s.live_venue == "binance"
    assert s.history_days == 30
    assert s.binance_api_key == ""
    assert s.binance_api_secret == ""
    assert s.polygon_api_key == ""


def test_settings_w1_4_env_override(monkeypatch) -> None:
    """All 5 W1.4 fields override from env vars."""
    monkeypatch.setenv("KAIRON_LIVE_VENUE", "bybit")
    monkeypatch.setenv("KAIRON_HISTORY_DAYS", "180")
    monkeypatch.setenv("KAIRON_BINANCE_API_KEY", "test_key")
    monkeypatch.setenv("KAIRON_BINANCE_API_SECRET", "test_secret")
    monkeypatch.setenv("KAIRON_POLYGON_API_KEY", "poly_key")
    s = KaironSettings()
    assert s.live_venue == "bybit"
    assert s.history_days == 180
    assert s.binance_api_key == "test_key"
    assert s.binance_api_secret == "test_secret"
    assert s.polygon_api_key == "poly_key"


def test_settings_w1_4_live_venue_validates_pattern(monkeypatch) -> None:
    """KAIRON_LIVE_VENUE only accepts binance|bybit|coinbase."""
    monkeypatch.setenv("KAIRON_LIVE_VENUE", "kraken")
    import pytest
    with pytest.raises(Exception):
        KaironSettings()


def test_settings_w1_4_history_days_validates_ge_1(monkeypatch) -> None:
    """KAIRON_HISTORY_DAYS must be >= 1."""
    monkeypatch.setenv("KAIRON_HISTORY_DAYS", "0")
    import pytest
    with pytest.raises(Exception):
        KaironSettings()


def test_settings_w1_4_overrides_includes_new_fields() -> None:
    """The overrides() helper includes the new W1.4 fields when set."""
    s = KaironSettings(history_days=90, live_venue="bybit")
    o = s.overrides()
    assert o.get("history_days") == 90
    assert o.get("live_venue") == "bybit"


def test_model_store_save_load(tmp_path) -> None:
    fm, y = _toy()
    m = LogisticRegressionModel(LinearConfig())
    trained = m.fit(fm, y)
    store = ModelStore(tmp_path / "models")
    art = store.save(trained, run_name="r1")
    assert art.run_name == "r1"
    assert art.meta_path.exists()
    assert art.state_path.exists()
    assert store.exists("r1")
    loaded = store.load("r1")
    assert loaded.backend == trained.backend
    assert loaded.feature_names == trained.feature_names
    # Reuse the loaded model
    pred = m.predict(loaded, fm)
    assert pred.y_class.shape == (fm.n_rows,)


def test_model_store_load_rejects_backend_mismatch(tmp_path) -> None:
    fm, y = _toy()
    m = LogisticRegressionModel(LinearConfig())
    trained = m.fit(fm, y)
    store = ModelStore(tmp_path / "models")
    store.save(trained, run_name="r1")
    import pytest
    with pytest.raises(ValueError, match="backend"):
        store.load("r1", backend="random_forest")


def test_model_store_load_missing_raises(tmp_path) -> None:
    store = ModelStore(tmp_path / "models")
    import pytest
    with pytest.raises(FileNotFoundError):
        store.load("missing")


def test_model_store_list_and_delete(tmp_path) -> None:
    fm, y = _toy()
    m = LogisticRegressionModel(LinearConfig())
    store = ModelStore(tmp_path / "models")
    store.save(m.fit(fm, y), run_name="a")
    store.save(m.fit(fm, y), run_name="b")
    runs = store.list_runs()
    assert "a" in runs
    assert "b" in runs
    assert store.delete("a")
    assert "a" not in store.list_runs()
    assert not store.delete("a")  # already gone


def test_model_store_list_empty(tmp_path) -> None:
    store = ModelStore(tmp_path / "missing_dir")
    assert store.list_runs() == ()


def _toy(n: int = 60, seed: int = 17) -> tuple[FeatureMatrix, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.5 * x2 + 0.05 * rng.normal(size=n) > 0).astype(np.int64)
    fm = FeatureMatrix(
        values=np.column_stack([x1, x2]).astype(np.float64),
        feature_names=("x1", "x2"),
    )
    return fm, y
