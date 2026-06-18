"""Tests for LiveConfig."""

from __future__ import annotations

import pytest

from kairon.config import KaironSettings
from kairon.live.config import LiveConfig


class TestLiveConfig:
    """Test LiveConfig creation and validation."""

    def test_defaults(self) -> None:
        config = LiveConfig()
        assert config.symbols == ("BTC-USDT-PERP", "ETH-USDT-PERP")
        assert config.dry_run is True
        assert config.bybit_testnet is True
        assert config.cadence_seconds == 60
        assert config.max_daily_loss_pct == 0.03
        assert config.max_open_positions == 5

    def test_grace_period_validation(self) -> None:
        # grace must be >= 2 * cadence
        with pytest.raises(Exception, match="reconcile_grace_seconds"):
            LiveConfig(cadence_seconds=60, reconcile_grace_seconds=60)  # too small

    def test_valid_grace_period(self) -> None:
        config = LiveConfig(cadence_seconds=60, reconcile_grace_seconds=120)
        assert config.reconcile_grace_seconds == 120

    def test_from_settings(self) -> None:
        settings = KaironSettings(
            live_symbols=("SOL-USDT-PERP",),
            live_timeframe="5m",
            live_cadence_seconds=30,
            live_dry_run=True,
            bybit_testnet=True,
        )
        config = LiveConfig.from_settings(settings)
        assert config.symbols == ("SOL-USDT-PERP",)
        assert config.timeframe == "5m"
        assert config.cadence_seconds == 30
        assert config.dry_run is True

    def test_from_settings_defaults(self) -> None:
        settings = KaironSettings()
        config = LiveConfig.from_settings(settings)
        assert config.symbols == ("BTC-USDT-PERP", "ETH-USDT-PERP")
        assert config.dry_run is True


class TestKaironSettingsLiveFields:
    """Test new KaironSettings fields for live trading."""

    def test_new_fields_exist(self) -> None:
        settings = KaironSettings()
        assert hasattr(settings, "bybit_api_key")
        assert hasattr(settings, "bybit_api_secret")
        assert hasattr(settings, "bybit_testnet")
        assert hasattr(settings, "live_dry_run")
        assert hasattr(settings, "live_symbols")
        assert hasattr(settings, "live_timeframe")
        assert hasattr(settings, "live_cadence_seconds")
        assert hasattr(settings, "live_max_daily_loss_pct")
        assert hasattr(settings, "live_max_open_positions")
        assert hasattr(settings, "live_warmup_bars")
        assert hasattr(settings, "live_reconcile_interval_seconds")
        assert hasattr(settings, "live_reconcile_grace_seconds")
        assert hasattr(settings, "live_model_path")
        assert hasattr(settings, "live_horizon")
        assert hasattr(settings, "live_promotion_ack")

    def test_defaults(self) -> None:
        settings = KaironSettings()
        assert settings.bybit_testnet is True
        assert settings.live_dry_run is True
        assert settings.live_symbols == ("BTC-USDT-PERP", "ETH-USDT-PERP")
        assert settings.live_timeframe == "1m"
        assert settings.live_cadence_seconds == 60
        assert settings.live_max_daily_loss_pct == 0.03
        assert settings.live_max_open_positions == 5
        assert settings.live_warmup_bars == 22

    def test_live_dry_run_requires_testnet(self) -> None:
        """Fail-closed: live_dry_run=True forces bybit_testnet=True."""
        with pytest.raises(Exception, match="live_dry_run=True requires bybit_testnet=True"):
            KaironSettings(live_dry_run=True, bybit_testnet=False)

    def test_live_dry_run_false_allows_mainnet(self) -> None:
        """Explicit mainnet opt-in when live_dry_run=False."""
        settings = KaironSettings(live_dry_run=False, bybit_testnet=False)
        assert settings.bybit_testnet is False
        assert settings.live_horizon == "day"
        assert settings.live_promotion_ack == 0