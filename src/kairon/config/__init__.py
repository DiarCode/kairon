"""Application settings.

Uses :class:`pydantic_settings.BaseSettings` so configuration can be
overridden by environment variables (e.g. ``KAIRON_LOG_LEVEL=DEBUG``)
or a ``.env`` file. The defaults are tuned for local development; the
production readme documents the recommended overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class KaironSettings(BaseSettings):
    """Top-level settings for the Kairon runtime."""

    model_config = SettingsConfigDict(
        env_prefix="KAIRON_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    artifact_root: Path = Field(default=Path("./artifacts"))
    data_root: Path = Field(default=Path("./data"))
    cache_root: Path = Field(default=Path("./.cache"))

    # Logging
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    log_format: str = Field(default="json", pattern="^(json|text)$")

    # MLflow
    mlflow_tracking_uri: str = Field(default="./mlruns")
    mlflow_experiment: str = Field(default="kairon")

    # LLM
    ollama_host: str = Field(default="https://ollama.com")
    ollama_model: str = Field(default="gpt-oss:120b-cloud")
    ollama_timeout_seconds: float = Field(default=60.0, gt=0)

    # API
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_cors_origins: tuple[str, ...] = Field(default_factory=lambda: ("http://localhost:8000",))

    # Backtest defaults
    default_initial_equity: float = Field(default=10_000.0, gt=0)
    default_n_trials: int = Field(default=1, ge=1)
    default_commission_bps: float = Field(default=10.0, ge=0)
    default_slippage_bps: float = Field(default=2.0, ge=0)
    default_half_spread_bps: float = Field(default=2.0, ge=0)

    # Live venue (W1.4)
    live_venue: str = Field(
        default="binance",
        pattern="^(binance|bybit|coinbase)$",
    )
    binance_api_key: str = Field(default="")
    binance_api_secret: str = Field(default="")
    polygon_api_key: str = Field(default="")
    history_days: int = Field(default=30, ge=1)

    # Bybit live trading
    bybit_api_key: str = Field(default="")
    bybit_api_secret: str = Field(default="")
    bybit_testnet: bool = Field(default=True)

    # Live trading mode and risk
    live_dry_run: bool = Field(default=True)
    live_symbols: tuple[str, ...] = Field(
        default=("BTC-USDT-PERP", "ETH-USDT-PERP"),
    )
    live_timeframe: str = Field(default="1m", pattern=r"^[0-9]+(m|h|d|w)$")
    live_cadence_seconds: int = Field(default=60, ge=10)
    live_max_daily_loss_pct: float = Field(default=0.03, gt=0, le=1)
    live_max_open_positions: int = Field(default=5, ge=1)
    live_warmup_bars: int = Field(default=22, ge=1)
    live_reconcile_interval_seconds: int = Field(default=30, ge=5)
    live_reconcile_grace_seconds: int = Field(default=120, ge=10)
    live_model_path: str = Field(default="")
    live_horizon: str = Field(default="day", pattern="^(day|swing|long)$")
    live_promotion_ack: int = Field(default=0, ge=0, le=1)

    # Risk
    max_position_equity_fraction: float = Field(default=0.20, gt=0, le=1)
    max_total_leverage: float = Field(default=1.0, gt=0)
    max_positions: int = Field(default=10, ge=1)

    # Drift
    drift_method: str = Field(default="psi", pattern="^(psi|ks)$")
    drift_bins: int = Field(default=10, ge=2, le=50)
    drift_psi_warning: float = Field(default=0.10, gt=0)
    drift_psi_critical: float = Field(default=0.20, gt=0)
    drift_ks_warning: float = Field(default=0.05, gt=0)
    drift_ks_critical: float = Field(default=0.10, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _validate_live_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Fail-closed: if live_dry_run is True, bybit_testnet must also be True."""
        if values.get("live_dry_run", True) and not values.get("bybit_testnet", True):
            raise ValueError(
                "live_dry_run=True requires bybit_testnet=True. "
                "Set live_dry_run=False only after passing the promotion checklist."
            )
        return values

    def overrides(self) -> dict[str, Any]:
        """Return a dict of non-default values, for telemetry / mlflow."""
        defaults = KaironSettings()
        out: dict[str, Any] = {}
        for field in type(self).model_fields:
            v = getattr(self, field)
            d = getattr(defaults, field)
            if v != d:
                out[field] = v
        return out


__all__ = ["KaironSettings"]
