"""Live trading configuration.

Separate from :class:`KaironSettings` (which is the global runtime config)
to keep live-trading-specific parameters in one typed place. Can be
constructed from :class:`KaironSettings` via :meth:`LiveConfig.from_settings`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from kairon.config import KaironSettings


class LiveConfig(BaseModel, frozen=True, strict=True):
    """Configuration for the live trading loop.

    Constructed from :class:`KaironSettings` or instantiated directly for
    tests. All fields have sensible defaults for testnet paper-trading.
    """

    model_config = {"extra": "forbid"}

    # Venue
    symbols: tuple[str, ...] = Field(
        default=("BTC-USDT-PERP", "ETH-USDT-PERP"),
        min_length=1,
        description="Canonical symbols to trade, e.g. ('BTC-USDT-PERP',).",
    )
    timeframe: str = Field(
        default="1m",
        pattern=r"^[0-9]+(m|h|d|w)$",
        description="Candle timeframe for the feed.",
    )
    cadence_seconds: int = Field(
        default=60,
        ge=10,
        description="Seconds between tick loop iterations.",
    )

    # Risk
    max_daily_loss_pct: float = Field(
        default=0.03,
        gt=0,
        le=1,
        description="Max daily loss as fraction of equity before halt.",
    )
    max_open_positions: int = Field(
        default=5,
        ge=1,
        description="Max number of simultaneous open positions.",
    )
    warmup_bars: int = Field(
        default=22,
        ge=1,
        description="Number of bars to skip before trading (feature pipeline warm-up).",
    )

    # Model
    model_path: str = Field(
        default="",
        description="Path to pre-trained model artifact. Empty = fail-closed.",
    )
    horizon: str = Field(
        default="day",
        pattern="^(day|swing|long)$",
        description="Horizon for model loading.",
    )

    # Reconciliation
    reconcile_interval_seconds: int = Field(
        default=30,
        ge=5,
        description="Seconds between reconciler runs.",
    )
    reconcile_grace_seconds: int = Field(
        default=120,
        ge=10,
        description="Grace period before drift is flagged (>= 2 * cadence).",
    )

    # Strategy
    strategy_name: str = Field(
        default="comprehensive",
        description="Strategy to use: 'comprehensive', 'ma_crossover', or 'momentum'.",
    )

    # Mode
    dry_run: bool = Field(
        default=True,
        description="If True, use PaperBroker instead of BybitBroker.",
    )
    bybit_testnet: bool = Field(
        default=True,
        description="If True, connect to Bybit testnet (not mainnet).",
    )
    bybit_tld: str = Field(
        default="com",
        pattern=r"^(com|kz|hk|eu|nl)$",
        description="Bybit top-level domain suffix (com, kz, hk, eu, nl).",
    )

    @model_validator(mode="after")
    def _validate_grace_period(self) -> LiveConfig:
        """Ensure grace period is at least 2x cadence."""
        min_grace = 2 * self.cadence_seconds
        if self.reconcile_grace_seconds < min_grace:
            msg = (
                f"reconcile_grace_seconds ({self.reconcile_grace_seconds}) "
                f"must be >= 2 * cadence_seconds ({min_grace})"
            )
            raise ValueError(msg)
        return self

    @classmethod
    def from_settings(cls, settings: KaironSettings) -> LiveConfig:
        """Construct LiveConfig from the global KaironSettings."""
        return cls(
            symbols=settings.live_symbols,
            timeframe=settings.live_timeframe,
            cadence_seconds=settings.live_cadence_seconds,
            max_daily_loss_pct=settings.live_max_daily_loss_pct,
            max_open_positions=settings.live_max_open_positions,
            warmup_bars=settings.live_warmup_bars,
            model_path=settings.live_model_path,
            horizon=settings.live_horizon,
            reconcile_interval_seconds=settings.live_reconcile_interval_seconds,
            reconcile_grace_seconds=settings.live_reconcile_grace_seconds,
            dry_run=settings.live_dry_run,
            bybit_testnet=settings.bybit_testnet,
            bybit_tld=settings.bybit_tld,
            strategy_name=getattr(settings, 'live_strategy', 'comprehensive'),
        )


__all__ = ["LiveConfig"]
