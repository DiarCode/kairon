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
        description="Strategy to use: 'comprehensive', 'ma_crossover', 'momentum', or 'scalping'.",
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


class BankrollConfig(BaseModel, frozen=True, strict=True):
    """Synthetic bankroll config for the growth session mode.

    When attached to a :class:`~kairon.live.orchestrator.TradingLoop`, the loop
    sizes positions off a *tracked nominal bankroll* (starting at ``start``
    USDT) rather than the live broker equity, compounds the bankroll with each
    closed trade's realized PnL, logs every change to the ``growth_ledger``
    table, and halts cleanly once ``stop_at`` is reached. Leverage is applied to
    the bankroll so the resulting notional clears Bybit's per-symbol minimum
    order quantities from a small starting stake. The real testnet account
    balance still serves as margin, so a synthetic-bankroll drawdown halts
    trading without liquidating the real account.
    """

    model_config = {"extra": "forbid"}

    start: float = Field(default=10.0, gt=0, description="Starting bankroll in USDT.")
    leverage: float = Field(
        default=10.0, gt=0, description="Leverage applied to the bankroll for sizing."
    )
    allocation: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description="Fraction of (bankroll * leverage) deployed per trade.",
    )
    stop_at: float | None = Field(
        default=100.0,
        gt=0,
        description="Bankroll level that halts the loop. None = never halt on profit.",
    )
    milestones: tuple[float, ...] = Field(
        default=(50.0, 100.0),
        description="Bankroll levels logged as milestone events when crossed upward.",
    )
    risk_per_trade: float = Field(
        default=0.025,
        ge=0,
        le=0.2,
        description=(
            "Fraction of the bankroll risked per trade for scalping risk-based "
            "sizing. 0 = fall back to notional sizing. When >0, position size is "
            "qty = (risk_per_trade * bankroll) / sl_distance, capped by the "
            "leverage notional and the broker min quantity."
        ),
    )
    rr_ratio: float = Field(
        default=1.3,
        gt=0,
        description="Reward:risk ratio used by the strategy to set take-profit.",
    )
    max_drawdown: float | None = Field(
        default=0.30,
        ge=0,
        le=1,
        description=(
            "Bankroll peak-to-trough drawdown fraction that halts the loop. "
            "None disables the drawdown halt."
        ),
    )
    enforce_risk_cap: bool = Field(
        default=True,
        description=(
            "When True, the sizer recomputes the implied risk after any qty "
            "rounding/overshoot and skips a trade whose implied risk would "
            "exceed risk_per_trade * (1 + risk_cap_tol). This is the runtime "
            "guarantee that the risk cap stays inviolable even when the broker "
            "floors quantity to the min lot or confidence-scaled sizing "
            "inflates the intended quantity. Default ON (a risk-correctness "
            "fix that ships active)."
        ),
    )
    allow_min_lot_overshoot: bool = Field(
        default=False,
        description=(
            "When True, a risk-sized qty below the broker min lot is bumped UP "
            "to the min lot (trading a larger-than-intended quantity) instead "
            "of being skipped — but only if the resulting implied risk still "
            "respects risk_per_trade * (1 + risk_cap_tol). Default OFF: sub-"
            "min-lot signals skip with a logged 'skip' ledger row, keeping the "
            "risk cap exactly bounded."
        ),
    )
    risk_cap_tol: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description=(
            "Tolerance on the risk cap for min-lot rounding. A trade is "
            "skipped when implied_risk > risk_per_trade * (1 + risk_cap_tol). "
            "0.10 allows a 10% overshoot from lot rounding; set 0 for a hard "
            "cap."
        ),
    )
    risk_per_trade_cap: float = Field(
        default=0.2,
        gt=0,
        le=0.2,
        description=(
            "Runtime hard-clamp target for confidence-scaled risk — the final "
            "authority after any confidence multiplier is applied "
            "(effective_risk = min(risk_per_trade * mult, risk_per_trade_cap)). "
            "Defaults to the same 0.2 ceiling as risk_per_trade's load-time "
            "validator so confidence scaling can never breach the cap."
        ),
    )

    @property
    def sizing_notional_factor(self) -> float:
        """Multiplier applied to the live bankroll to get target notional."""
        return self.leverage * self.allocation


__all__ = ["BankrollConfig", "LiveConfig"]
