"""Trade CLI: start, stop, status, flatten, and analyze the live trading loop.

Usage::

    kairon trade start --dry-run      # Paper trading (default)
    kairon trade start --testnet      # Bybit testnet with real API
    kairon trade start --live         # Bybit mainnet (requires KAIRON_LIVE_PROMOTION_ACK=1)
    kairon trade stop                 # Halt the trading loop
    kairon trade status               # Print current status
    kairon trade flatten              # Flatten all open positions and halt
    kairon trade analyze              # Post-session deep analytics

The ``--live`` flag is refused unless ``KAIRON_LIVE_PROMOTION_ACK=1`` is
set in the environment. This is a manual safety gate — the user must
explicitly acknowledge the promotion checklist before trading with real money.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from kairon.config import KaironSettings
    from kairon.data.symbols import CryptoVenue

trade_app = typer.Typer(
    name="trade",
    help="Live trading: start, stop, status, flatten, and analyze.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _load_store(db_path: Path | None = None) -> LiveStore:
    """Load the LiveStore from the default or given database path."""
    from kairon.live.store import LiveStore  # noqa: PLC0415

    if db_path is None:
        db_path = Path("data/runs.db")
    return LiveStore(db_path)


def _print_status(store: LiveStore) -> None:
    """Print a human-readable status summary."""
    halted = store.is_halted()
    halt_reason = store.get_runtime_state("halted") if halted else None
    positions = store.get_positions()
    heartbeat = store.get_recent_heartbeat()

    mode = heartbeat["mode"] if heartbeat else "unknown"
    equity = heartbeat["equity"] if heartbeat else 0.0
    n_positions = heartbeat["n_positions"] if heartbeat else len(positions)
    last_ts = heartbeat["ts"] if heartbeat else "never"

    status_label = "HALTED" if halted else "Running"
    halt_info = f" (reason: {halt_reason})" if halt_reason else ""

    typer.echo(f"Status:    {status_label}{halt_info}")
    typer.echo(f"Mode:      {mode}")
    typer.echo(f"Equity:    ${equity:,.2f}")
    typer.echo(f"Positions: {n_positions}")
    typer.echo(f"Last tick: {last_ts}")

    if positions:
        typer.echo("\nOpen positions:")
        for p in positions:
            pnl_str = f"{p.unrealized_pnl:+.2f}" if p.unrealized_pnl else "—"
            typer.echo(
                f"  {p.symbol:20s} {p.side.value:4s} "
                f"qty={p.qty:.6f} avg={p.avg_entry:,.2f} PnL={pnl_str}"
            )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@trade_app.command("start")  # pyright: ignore[reportUntypedFunctionDecorator]
def start(
    dry_run: bool = typer.Option(False, "--dry-run", help="Paper trading (no real orders)."),
    testnet: bool = typer.Option(False, "--testnet", help="Bybit testnet with real API calls."),
    live: bool = typer.Option(False, "--live", help="Bybit mainnet (requires promotion ack)."),
    strategy: str = typer.Option(
        "comprehensive",
        "--strategy",
        help="Trading strategy: comprehensive, ma_crossover, or momentum.",
    ),
    db: str | None = typer.Option(None, "--db", help="Path to runs.db."),
) -> None:
    """Start the live trading loop.

    Exactly one mode flag must be provided: --dry-run, --testnet, or --live.
    The --live flag requires KAIRON_LIVE_PROMOTION_ACK=1 in the environment.
    """
    flags = sum([dry_run, testnet, live])
    if flags == 0:
        typer.echo("No mode specified. Defaulting to --dry-run (paper trading).")
        dry_run = True
    elif flags > 1:
        typer.echo("Error: specify exactly one of --dry-run, --testnet, or --live.", err=True)
        raise typer.Exit(code=1)

    if live:
        ack = os.environ.get("KAIRON_LIVE_PROMOTION_ACK", "0")
        if ack != "1":
            typer.echo(
                "Error: --live requires KAIRON_LIVE_PROMOTION_ACK=1 in the environment.\n"
                "This is a safety gate. Please review the promotion checklist in\n"
                "docs/live-promotion.md and set the environment variable to\n"
                "acknowledge you have reviewed and approved live trading.",
                err=True,
            )
            raise typer.Exit(code=1)

    from kairon.config import KaironSettings  # noqa: PLC0415

    settings = KaironSettings()

    if dry_run:
        mode_str = "dry-run (paper trading)"
    elif testnet:
        mode_str = "testnet (Bybit testnet)"
    else:
        mode_str = "LIVE (Bybit mainnet)"

    typer.echo(f"Starting trading loop in {mode_str} mode")
    typer.echo(f"  Symbols:  {settings.live_symbols}")
    typer.echo(f"  Timeframe: {settings.live_timeframe}")
    typer.echo(f"  Cadence:   {settings.live_cadence_seconds}s")

    if dry_run:
        typer.echo("\n  [!] Dry-run mode: no real orders will be placed.")
    elif testnet:
        typer.echo("\n  [!] Testnet mode: orders go to Bybit testnet with test funds.")
    else:
        typer.echo("\n  [!!] LIVE mode: real money at risk. Kill switch is active.")

    # Import and run the trading loop
    typer.echo("\nStarting loop... (press Ctrl+C to stop)")
    _run_trading_loop(
        settings,
        dry_run=dry_run,
        testnet=testnet,
        live=live,
        db_path=Path(db) if db else None,
        strategy_name=strategy,
    )


@trade_app.command("stop")  # pyright: ignore[reportUntypedFunctionDecorator]
def stop(
    reason: str = typer.Option("manual_stop", "--reason", help="Reason for halting."),
    db: str | None = typer.Option(None, "--db", help="Path to runs.db."),
) -> None:
    """Halt the trading loop (kill switch).

    This sets the halt flag in the LiveStore, which the TradingLoop
    checks at the top of every tick. The loop will stop placing new orders
    on the next tick.
    """
    store = _load_store(Path(db) if db else None)
    store.halt(reason=reason)
    store.write_event(kind="halt", severity="critical", payload_json=json.dumps({"reason": reason}))
    typer.echo(f"Trading halted. Reason: {reason}")
    store.close()


@trade_app.command("status")  # pyright: ignore[reportUntypedFunctionDecorator]
def status(
    db: str | None = typer.Option(None, "--db", help="Path to runs.db."),
) -> None:
    """Print current trading status."""
    store = _load_store(Path(db) if db else None)
    _print_status(store)
    store.close()


@trade_app.command("flatten")  # pyright: ignore[reportUntypedFunctionDecorator]
def flatten(
    db: str | None = typer.Option(None, "--db", help="Path to runs.db."),
) -> None:
    """Halt the loop and flatten all open positions.

    Sets the halt flag and records a 'flatten' event. The actual
    flattening (closing all positions) is handled by the TradingLoop
    on the next tick.
    """
    store = _load_store(Path(db) if db else None)
    store.halt(reason="flatten_requested")
    store.write_event(
        kind="flatten",
        severity="critical",
        payload_json=json.dumps({"action": "flatten_all", "ts": datetime.now(UTC).isoformat()}),
    )
    typer.echo("Flatten requested. The trading loop will close all positions on the next tick.")
    _print_status(store)
    store.close()


@trade_app.command("analyze")  # pyright: ignore[reportUntypedFunctionDecorator]
def analyze(
    db: str | None = typer.Option(None, "--db", help="Path to runs.db."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
    timeframe: str = typer.Option("1m", "--timeframe", help="Bar timeframe for annualization."),
) -> None:
    """Post-session deep analytics.

    Computes PnL, Sharpe, Sortino, max drawdown, win rate, profit factor,
    per-symbol breakdown, equity curve, and fill latency from the LiveStore.
    """
    from kairon.live.analytics import (  # noqa: PLC0415
        compute_session_report,
        format_report,
    )

    db_path = Path(db) if db else Path("data/runs.db")
    if not db_path.exists():
        typer.echo(f"Error: database not found at {db_path}", err=True)
        raise typer.Exit(code=1)

    store = LiveStore(db_path)
    try:
        report = compute_session_report(store, timeframe=timeframe)
        if json_output:
            # Convert to dict for JSON serialization
            data = {
                "session_id": report.session_id,
                "start_ts": report.start_ts,
                "end_ts": report.end_ts,
                "duration_minutes": report.duration_minutes,
                "mode": report.mode,
                "initial_equity": report.initial_equity,
                "final_equity": report.final_equity,
                "total_pnl": report.total_pnl,
                "total_pnl_pct": report.total_pnl_pct,
                "n_ticks": report.n_ticks,
                "n_orders": report.n_orders,
                "n_fills": report.n_fills,
                "n_trades": report.n_trades,
                "sharpe": report.sharpe,
                "sortino": report.sortino,
                "max_drawdown": report.max_drawdown,
                "calmar": report.calmar,
                "win_rate": report.win_rate,
                "profit_factor": report.profit_factor,
                "per_symbol": {
                    sym: {
                        "symbol": sr.symbol,
                        "n_trades": sr.n_trades,
                        "total_pnl": sr.total_pnl,
                        "win_rate": sr.win_rate,
                        "max_drawdown": sr.max_drawdown,
                        "avg_trade_duration_minutes": sr.avg_trade_duration_minutes,
                    }
                    for sym, sr in report.per_symbol.items()
                },
                "avg_fill_latency_ms": report.avg_fill_latency_ms,
                "p50_fill_latency_ms": report.p50_fill_latency_ms,
                "p99_fill_latency_ms": report.p99_fill_latency_ms,
                "equity_curve": report.equity_curve,
                "guardian_blocks": report.guardian_blocks,
                "reconciler_alerts": report.reconciler_alerts,
                "closed_trades": report.closed_trades,
            }
            typer.echo(json.dumps(data, indent=2, default=str))
        else:
            typer.echo(format_report(report))
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_symbols(symbol_strs: tuple[str, ...], venue: CryptoVenue) -> list:
    """Parse canonical symbol strings into Symbol objects."""
    from kairon.data.symbols import crypto_perp  # noqa: PLC0415

    symbols = []
    for s in symbol_strs:
        parts = s.split("-")
        if len(parts) == 3 and parts[2] == "PERP":
            symbols.append(crypto_perp(parts[0], parts[1], venue))
        elif len(parts) == 2:
            # Spot crypto — not commonly used, but handle it
            from kairon.data.symbols import AssetClass, Symbol  # noqa: PLC0415

            symbols.append(Symbol(
                canonical=s,
                asset_class=AssetClass.CRYPTO_SPOT,
                venue=venue,
                base=parts[0],
                quote=parts[1],
            ))
        else:
            typer.echo(f"Warning: cannot parse symbol {s!r}, skipping", err=True)
    return symbols


def _run_trading_loop(  # noqa: PLR0915
    settings: KaironSettings,
    *,
    dry_run: bool,
    testnet: bool,
    live: bool,
    db_path: Path | None,
    strategy_name: str = "comprehensive",
) -> None:
    """Construct and run the TradingLoop with the appropriate broker and feed."""
    import asyncio  # noqa: PLC0415
    import logging  # noqa: PLC0415

    from kairon.data.symbols import CryptoVenue  # noqa: PLC0415
    from kairon.live.broker import (  # noqa: PLC0415
        BybitBroker,
        BybitRawBroker,
        PaperBroker,
    )
    from kairon.live.config import LiveConfig  # noqa: PLC0415
    from kairon.live.feed import CcxtCandleFeed, CcxtCandleFeedConfig  # noqa: PLC0415
    from kairon.live.guardian import Guardian  # noqa: PLC0415
    from kairon.live.orchestrator import TradingLoop  # noqa: PLC0415
    from kairon.live.reconciler import Reconciler  # noqa: PLC0415
    from kairon.live.store import LiveStore  # noqa: PLC0415
    from kairon.live.strategy import (  # noqa: PLC0415
        ComprehensiveStrategy,
        MACrossoverStrategy,
        MomentumStrategy,
    )

    # Configure logging
    log_level = os.environ.get("KAIRON_LOG_LEVEL", "INFO")
    log_format = os.environ.get("KAIRON_LOG_FORMAT", "text")
    json_format = (
        '{"time":"%(asctime)s","level":"%(levelname)s",'
        '"name":"%(name)s","msg":"%(message)s"}'
    )
    text_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=log_level.upper(),
        format=json_format if log_format == "json" else text_format,
    )

    logger = logging.getLogger("kairon.trade")

    # Set up store
    if db_path is None:
        db_path = Path("data/runs.db")
    store = LiveStore(db_path)

    # Set up LiveConfig from settings
    live_config = LiveConfig.from_settings(settings)

    # Determine venue
    venue = CryptoVenue(settings.live_venue) if dry_run else CryptoVenue.BYBIT

    # Set up broker
    if dry_run:
        broker = PaperBroker(initial_balance=10_000.0)
        logger.info("Using PaperBroker (dry-run mode)")
    else:
        mode = "testnet" if testnet else "live"
        if settings.bybit_broker == "raw":
            broker = BybitRawBroker(
                api_key=settings.bybit_api_key,
                api_secret=settings.bybit_api_secret,
                testnet=testnet or not live,
                tld=settings.bybit_tld,
            )
            logger.info("Using BybitRawBroker (%s mode)", mode)
        else:
            broker = BybitBroker(
                api_key=settings.bybit_api_key,
                api_secret=settings.bybit_api_secret,
                testnet=testnet or not live,
                tld=settings.bybit_tld,
            )
            logger.info("Using BybitBroker (%s mode)", mode)


    # Guardian with store for halt capability
    guardian = Guardian(
        max_position_equity_fraction=settings.max_position_equity_fraction,
        max_open_positions=settings.live_max_open_positions,
        max_daily_loss_pct=settings.live_max_daily_loss_pct,
        store=store,
    )

    # Reconciler
    reconciler = Reconciler(
        drift_tolerance_pct=0.05,
        grace_seconds=live_config.reconcile_grace_seconds,
        reconcile_interval_seconds=live_config.reconcile_interval_seconds,
        store=store,
        broker=broker,
    )

    # Parse symbols into Symbol objects
    symbols = _parse_symbols(settings.live_symbols, venue)
    if not symbols:
        typer.echo("Error: no valid symbols configured.", err=True)
        raise typer.Exit(code=1)

    # Set up candle feed
    feed_config = CcxtCandleFeedConfig(
        venue=venue,
        timeframe=live_config.timeframe,
        testnet=testnet,
        api_key=settings.bybit_api_key if not dry_run else "",
        api_secret=settings.bybit_api_secret if not dry_run else "",
    )
    feed = CcxtCandleFeed(symbols, config=feed_config)

    # Set up strategy based on CLI flag
    strategy_map = {
        "comprehensive": ComprehensiveStrategy,
        "ma_crossover": MACrossoverStrategy,
        "momentum": MomentumStrategy,
    }
    strategy_cls = strategy_map.get(strategy_name)
    if strategy_cls is None:
        typer.echo(
            f"Error: unknown strategy '{strategy_name}'. "
            f"Choose from: {', '.join(strategy_map.keys())}",
            err=True,
        )
        raise typer.Exit(code=1)
    strategy = strategy_cls()
    logger.info(
        "Using %s (warmup=%d)",
        type(strategy).__name__, strategy.warmup_bars,
    )

    # Build the trading loop
    loop = TradingLoop(
        config=live_config,
        broker=broker,
        strategy=strategy,
        guardian=guardian,
        reconciler=reconciler,
        store=store,
        feed=feed,
    )

    logger.info("Trading loop initialized. Press Ctrl+C to stop.")

    async def _main() -> None:
        """Run the feed and trading loop together."""
        feed_task = asyncio.create_task(feed.run(), name="candle-feed")
        loop_task = asyncio.create_task(loop.start(), name="trading-loop")

        logger.info("Candle feed and trading loop started.")

        # Wait for either task to complete (or fail)
        done, pending = await asyncio.wait(
            [feed_task, loop_task],
            return_when=asyncio.FIRST_EXCEPTION,
        )

        # If one task raised, cancel the other
        for task in done:
            if task.exception() is not None:
                logger.error("Task %s failed: %s", task.get_name(), task.exception())

        # Clean up
        await loop.stop()
        await feed.aclose()
        if not isinstance(broker, PaperBroker):
            await broker.aclose()
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.debug("Cancelled task during cleanup")

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Stopping...")
    finally:
        # Ensure cleanup happens
        try:
            asyncio.get_event_loop().run_until_complete(feed.aclose())
        except Exception as exc:
            logger.debug("Feed cleanup ignored: %s", exc)
        try:
            if not isinstance(broker, PaperBroker):
                broker.close()
        except Exception as exc:
            logger.debug("Broker cleanup ignored: %s", exc)
        store.close()
        logger.info("Trading session ended.")


# Type alias for annotation
from kairon.live.store import LiveStore  # noqa: E402 needed for type hints

__all__ = ["trade_app"]
