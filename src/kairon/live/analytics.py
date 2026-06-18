"""Post-session analytics: compute live trading performance from LiveStore data.

Produces a :class:`LiveSessionReport` that bridges LiveStore's trade/heartbeat
data into the backtest metrics engine (:func:`kairon.backtest.metrics.summarize`)
for Sharpe, Sortino, drawdown, win rate, and profit factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from kairon.backtest.metrics import (
    BARS_PER_YEAR_1H,
    BARS_PER_YEAR_5M,
    max_drawdown,
    profit_factor,
    summarize,
    win_rate,
)
from kairon.live.store import LiveStore


def _parse_ts(ts_str: str) -> datetime:
    """Parse an ISO timestamp string."""
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class SymbolReport:
    """Per-symbol breakdown of trading performance."""

    symbol: str
    n_trades: int
    total_pnl: float
    win_rate: float
    max_drawdown: float
    avg_trade_duration_minutes: float


@dataclass(frozen=True, slots=True)
class LiveSessionReport:
    """Complete analytics for a live trading session."""

    session_id: str
    start_ts: str
    end_ts: str
    duration_minutes: float
    mode: str

    # Equity
    initial_equity: float
    final_equity: float
    total_pnl: float
    total_pnl_pct: float

    # Counts
    n_ticks: int
    n_orders: int
    n_fills: int
    n_trades: int

    # Risk metrics
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float

    # Per-symbol breakdown
    per_symbol: dict[str, SymbolReport] = field(default_factory=dict)

    # Latency
    avg_fill_latency_ms: float = 0.0
    p50_fill_latency_ms: float = 0.0
    p99_fill_latency_ms: float = 0.0

    # Equity curve
    equity_curve: list[tuple[str, float]] = field(default_factory=list)

    # Safety
    guardian_blocks: int = 0
    reconciler_alerts: int = 0

    # Decision journal stats
    n_decisions: int = 0
    n_decisions_with_outcome: int = 0
    decisions_win_rate: float = 0.0  # win rate of closed decisions
    top_strategy: str = ""
    avg_decision_confidence: float = 0.0
    top_justifications: list[tuple[str, int]] = field(default_factory=list)

    # Closed trades (realized PnL)
    closed_trades: list[dict] = field(default_factory=list)


def _bars_per_year(timeframe: str) -> int:
    """Map a timeframe string to annualization factor."""
    mapping = {
        "1m": 365 * 24 * 60,
        "5m": BARS_PER_YEAR_5M,
        "15m": 365 * 24 * 4,
        "1h": BARS_PER_YEAR_1H,
        "4h": 365 * 6,
        "1d": 365,
    }
    return mapping.get(timeframe, BARS_PER_YEAR_1H)


def _compute_trade_pnl(fills: list[dict]) -> dict[str, list[float]]:
    """Compute per-symbol realized PnL using FIFO matching.

    For each symbol, matches BUY and SELL fills in chronological order
    to compute per-trade PnL.
    """
    # Group fills by symbol
    by_symbol: dict[str, list[dict]] = {}
    for f in fills:
        sym = f.get("symbol", "")
        if sym not in by_symbol:
            by_symbol[sym] = []
        by_symbol[sym].append(f)

    result: dict[str, list[float]] = {}
    for sym, sym_fills in by_symbol.items():
        trades: list[float] = []
        # Separate buys and sells, track remaining qty
        buys: list[tuple[float, float]] = []  # (price, remaining_qty)
        pnl = 0.0

        for f in sym_fills:
            side = f.get("side", "")
            qty = float(f.get("qty", 0))
            price = float(f.get("price", 0))
            fee = float(f.get("fee", 0))

            if side.lower() == "buy":
                buys.append((price, qty))
            elif side.lower() == "sell":
                remaining = qty
                while remaining > 1e-12 and buys:
                    buy_price, buy_qty = buys[0]
                    matched = min(remaining, buy_qty)
                    trade_pnl = matched * (price - buy_price) - fee * (matched / qty)
                    trades.append(trade_pnl)
                    buys[0] = (buy_price, buy_qty - matched)
                    remaining -= matched
                    if buys[0][1] < 1e-12:
                        buys.pop(0)
                if remaining > 1e-12:
                    # Short entry: opening a short position
                    buys.append((price, -remaining))
                    pnl -= fee * (remaining / qty)

        result[sym] = trades

    return result


def compute_session_report(
    store: LiveStore,
    *,
    timeframe: str = "1m",
    session_id: str = "",
) -> LiveSessionReport:
    """Compute a full :class:`LiveSessionReport` from a :class:`LiveStore`.

    Parameters
    ----------
    store:
        The live store to pull trade data from.
    timeframe:
        Bar timeframe for annualization (e.g. ``"1m"`` or ``"5m"``).
    session_id:
        Optional session identifier. Defaults to empty string.
    """
    # Fetch all data
    fills = store.get_all_fills()
    orders = store.get_all_orders()
    heartbeats = store.get_all_heartbeats()
    events = store.get_all_events()

    # Session boundaries
    if heartbeats:
        start_ts = heartbeats[0]["ts"]
        end_ts = heartbeats[-1]["ts"]
        start_dt = _parse_ts(start_ts)
        end_dt = _parse_ts(end_ts)
        duration_minutes = (end_dt - start_dt).total_seconds() / 60.0
        mode = heartbeats[0].get("mode", "unknown")
    else:
        start_ts = ""
        end_ts = ""
        duration_minutes = 0.0
        mode = "unknown"

    # Equity curve from heartbeats
    equity_values: list[float] = []
    equity_ts: list[str] = []
    for hb in heartbeats:
        eq = hb.get("equity")
        if eq is not None and eq > 0:
            equity_values.append(float(eq))
            equity_ts.append(hb["ts"])

    initial_equity = equity_values[0] if equity_values else 0.0
    final_equity = equity_values[-1] if equity_values else 0.0
    total_pnl = final_equity - initial_equity
    total_pnl_pct = (total_pnl / initial_equity * 100) if initial_equity > 0 else 0.0

    # Compute risk metrics from equity curve
    if len(equity_values) >= 2:
        equity_arr = np.array(equity_values, dtype=float)
        bpyear = _bars_per_year(timeframe)
        trade_pnl = _compute_trade_pnl(fills)
        # Flatten all per-symbol PnL into one array for win_rate/profit_factor
        all_pnl_lists = [v for v in trade_pnl.values() if len(v) > 0]
        all_pnl = np.concatenate([np.array(v) for v in all_pnl_lists]) if all_pnl_lists else None
        try:
            perf = summarize(equity_arr, bars_per_year=bpyear, trade_pnl=all_pnl)
            sharpe = perf.sharpe
            sortino = perf.sortino
            max_dd = perf.max_drawdown
            calmar = perf.calmar
            wr = perf.win_rate
            pf = perf.profit_factor
        except (OverflowError, ZeroDivisionError):
            # Short sessions can cause overflow in annualization;
            # fall back to simple metrics
            rets = np.diff(equity_arr) / equity_arr[:-1]
            sharpe = float(np.mean(rets) / np.std(rets, ddof=0)) if np.std(rets, ddof=0) > 0 else 0.0
            sortino = float("nan")
            max_dd = float(max_drawdown(equity_arr))
            calmar = float("nan")
            wr = float(win_rate(all_pnl)) if all_pnl is not None else float("nan")
            pf = float(profit_factor(all_pnl)) if all_pnl is not None else float("nan")
    else:
        sharpe = sortino = max_dd = calmar = wr = pf = float("nan")

    # Per-symbol breakdown using closed trades (realized PnL)
    per_symbol: dict[str, SymbolReport] = {}
    closed_trades = store.get_closed_trades()

    # Build per-symbol trade lists from closed_trades
    symbol_trades: dict[str, list[dict]] = {}
    for ct in closed_trades:
        sym = ct.get("symbol", "")
        if sym not in symbol_trades:
            symbol_trades[sym] = []
        symbol_trades[sym].append(ct)

    # Also include fills for symbols that had activity but no closed trades yet
    symbols_seen: dict[str, list[dict]] = {}
    for f in fills:
        sym = f.get("symbol", "")
        if sym not in symbols_seen:
            symbols_seen[sym] = []
        symbols_seen[sym].append(f)

    # Merge: closed trades for PnL/win rate, fills for supplementary info
    all_symbols = set(list(symbol_trades.keys()) + list(symbols_seen.keys()))

    for sym in all_symbols:
        sym_closed = symbol_trades.get(sym, [])
        sym_fills = symbols_seen.get(sym, [])

        # Realized PnL from closed trades (accurate)
        total_realized_pnl = sum(float(ct.get("realized_pnl", 0)) for ct in sym_closed)

        # Win rate from closed trades
        if sym_closed:
            wins = sum(1 for ct in sym_closed if float(ct.get("realized_pnl", 0)) > 0)
            sym_wr = wins / len(sym_closed)
            n_trades = len(sym_closed)

            # Average trade duration
            durations = [float(ct.get("duration_seconds", 0) or 0) for ct in sym_closed]
            avg_dur = (sum(durations) / len(durations) / 60.0) if durations else 0.0

            # Per-symbol max drawdown from realized PnL cumulative curve
            cum_pnl = np.cumsum([float(ct.get("realized_pnl", 0)) for ct in sym_closed])
            if len(cum_pnl) > 0:
                running_max = np.maximum.accumulate(cum_pnl)
                drawdowns = cum_pnl - running_max
                sym_mdd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            else:
                sym_mdd = 0.0
        else:
            # Fallback for symbols with only fills (no closed trades yet)
            n_trades = 0
            sym_wr = 0.0
            avg_dur = 0.0
            sym_mdd = 0.0

        # Also add fill-based notional PnL for supplementary context
        sym_pnl = sum(
            float(f.get("qty", 0)) * float(f.get("price", 0))
            * (1 if f.get("side", "").lower() == "sell" else -1)
            for f in sym_fills
        ) if sym_fills else 0.0

        per_symbol[sym] = SymbolReport(
            symbol=sym,
            n_trades=n_trades,
            total_pnl=total_realized_pnl if sym_closed else sym_pnl,
            win_rate=sym_wr,
            max_drawdown=sym_mdd,
            avg_trade_duration_minutes=avg_dur,
        )

    # Fill latency (order ts vs fill ts)
    order_ts_map = {o["id"]: o["ts"] for o in orders}
    latencies: list[float] = []
    for f in fills:
        ots = order_ts_map.get(f.get("order_id", ""))
        if ots and f.get("ts"):
            try:
                order_dt = _parse_ts(ots)
                fill_dt = _parse_ts(f["ts"])
                latency = (fill_dt - order_dt).total_seconds() * 1000  # ms
                if latency >= 0:
                    latencies.append(latency)
            except (ValueError, TypeError):
                pass

    if latencies:
        lat_arr = np.array(latencies)
        avg_lat = float(np.mean(lat_arr))
        p50_lat = float(np.percentile(lat_arr, 50))
        p99_lat = float(np.percentile(lat_arr, 99))
    else:
        avg_lat = p50_lat = p99_lat = 0.0

    # Safety metrics
    guardian_blocks = sum(1 for e in events if e.get("kind") == "guardian_block")
    reconciler_alerts = sum(1 for e in events if e.get("kind") == "reconciler_alert")

    # Decision journal stats
    try:
        decisions = store.get_decisions(limit=10000)
        n_decisions = len(decisions)
        n_decisions_with_outcome = sum(1 for d in decisions if d.outcome is not None)
        decisions_win_rate = (
            sum(1 for d in decisions if d.outcome is not None and (d.outcome_pnl or 0) > 0)
            / n_decisions_with_outcome
            if n_decisions_with_outcome > 0
            else float("nan")
        )
        # Most common strategy
        strategy_counts: dict[str, int] = {}
        for d in decisions:
            strategy_counts[d.strategy_name] = strategy_counts.get(d.strategy_name, 0) + 1
        top_strategy = max(strategy_counts, key=strategy_counts.get) if strategy_counts else ""
        # Average confidence
        avg_confidence = (
            sum(d.confidence for d in decisions) / n_decisions if n_decisions > 0 else float("nan")
        )
        # Top justifications
        just_counts: dict[str, int] = {}
        for d in decisions:
            for j in d.justifications:
                just_counts[j] = just_counts.get(j, 0) + 1
        top_justifications = sorted(just_counts.items(), key=lambda x: -x[1])[:10]
    except Exception:
        n_decisions = 0
        n_decisions_with_outcome = 0
        decisions_win_rate = float("nan")
        top_strategy = ""
        avg_confidence = float("nan")
        top_justifications = []

    # Count completed round-trip trades from closed trades
    n_trades = len(closed_trades)

    equity_curve = list(zip(equity_ts, equity_values, strict=False)) if equity_ts else []

    return LiveSessionReport(
        session_id=session_id,
        start_ts=start_ts,
        end_ts=end_ts,
        duration_minutes=round(duration_minutes, 2),
        mode=mode,
        initial_equity=round(initial_equity, 2),
        final_equity=round(final_equity, 2),
        total_pnl=round(total_pnl, 2),
        total_pnl_pct=round(total_pnl_pct, 2),
        n_ticks=len(heartbeats),
        n_orders=len(orders),
        n_fills=len(fills),
        n_trades=n_trades,
        sharpe=round(sharpe, 4) if not math.isnan(sharpe) else sharpe,
        sortino=round(sortino, 4) if not math.isnan(sortino) else sortino,
        max_drawdown=round(max_dd, 4) if not math.isnan(max_dd) else max_dd,
        calmar=round(calmar, 4) if not math.isnan(calmar) else calmar,
        win_rate=round(wr, 4) if not math.isnan(wr) else wr,
        profit_factor=round(pf, 4) if not math.isnan(pf) and pf != float("inf") else pf,
        per_symbol=per_symbol,
        avg_fill_latency_ms=round(avg_lat, 2),
        p50_fill_latency_ms=round(p50_lat, 2),
        p99_fill_latency_ms=round(p99_lat, 2),
        equity_curve=equity_curve,
        guardian_blocks=guardian_blocks,
        reconciler_alerts=reconciler_alerts,
        closed_trades=closed_trades,
        n_decisions=n_decisions,
        n_decisions_with_outcome=n_decisions_with_outcome,
        decisions_win_rate=round(decisions_win_rate, 4) if not math.isnan(decisions_win_rate) else decisions_win_rate,
        top_strategy=top_strategy,
        avg_decision_confidence=round(avg_confidence, 4) if not math.isnan(avg_confidence) else avg_confidence,
        top_justifications=top_justifications,
    )


def format_report(report: LiveSessionReport) -> str:
    """Format a :class:`LiveSessionReport` as a human-readable string."""
    lines = [
        "=" * 60,
        f"  LIVE SESSION REPORT  —  {report.session_id or 'N/A'}",
        "=" * 60,
        "",
        f"  Mode:            {report.mode}",
        f"  Start:           {report.start_ts or 'N/A'}",
        f"  End:             {report.end_ts or 'N/A'}",
        f"  Duration:        {report.duration_minutes:.1f} min",
        "",
        "  EQUITY",
        f"  Initial equity:  ${report.initial_equity:,.2f}",
        f"  Final equity:    ${report.final_equity:,.2f}",
        f"  Total PnL:       ${report.total_pnl:,.2f} ({report.total_pnl_pct:+.2f}%)",
        "",
        "  COUNTS",
        f"  Ticks:           {report.n_ticks}",
        f"  Orders:          {report.n_orders}",
        f"  Fills:           {report.n_fills}",
        f"  Round-trip trades: {report.n_trades}",
        "",
        "  RISK METRICS",
        f"  Sharpe ratio:    {report.sharpe:.4f}" if not math.isnan(report.sharpe) else "  Sharpe ratio:    N/A",
        f"  Sortino ratio:   {report.sortino:.4f}" if not math.isnan(report.sortino) else "  Sortino ratio:   N/A",
        f"  Max drawdown:    {report.max_drawdown:.4f}" if not math.isnan(report.max_drawdown) else "  Max drawdown:    N/A",
        f"  Calmar ratio:    {report.calmar:.4f}" if not math.isnan(report.calmar) else "  Calmar ratio:    N/A",
        f"  Win rate:        {report.win_rate:.2%}" if not math.isnan(report.win_rate) else "  Win rate:        N/A",
        f"  Profit factor:   {report.profit_factor:.4f}" if not math.isnan(report.profit_factor) and report.profit_factor != float("inf") else "  Profit factor:   N/A",
        "",
        "  LATENCY",
        f"  Avg fill:        {report.avg_fill_latency_ms:.1f} ms",
        f"  P50 fill:        {report.p50_fill_latency_ms:.1f} ms",
        f"  P99 fill:        {report.p99_fill_latency_ms:.1f} ms",
        "",
        "  SAFETY",
        f"  Guardian blocks:  {report.guardian_blocks}",
        f"  Reconciler alerts: {report.reconciler_alerts}",
    ]

    # Decision journal stats
    if report.n_decisions > 0:
        lines.append("")
        lines.append("  DECISION JOURNAL")
        lines.append("  " + "-" * 56)
        lines.append(f"  Total decisions:  {report.n_decisions}")
        lines.append(f"  Closed outcomes:  {report.n_decisions_with_outcome}")
        if not math.isnan(report.decisions_win_rate):
            lines.append(f"  Decision win rate: {report.decisions_win_rate:.2%}")
        if report.top_strategy:
            lines.append(f"  Top strategy:      {report.top_strategy}")
        if not math.isnan(report.avg_decision_confidence):
            lines.append(f"  Avg confidence:    {report.avg_decision_confidence:.4f}")
        if report.top_justifications:
            lines.append("  Top justifications:")
            for just, count in report.top_justifications[:5]:
                lines.append(f"    {count:3d}x  {just}")

    if report.per_symbol:
        lines.append("")
        lines.append("  PER-SYMBOL BREAKDOWN")
        lines.append("  " + "-" * 56)
        for sym, sr in report.per_symbol.items():
            lines.append(f"  {sym}:")
            lines.append(f"    Trades: {sr.n_trades}  |  Realized PnL: ${sr.total_pnl:,.2f}  |  Win rate: {sr.win_rate:.0%}")
            if sr.max_drawdown != 0.0:
                lines.append(f"    Max drawdown: {sr.max_drawdown:.4f}  |  Avg duration: {sr.avg_trade_duration_minutes:.1f} min")

    if report.closed_trades:
        lines.append("")
        lines.append("  CLOSED TRADES")
        lines.append("  " + "-" * 56)
        for ct in report.closed_trades:
            side = ct.get("side", "?")
            sym = ct.get("symbol", "?")
            qty = ct.get("entry_qty", 0)
            entry = ct.get("entry_price", 0)
            exit_p = ct.get("exit_price", 0)
            pnl = ct.get("realized_pnl", 0)
            dur = ct.get("duration_seconds", 0)
            dur_min = dur / 60 if dur else 0
            pnl_str = f"${pnl:+.2f}" if pnl else "$0.00"
            lines.append(f"  {sym:20s} {side:4s} qty={qty:.6f} entry=${entry:,.2f} exit=${exit_p:,.2f} PnL={pnl_str} dur={dur_min:.1f}m")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


__all__ = ["LiveSessionReport", "SymbolReport", "compute_session_report", "format_report"]
