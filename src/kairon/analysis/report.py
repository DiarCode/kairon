"""Markdown report generation for Kairon analysis results."""

from __future__ import annotations

from datetime import UTC, datetime

from kairon.analysis.engine import AnalysisResult, ModelPrediction
from kairon.analysis.risk import RiskLevels
from kairon.analysis.signals import SweetSpot


def _format_price(price: float) -> str:
    """Format a price with appropriate decimal places."""
    if abs(price) >= 1000:
        return f"${price:,.2f}"
    elif abs(price) >= 1:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"


def _format_pct(value: float) -> str:
    """Format a probability/percentage value."""
    return f"{value:.1%}"


def _direction_emoji(direction: str) -> str:
    """Return emoji for sweet spot direction."""
    return "🟢" if direction == "BUY" else "🔴"


def _regime_bar(prob: float, label: str) -> str:
    """Create a simple text progress bar for regime probability."""
    filled = int(prob * 20)
    bar = "█" * filled + "░" * (20 - filled)
    return f"{label}: |{bar}| {_format_pct(prob)}"


def _generate_sweet_spots_section(spots: tuple[SweetSpot, ...]) -> str:
    """Generate the sweet spots section of the report."""
    if not spots:
        return "## Sweet Spots\n\nNo sweet spots detected with current threshold.\n"

    lines = [
        "## Sweet Spots\n",
        f"**{len(spots)} sweet spot(s) detected:**\n",
    ]

    for i, spot in enumerate(spots, 1):
        lines.append(
            f"### {i}. {_direction_emoji(spot.direction)} {spot.direction} @ {_format_price(spot.price)}"
        )
        lines.append("")
        lines.append(f"| Attribute | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| **Bar Index** | {spot.bar_index} |")
        lines.append(f"| **Timestamp** | {spot.timestamp} |")
        lines.append(f"| **Timing Horizon** | {spot.timing_horizon} |")
        lines.append(f"| **Model Confidence** | {_format_pct(spot.model_confidence)} |")
        lines.append(f"| **Combined Score** | {_format_pct(spot.combined_score)} |")
        lines.append(f"| **Direction** | {spot.direction} |")
        lines.append("")

        if spot.justifications:
            lines.append("**Justifications:**")
            for j in spot.justifications:
                lines.append(f"- {j}")
            lines.append("")

        if spot.corroboration:
            lines.append("**Corroboration:**")
            for name, adj in spot.corroboration:
                lines.append(f"- {name}: +{adj:.2f}")
            lines.append("")

    return "\n".join(lines)


def _generate_risk_section(risk: RiskLevels, current_price: float) -> str:
    """Generate the risk levels section."""
    lines = [
        "## Risk Levels\n",
        f"**Current Price:** {_format_price(current_price)}  ",
        f"**ATR(14):** {_format_price(risk.atr)}\n",
        "### Stop Loss & Take Profit\n",
        "| Level | Long | Short |",
        "|---|---|---|",
        f"| **Stop Loss (2x ATR)** | {_format_price(risk.stop_loss_long)} | {_format_price(risk.stop_loss_short)} |",
        f"| **Stop Loss Tight (1.5x ATR)** | {_format_price(risk.stop_loss_long_tight)} | {_format_price(risk.stop_loss_short_tight)} |",
        f"| **Take Profit 1 (2x ATR)** | {_format_price(risk.take_profit_long_1)} | {_format_price(risk.take_profit_short_1)} |",
        f"| **Take Profit 2 (3x ATR)** | {_format_price(risk.take_profit_long_2)} | {_format_price(risk.take_profit_short_2)} |",
        f"| **Fib 1.618 Extension** | {_format_price(risk.fib_tp_long)} | {_format_price(risk.fib_tp_short)} |",
        "",
    ]

    # Risk/Reward ratios
    if risk.atr > 0:
        sl_distance = 2 * risk.atr
        tp1_distance = 2 * risk.atr
        tp2_distance = 3 * risk.atr
        lines.append(f"**Risk/Reward Ratios:**")
        lines.append(f"- TP1 (2x ATR): 1:1 R:R")
        lines.append(f"- TP2 (3x ATR): 1:1.5 R:R")
        lines.append(
            f"- Fib Extension: 1:{risk.fib_tp_long / sl_distance:.1f} R:R (long)"
            if sl_distance > 0
            else ""
        )
        lines.append("")

    lines.append(f"### Position Sizing\n")
    lines.append(
        f"- **Recommended Position Size:** {_format_pct(risk.position_size_pct)} of equity"
    )
    lines.append("")

    return "\n".join(lines)


def _generate_model_section(predictions: tuple[ModelPrediction, ...], heuristic: bool) -> str:
    """Generate the model predictions section."""
    if heuristic:
        return (
            "## Model Predictions\n\n"
            "> **Heuristic Mode** — No ML model was trained. "
            "Sweet spots are based on structural features only with a base confidence of 0.50. "
            "Results should be interpreted with caution.\n"
        )

    if not predictions:
        return "## Model Predictions\n\nNo model predictions available.\n"

    lines = [
        "## Model Predictions\n",
        f"**{len(predictions)} model(s) trained on 80/20 walk-forward split:**\n",
    ]

    for pred in predictions:
        lines.append(f"### {pred.model_name.upper()} Model")
        lines.append("")
        lines.append(f"| Attribute | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| **Direction** | {pred.direction.upper()} |")
        lines.append(f"| **Confidence** | {_format_pct(pred.confidence)} |")
        lines.append(f"| **Down Probability** | {_format_pct(pred.proba[0])} |")
        lines.append(f"| **Flat Probability** | {_format_pct(pred.proba[1])} |")
        lines.append(f"| **Up Probability** | {_format_pct(pred.proba[2])} |")
        lines.append(f"| **Magnitude Forecast** | {pred.magnitude:.4f} |")
        lines.append(f"| **Volatility Forecast** | {pred.vol_forecast:.4f} |")
        lines.append("")

    return "\n".join(lines)


def _generate_state_section(result: AnalysisResult) -> str:
    """Generate the current market state section."""
    cs = result.current_state

    lines = [
        "## Current Market State\n",
        f"| Metric | Value |",
        f"|---|---|",
        f"| **Symbol** | {result.symbol} |",
        f"| **Timeframe** | {result.timeframe} |",
        f"| **Timestamp** | {cs.timestamp} |",
        f"| **Close** | {_format_price(cs.close)} |",
        f"| **Has Volume** | {'Yes' if result.has_volume else 'No'} |",
        "",
        "### Elliott Wave",
        f"| Metric | Value |",
        f"|---|---|",
        f"| **Wave Position** | {cs.ew_wave_position} |",
        f"| **Wave Direction** | {'Bullish' if cs.ew_wave_direction > 0.5 else 'Bearish' if cs.ew_wave_direction < -0.5 else 'Neutral'} |",
        f"| **Impulse** | {'Yes' if cs.ew_is_impulse else 'No'} |",
        f"| **Completion Probability** | {_format_pct(cs.ew_completion_prob)} |",
        f"| **Fib Confluence** | {cs.ew_fib_confluence:.3f} |",
        "",
        "### Regime Detection",
        f"| {_regime_bar(cs.regime_prob_trending, 'Trending')} |",
        f"| {_regime_bar(cs.regime_prob_ranging, 'Ranging')} |",
        f"| {_regime_bar(cs.regime_prob_volatile, 'Volatile')} |",
        f"| {_regime_bar(cs.regime_prob_stressed, 'Stressed')} |",
        f"| **Dominant Regime:** {cs.regime.upper()} |",
        "",
        "### Volatility",
        f"| Metric | Value |",
        f"|---|---|",
        f"| **Hurst Exponent** | {cs.hurst_exp:.3f} |",
        f"| **GARCH Volatility** | {cs.garch_vol:.4f} |",
        f"| **ATR(14)** | {_format_price(cs.atr_14)} |",
        f"| **RSI(14)** | {cs.rsi_14:.1f} |",
        "",
        "### Structure",
        f"| Metric | Value |",
        f"|---|---|",
        f"| **Fib 23.6% Distance** | {cs.fib_dist_236:.3f} ATR |",
        f"| **Fib 38.2% Distance** | {cs.fib_dist_382:.3f} ATR |",
        f"| **Fib 50% Distance** | {cs.fib_dist_500:.3f} ATR |",
        f"| **Fib 61.8% Distance** | {cs.fib_dist_618:.3f} ATR |",
        f"| **Fib 78.6% Distance** | {cs.fib_dist_786:.3f} ATR |",
        f"| **Bullish FVG** | {'Yes' if cs.fvg_bullish else 'No'} |",
        f"| **Bearish FVG** | {'Yes' if cs.fvg_bearish else 'No'} |",
        f"| **FVG Fill %** | {_format_pct(cs.fvg_fill_pct)} |",
        f"| **Bullish Order Block Zone** | {'Yes' if cs.ob_in_bullish_zone else 'No'} |",
        f"| **Bearish Order Block Zone** | {'Yes' if cs.ob_in_bearish_zone else 'No'} |",
        f"| **BOS Direction** | {'Bullish' if cs.bos_direction == 1 else 'Bearish' if cs.bos_direction == -1 else 'None'} |",
        f"| **CHoCH Direction** | {'Bullish' if cs.choch_direction == 1 else 'Bearish' if cs.choch_direction == -1 else 'None'} |",
        "",
        "### Moving Averages & Bollinger",
        f"| Metric | Value |",
        f"|---|---|",
        f"| **EMA 50** | {_format_price(cs.ema_50)} |",
        f"| **EMA 200** | {_format_price(cs.ema_200)} |",
        f"| **BB Upper** | {_format_price(cs.bb_upper)} |",
        f"| **BB Mid** | {_format_price(cs.bb_mid)} |",
        f"| **BB Lower** | {_format_price(cs.bb_lower)} |",
        "",
    ]

    return "\n".join(lines)


def _generate_pivots_section(result: AnalysisResult) -> str:
    """Generate Elliott Wave pivots section."""
    pivots = result.pivots
    if not pivots:
        return "## Elliott Wave Pivots\n\nNo zigzag pivots detected.\n"

    lines = [
        "## Elliott Wave Pivots\n",
        f"**{len(pivots)} zigzag pivot(s) detected:**\n",
        "| # | Price | Type |",
        "|---|---|---|",
    ]

    for i, p in enumerate(pivots[-20:], 1):  # Show last 20 pivots
        ptype = p.kind.upper()
        lines.append(f"| {i} | {_format_price(p.price)} | {ptype} |")

    lines.append("")
    return "\n".join(lines)


def _generate_features_section(result: AnalysisResult) -> str:
    """Generate the features extracted section."""
    n_features = len(result.feature_names)
    lines = [
        "## Features Extracted\n",
        f"**{n_features} features** extracted from the Kairon pipeline.\n",
    ]

    if n_features <= 40:
        lines.append("`" + "`, `".join(result.feature_names) + "`\n")
    else:
        # Group by prefix
        groups: dict[str, list[str]] = {}
        for fname in result.feature_names:
            prefix = fname.split(".")[0] if "." in fname else fname.split("_")[0]
            groups.setdefault(prefix, []).append(fname)

        for prefix, features in sorted(groups.items()):
            lines.append(
                f"- **{prefix}** ({len(features)}): `{', '.join(features[:8])}"
                + (f"` ... +{len(features) - 8} more" if len(features) > 8 else "`")
            )

    lines.append("")
    return "\n".join(lines)


def _generate_executive_summary(result: AnalysisResult) -> str:
    """Generate the executive summary."""
    cs = result.current_state
    spots = result.sweet_spots
    risk = result.risk_levels

    # Determine overall bias
    buy_spots = [s for s in spots if s.direction == "BUY"]
    sell_spots = [s for s in spots if s.direction == "SELL"]
    n_buy = len(buy_spots)
    n_sell = len(sell_spots)

    if n_buy > n_sell:
        bias = "BULLISH"
        bias_emoji = "🟢"
    elif n_sell > n_buy:
        bias = "BEARISH"
        bias_emoji = "🔴"
    else:
        bias = "NEUTRAL"
        bias_emoji = "⚪"

    # Best sweet spot
    best_spot = max(spots, key=lambda s: s.combined_score) if spots else None

    lines = [
        "# Kairon Analysis Report\n",
        f"**{result.symbol}** | **{result.timeframe}** | {cs.timestamp.strftime('%Y-%m-%d %H:%M UTC')}\n",
        "---\n",
        f"## Executive Summary\n",
        f"- **Current Price:** {_format_price(cs.close)}",
        f"- **Market Bias:** {bias_emoji} {bias}",
        f"- **Regime:** {cs.regime.upper()}",
        f"- **Sweet Spots:** {n_buy} BUY, {n_sell} SELL",
        f"- **Features Extracted:** {len(result.feature_names)}",
        f"- **Model Predictions:** {len(result.model_predictions)}",
        "",
    ]

    if best_spot:
        lines.append(
            f"**Strongest Signal:** {_direction_emoji(best_spot.direction)} {best_spot.direction} "
            f"@ {_format_price(best_spot.price)} "
            f"(score: {_format_pct(best_spot.combined_score)}, "
            f"horizon: {best_spot.timing_horizon})"
        )
        lines.append("")

    # Quick risk summary
    lines.append(f"**Risk Summary:**")
    lines.append(
        f"- Long SL: {_format_price(risk.stop_loss_long)} | "
        f"Long TP1: {_format_price(risk.take_profit_long_1)} | "
        f"Long TP2: {_format_price(risk.take_profit_long_2)}"
    )
    lines.append(
        f"- Short SL: {_format_price(risk.stop_loss_short)} | "
        f"Short TP1: {_format_price(risk.take_profit_short_1)} | "
        f"Short TP2: {_format_price(risk.take_profit_short_2)}"
    )
    lines.append("")

    # Model predictions summary
    if result.model_predictions:
        for pred in result.model_predictions:
            lines.append(
                f"- **{pred.model_name.upper()}**: {pred.direction.upper()} "
                f"(confidence: {_format_pct(pred.confidence)})"
            )
        lines.append("")

    return "\n".join(lines)


def generate_markdown_report(result: AnalysisResult) -> str:
    """Generate a comprehensive markdown analysis report.

    Parameters
    ----------
    result : AnalysisResult
        Complete analysis result from run_analysis().

    Returns
    -------
    str
        Full markdown report string.
    """
    heuristic = len(result.model_predictions) == 0

    sections = [
        _generate_executive_summary(result),
        _generate_sweet_spots_section(result.sweet_spots),
        _generate_risk_section(result.risk_levels, result.current_state.close),
        _generate_model_section(result.model_predictions, heuristic),
        _generate_state_section(result),
        _generate_pivots_section(result),
        _generate_features_section(result),
        "---\n",
        f"*Generated by Kairon Analysis Engine on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}*\n",
        f"*Data: {result.symbol} {result.timeframe} | {len(result.df)} bars | "
        f"{len(result.feature_names)} features | "
        f"{'Heuristic' if heuristic else 'ML-enhanced'} mode*\n",
    ]

    return "\n".join(sections)
