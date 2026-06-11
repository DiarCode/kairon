"""Interactive Plotly dashboard builder for Kairon analysis results.

Produces a 4-row layout:
  Row 1 (50%): Price candlestick + all overlays (EW, Fib, BB, EMAs, sweet spots, regime)
  Row 2 (18%): Tabbed — Volume / Sweet Spot Confidence
  Row 3 (16%): Tabbed — RSI / MACD
  Row 4 (16%): Tabbed — EW position / Regime probs / Volatility

All rows share the x-axis for synchronized zoom/pan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kairon.analysis.engine import AnalysisResult
from kairon.analysis.signals import SweetSpot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_price(price: float) -> str:
    """Format price for hover text."""
    if abs(price) >= 1000:
        return f"${price:,.2f}"
    elif abs(price) >= 1:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"


def _ew_label(idx: int, total: int) -> str:
    """Generate an Elliott Wave label like 'W1', 'W2', etc."""
    # Simple mapping: alternate impulse waves
    cycle_pos = idx % 8
    labels = ["W1", "W2", "W3", "W4", "W5", "A", "B", "C"]
    return labels[cycle_pos]


def _regime_color(regime: str) -> str:
    """Map regime name to background color (semi-transparent)."""
    return {
        "trending": "rgba(46, 204, 113, 0.08)",
        "ranging": "rgba(52, 152, 219, 0.08)",
        "volatile": "rgba(241, 196, 15, 0.08)",
        "stressed": "rgba(231, 76, 60, 0.08)",
    }.get(regime, "rgba(149, 165, 166, 0.05)")


def _add_regime_shapes(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    """Add regime background shading to a subplot row."""
    if "regime" not in df.columns:
        return

    regimes = df["regime"].values
    timestamps = df["ts"].values if "ts" in df.columns else df.index

    # Group consecutive bars with the same regime
    i = 0
    while i < len(regimes):
        regime = regimes[i]
        start_idx = i
        while i < len(regimes) and regimes[i] == regime:
            i += 1
        end_idx = i - 1

        start_ts = timestamps[start_idx]
        end_ts = timestamps[min(end_idx, len(timestamps) - 1)]

        fig.add_shape(
            type="rect",
            x0=start_ts, x1=end_ts,
            y0=0, y1=1,
            yref=f"y{row} domain" if row > 1 else "y domain",
            fillcolor=_regime_color(str(regime)),
            line_width=0,
            layer="below",
            row=row, col=1,
        )


def _add_sweet_spot_markers(
    fig: go.Figure,
    spots: tuple[SweetSpot, ...],
    df: pd.DataFrame,
    row: int,
) -> None:
    """Add buy/sell sweet spot markers to the price chart."""
    if not spots:
        return

    buy_spots = [s for s in spots if s.direction == "BUY"]
    sell_spots = [s for s in spots if s.direction == "SELL"]

    for spot_list, color, symbol, name in [
        (buy_spots, "#2ecc71", "triangle-up", "BUY Sweet Spots"),
        (sell_spots, "#e74c3c", "triangle-down", "SELL Sweet Spots"),
    ]:
        if not spot_list:
            continue

        fig.add_trace(
            go.Scatter(
                x=[s.timestamp for s in spot_list],
                y=[s.price for s in spot_list],
                mode="markers",
                name=name,
                marker=dict(
                    symbol=symbol,
                    size=[max(8, s.combined_score * 18) for s in spot_list],
                    color=color,
                    line=dict(width=1, color="white"),
                ),
                hovertext=[
                    f"<b>{s.direction}</b><br>"
                    f"Price: {_fmt_price(s.price)}<br>"
                    f"Confidence: {s.model_confidence:.1%}<br>"
                    f"Score: {s.combined_score:.1%}<br>"
                    f"Horizon: {s.timing_horizon}<br>"
                    f"<b>Justifications:</b><br>"
                    + "<br>".join(f"- {j}" for j in s.justifications)
                    for s in spot_list
                ],
                hoverinfo="text",
                showlegend=True,
            ),
            row=row, col=1,
        )


def _add_elliott_wave(
    fig: go.Figure,
    result: AnalysisResult,
    df: pd.DataFrame,
    row: int,
) -> None:
    """Add Elliott Wave zigzag lines and labels to the price chart."""
    pivots = result.pivots
    if len(pivots) < 2:
        return

    timestamps = df["ts"].values if "ts" in df.columns else None

    # Build zigzag line through pivots
    pivot_indices = []
    for p in pivots:
        # Find closest bar index to this pivot
        if p.idx < len(df):
            pivot_indices.append(p.idx)
        else:
            pivot_indices.append(len(df) - 1)

    # Zigzag line
    pivot_ts = []
    pivot_prices = []
    for idx in pivot_indices:
        if timestamps is not None and idx < len(timestamps):
            pivot_ts.append(timestamps[idx])
        else:
            pivot_ts.append(idx)
        pivot_prices.append(pivots[list(pivot_indices).index(idx) if idx in pivot_indices else 0].price)

    # Re-map pivot_ts from indices to timestamps
    pivot_ts_clean = []
    pivot_prices_clean = []
    for i, p in enumerate(pivots):
        idx = p.idx if p.idx < len(df) else len(df) - 1
        if timestamps is not None and idx < len(timestamps):
            pivot_ts_clean.append(timestamps[idx])
        else:
            pivot_ts_clean.append(idx)
        pivot_prices_clean.append(p.price)

    if len(pivot_ts_clean) >= 2:
        fig.add_trace(
            go.Scatter(
                x=pivot_ts_clean,
                y=pivot_prices_clean,
                mode="lines+markers+text",
                name="Elliott Wave",
                line=dict(color="#9b59b6", width=2, dash="dash"),
                marker=dict(size=8, color="#9b59b6", symbol="diamond"),
                text=[_ew_label(i, len(pivots)) for i in range(len(pivots))],
                textposition="top center" if pivots[0].kind == "high" else "bottom center",
                textfont=dict(size=10, color="#9b59b6"),
                hovertext=[
                    f"Wave: {_ew_label(i, len(pivots))}<br>"
                    f"Price: {_fmt_price(p.price)}<br>"
                    f"Type: {p.kind.upper()}"
                    for i, p in enumerate(pivots)
                ],
                hoverinfo="text",
                showlegend=True,
            ),
            row=row, col=1,
        )


def _add_fibonacci_levels(
    fig: go.Figure,
    result: AnalysisResult,
    row: int,
) -> None:
    """Add Fibonacci retracement levels as horizontal lines."""
    pivots = result.pivots
    if len(pivots) < 2:
        return

    # Use last two pivots for Fibonacci levels
    high_pivot = pivots[-1] if pivots[-1].kind == "high" else pivots[-2]
    low_pivot = pivots[-2] if pivots[-1].kind == "high" else pivots[-1]

    if high_pivot.kind != "high":
        # Swap if needed
        high_pivot, low_pivot = low_pivot, high_pivot

    swing = high_pivot.price - low_pivot.price
    if swing <= 0:
        return

    fib_levels = [
        (0.236, "#e74c3c", "Fib 23.6%"),
        (0.382, "#e67e22", "Fib 38.2%"),
        (0.500, "#f1c40f", "Fib 50.0%"),
        (0.618, "#2ecc71", "Fib 61.8%"),
        (0.786, "#3498db", "Fib 78.6%"),
    ]

    for ratio, color, label in fib_levels:
        level_price = high_pivot.price - ratio * swing
        fig.add_hline(
            y=level_price,
            line_dash="dot",
            line_color=color,
            line_width=1,
            opacity=0.6,
            annotation_text=f"{label} ({_fmt_price(level_price)})",
            annotation_position="right",
            annotation_font_size=9,
            annotation_font_color=color,
            row=row, col=1,
        )


def _add_bollinger_bands(
    fig: go.Figure,
    df: pd.DataFrame,
    row: int,
) -> None:
    """Add Bollinger Bands to the price chart."""
    if "bb_upper" not in df.columns or "bb_lower" not in df.columns:
        return

    timestamps = df["ts"] if "ts" in df.columns else df.index

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=df["bb_upper"],
            mode="lines",
            name="BB Upper",
            line=dict(color="rgba(52, 152, 219, 0.5)", width=1),
            showlegend=True,
            hoverinfo="y",
        ),
        row=row, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=df["bb_lower"],
            mode="lines",
            name="BB Lower",
            line=dict(color="rgba(52, 152, 219, 0.5)", width=1),
            fill="tonexty",
            fillcolor="rgba(52, 152, 219, 0.05)",
            showlegend=True,
            hoverinfo="y",
        ),
        row=row, col=1,
    )


def _add_emas(
    fig: go.Figure,
    df: pd.DataFrame,
    row: int,
) -> None:
    """Add EMA 50 and EMA 200 lines."""
    timestamps = df["ts"] if "ts" in df.columns else df.index

    for ema_col, color, name in [
        ("ema_50", "#f39c12", "EMA 50"),
        ("ema_200", "#e74c3c", "EMA 200"),
    ]:
        if ema_col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df[ema_col],
                mode="lines",
                name=name,
                line=dict(color=color, width=1.5),
                showlegend=True,
                hoverinfo="y",
            ),
            row=row, col=1,
        )


def _add_rsi(
    fig: go.Figure,
    df: pd.DataFrame,
    row: int,
) -> None:
    """Add RSI indicator to subplot."""
    if "rsi_14" not in df.columns:
        return

    timestamps = df["ts"] if "ts" in df.columns else df.index

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=df["rsi_14"],
            mode="lines",
            name="RSI(14)",
            line=dict(color="#8e44ad", width=1.5),
            showlegend=True,
            hoverinfo="y",
        ),
        row=row, col=1,
    )

    # Overbought/oversold lines
    fig.add_hline(
        y=70, line_dash="dash", line_color="#e74c3c", line_width=1,
        annotation_text="Overbought (70)", annotation_position="right",
        row=row, col=1,
    )
    fig.add_hline(
        y=30, line_dash="dash", line_color="#2ecc71", line_width=1,
        annotation_text="Oversold (30)", annotation_position="right",
        row=row, col=1,
    )


def _add_volume(
    fig: go.Figure,
    df: pd.DataFrame,
    row: int,
) -> None:
    """Add volume bars to subplot."""
    if "volume" not in df.columns or not (df["volume"] > 0).any():
        return

    timestamps = df["ts"] if "ts" in df.columns else df.index

    # Color by direction
    colors = np.where(
        df["close"].values >= df["open"].values,
        "#2ecc71",  # green for up
        "#e74c3c",  # red for down
    )

    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=df["volume"],
            name="Volume",
            marker_color=colors,
            showlegend=True,
            hoverinfo="y",
        ),
        row=row, col=1,
    )


def _add_sweet_spot_confidence_bars(
    fig: go.Figure,
    spots: tuple[SweetSpot, ...],
    row: int,
) -> None:
    """Add sweet spot confidence as colored bars."""
    if not spots:
        return

    buy_spots = [s for s in spots if s.direction == "BUY"]
    sell_spots = [s for s in spots if s.direction == "SELL"]

    for spot_list, color, name in [
        (buy_spots, "#2ecc71", "BUY Confidence"),
        (sell_spots, "#e74c3c", "SELL Confidence"),
    ]:
        if not spot_list:
            continue

        fig.add_trace(
            go.Bar(
                x=[s.timestamp for s in spot_list],
                y=[s.combined_score for s in spot_list],
                name=name,
                marker_color=color,
                showlegend=True,
                hovertext=[
                    f"{s.direction}<br>Score: {s.combined_score:.1%}<br>"
                    f"Model Conf: {s.model_confidence:.1%}"
                    for s in spot_list
                ],
                hoverinfo="text",
            ),
            row=row, col=1,
        )


def _add_ew_position(
    fig: go.Figure,
    df: pd.DataFrame,
    row: int,
) -> None:
    """Add Elliott Wave position and completion probability."""
    timestamps = df["ts"] if "ts" in df.columns else df.index

    if "ew_wave_position" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["ew_wave_position"],
                mode="lines",
                name="EW Position",
                line=dict(color="#9b59b6", width=1.5),
                showlegend=True,
                hoverinfo="y",
            ),
            row=row, col=1,
        )

    if "ew_completion_prob" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["ew_completion_prob"],
                mode="lines",
                name="EW Completion Prob",
                line=dict(color="#e67e22", width=1.5, dash="dash"),
                showlegend=True,
                hoverinfo="y",
            ),
            row=row, col=1,
        )


def _add_regime_probs(
    fig: go.Figure,
    df: pd.DataFrame,
    row: int,
) -> None:
    """Add regime probability stacked area chart."""
    timestamps = df["ts"] if "ts" in df.columns else df.index

    regime_cols = {
        "regime_prob_trending": ("#2ecc71", "Trending"),
        "regime_prob_ranging": ("#3498db", "Ranging"),
        "regime_prob_volatile": ("#f1c40f", "Volatile"),
        "regime_prob_stressed": ("#e74c3c", "Stressed"),
    }

    for col, (color, name) in regime_cols.items():
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df[col],
                mode="lines",
                name=name,
                line=dict(color=color, width=1.5),
                stackgroup="regime",
                showlegend=True,
                hoverinfo="y",
            ),
            row=row, col=1,
        )


def _add_volatility(
    fig: go.Figure,
    df: pd.DataFrame,
    row: int,
) -> None:
    """Add GARCH volatility and ATR chart."""
    timestamps = df["ts"] if "ts" in df.columns else df.index

    if "garch_vol" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["garch_vol"],
                mode="lines",
                name="GARCH Vol",
                line=dict(color="#e74c3c", width=1.5),
                showlegend=True,
                hoverinfo="y",
            ),
            row=row, col=1,
        )

    if "atr_14" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["atr_14"],
                mode="lines",
                name="ATR(14)",
                line=dict(color="#3498db", width=1.5, dash="dash"),
                showlegend=True,
                hoverinfo="y",
            ),
            row=row, col=1,
        )


# ---------------------------------------------------------------------------
# Main dashboard builder
# ---------------------------------------------------------------------------

def build_dashboard(
    result: AnalysisResult,
    output_path: Path | str | None = None,
    *,
    static: bool = False,
    dpi: int = 300,
) -> Path:
    """Build an interactive Plotly dashboard from analysis results.

    Parameters
    ----------
    result : AnalysisResult
        Complete analysis result from run_analysis().
    output_path : Path or str or None
        Where to save the dashboard HTML (or PNG if static=True).
        Defaults to ./reports/{symbol}_{timeframe}_dashboard.html
    static : bool
        If True, export as PNG instead of interactive HTML.
    dpi : int
        DPI for static export (default 300).

    Returns
    -------
    Path
        Path to the saved dashboard file.
    """
    df = result.df
    pivots = result.pivots
    spots = result.sweet_spots
    cs = result.current_state

    # Determine output path
    if output_path is None:
        output_dir = Path("./reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        date_str = cs.timestamp.strftime("%Y%m%d")
        ext = "png" if static else "html"
        output_path = output_dir / f"{result.symbol}_{result.timeframe}_{date_str}_dashboard.{ext}"
    else:
        output_path = Path(output_path)
        # If output_path is a directory, construct the filename inside it
        if output_path.is_dir() or output_path.suffix == "":
            date_str = cs.timestamp.strftime("%Y%m%d")
            ext = "png" if static else "html"
            output_path = output_path / f"{result.symbol}_{result.timeframe}_{date_str}_dashboard.{ext}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = df["ts"] if "ts" in df.columns else df.index

    # -----------------------------------------------------------------------
    # Create 4-row subplot layout
    # -----------------------------------------------------------------------
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.50, 0.18, 0.16, 0.16],
        subplot_titles=(
            f"{result.symbol} {result.timeframe} — {_fmt_price(cs.close)}",
            "Volume / Sweet Spot Confidence",
            "RSI / MACD",
            "EW Position / Regime / Volatility",
        ),
    )

    # -----------------------------------------------------------------------
    # Row 1: Price candlestick + overlays
    # -----------------------------------------------------------------------
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=timestamps,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLCV",
            increasing_line_color="#2ecc71",
            decreasing_line_color="#e74c3c",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Bollinger Bands
    _add_bollinger_bands(fig, df, row=1)

    # EMAs
    _add_emas(fig, df, row=1)

    # Elliott Wave
    _add_elliott_wave(fig, result, df, row=1)

    # Fibonacci levels
    _add_fibonacci_levels(fig, result, row=1)

    # Regime background
    _add_regime_shapes(fig, df, row=1)

    # Sweet spots
    _add_sweet_spot_markers(fig, spots, df, row=1)

    # -----------------------------------------------------------------------
    # Row 2: Volume (Tab A) / Sweet Spot Confidence (Tab B)
    # -----------------------------------------------------------------------
    _add_volume(fig, df, row=2)
    _add_sweet_spot_confidence_bars(fig, spots, row=2)

    # -----------------------------------------------------------------------
    # Row 3: RSI
    # -----------------------------------------------------------------------
    _add_rsi(fig, df, row=3)

    # -----------------------------------------------------------------------
    # Row 4: EW Position / Regime Probs / Volatility
    # -----------------------------------------------------------------------
    _add_ew_position(fig, df, row=4)
    _add_regime_probs(fig, df, row=4)
    _add_volatility(fig, df, row=4)

    # -----------------------------------------------------------------------
    # Layout configuration
    # -----------------------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=(
                f"Kairon Analysis: {result.symbol} {result.timeframe} | "
                f"{cs.timestamp.strftime('%Y-%m-%d')} | "
                f"Close: {_fmt_price(cs.close)} | "
                f"Regime: {cs.regime.upper()}"
            ),
            font=dict(size=16),
        ),
        height=1200,
        template="plotly_dark",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9),
        ),
        margin=dict(l=60, r=40, t=80, b=40),
    )

    # Row 1 y-axis — price
    fig.update_yaxes(title_text="Price", row=1, col=1, side="right")
    # Row 2 y-axis — volume/confidence
    fig.update_yaxes(title_text="Volume", row=2, col=1, side="right")
    # Row 3 y-axis — RSI
    fig.update_yaxes(title_text="RSI", row=3, col=1, side="right")
    # Row 4 y-axis — EW/Regime
    fig.update_yaxes(title_text="EW / Regime", row=4, col=1, side="right")

    # RSI range
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    # Add sweet spot annotations on the price chart
    for spot in spots:
        emoji = "^" if spot.direction == "BUY" else "v"
        color = "#2ecc71" if spot.direction == "BUY" else "#e74c3c"
        fig.add_annotation(
            x=spot.timestamp,
            y=spot.price,
            text=f"{emoji} {spot.direction}<br>{_fmt_price(spot.price)}<br>"
                 f"Score: {spot.combined_score:.0%}",
            showarrow=True,
            arrowhead=2 if spot.direction == "SELL" else 3,
            arrowsize=0.8,
            arrowcolor=color,
            font=dict(size=9, color=color),
            row=1, col=1,
        )

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------
    if static:
        fig.write_image(str(output_path), scale=dpi / 96)
    else:
        # Add metadata as HTML comment
        meta = (
            f"<!-- Kairon Analysis Dashboard\n"
            f"Symbol: {result.symbol}\n"
            f"Timeframe: {result.timeframe}\n"
            f"Date: {cs.timestamp}\n"
            f"Features: {len(result.feature_names)}\n"
            f"Sweet Spots: {len(spots)}\n"
            f"Models: {len(result.model_predictions)}\n"
            f"-->\n"
        )
        html_content = fig.to_html(
            include_plotlyjs="cdn",
            full_html=True,
            config=dict(
                responsive=True,
                scrollZoom=True,
                displaylogo=False,
                modeBarButtonsToAdd=[
                    "drawline",
                    "drawopenpath",
                    "eraseshape",
                ],
            ),
        )
        output_path.write_text(meta + html_content, encoding="utf-8")

    return output_path