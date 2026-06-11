"""`kairon analyze` subcommand — run full analysis on OHLCV data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from kairon.cli.utils import console_error, console_info, console_success, console_warn, open_in_browser

analyze_app = typer.Typer(
    name="analyze",
    help="Analyze OHLCV data: run features, detect signals, open interactive dashboard.",
    no_args_is_help=True,
)


@analyze_app.command()
def run(
    data: Path = typer.Argument(
        ...,
        help="Path to CSV file with OHLCV columns",
        exists=True,
    ),
    symbol: str = typer.Option("AUTO", "--symbol", "-s", help="Asset symbol (e.g. BTC, ETH)"),
    timeframe: Optional[str] = typer.Option(
        None, "--timeframe", "-t",
        help="Override timeframe auto-detection (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)",
    ),
    horizon: Optional[str] = typer.Option(
        None, "--horizon", "-h",
        help="Prediction horizon (default: derived from timeframe)",
    ),
    pivot_scale: float = typer.Option(1.5, "--pivot-scale", help="EW zigzag sensitivity (default: 1.5)"),
    features: str = typer.Option("all", "--features", "-f", help="Feature set: all, default, phase1, phase2, phase3"),
    no_model: bool = typer.Option(False, "--no-model", help="Skip model prediction, use heuristic mode"),
    no_dashboard: bool = typer.Option(False, "--no-dashboard", help="Skip dashboard, generate report only"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory (default: ./reports)"),
    threshold: float = typer.Option(0.45, "--threshold", help="Sweet spot firing threshold (default: 0.45)"),
    static: bool = typer.Option(False, "--static", help="Export static PNG/PDF instead of interactive HTML"),
    equity: float = typer.Option(10000.0, "--equity", "-e", help="Portfolio equity for position sizing"),
) -> None:
    """Load OHLCV data, run feature pipeline, detect signals, open dashboard."""
    from kairon.analysis.loader import load_csv
    from kairon.analysis.engine import run_analysis
    from kairon.analysis.viz import build_dashboard
    from kairon.analysis.report import generate_markdown_report

    if symbol == "AUTO":
        symbol = data.stem.split("_")[0].upper()

    output_dir = output or Path("./reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    console_info(f"Loading data from [bold]{data}[/]")
    load_result = load_csv(data, symbol=symbol, timeframe_override=timeframe)
    console_success(f"Loaded {load_result.table.num_rows} bars of {load_result.timeframe.name} data for {symbol}")

    if not load_result.has_volume:
        console_warn("No volume data detected — volume-dependent features will be skipped")

    console_info("Running analysis pipeline...")
    result = run_analysis(
        load_result.table,
        symbol=load_result.symbol,
        timeframe=load_result.timeframe.name,
        has_volume=load_result.has_volume,
        feature_set=features,
        pivot_scale=pivot_scale,
        run_model=not no_model,
        horizon=horizon or load_result.timeframe.name,
        equity=equity,
        threshold=threshold,
    )

    if no_model:
        console_warn("No ML model -- heuristic mode only")

    # Markdown report
    console_info("Generating markdown report...")
    report = generate_markdown_report(result)
    date_str = result.current_state.timestamp.strftime("%Y%m%d")
    report_path = output_dir / f"{symbol}_{load_result.timeframe.name}_{date_str}_analysis.md"
    report_path.write_text(report, encoding="utf-8")
    console_success(f"Report saved to {report_path}")

    # Dashboard
    if not no_dashboard:
        console_info("Building interactive dashboard...")
        dashboard_path = build_dashboard(
            result,
            output_path=output_dir,
            static=static,
        )
        console_success(f"Dashboard saved to {dashboard_path}")
        if not static:
            open_in_browser(dashboard_path)

    # Console summary
    cs = result.current_state
    print()
    print("=" * 70)
    print(f"  {symbol} {load_result.timeframe.name.upper()} ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Price: ${cs.close:,.2f}")
    if result.model_predictions:
        for mp in result.model_predictions:
            dir_str = {"up": "UP ^", "down": "DOWN v", "flat": "FLAT -"}.get(mp.direction, mp.direction)
            print(f"  {mp.model_name}: {dir_str} ({mp.confidence:.0%})")
    print(f"  EW: W{int(cs.ew_wave_position)} "
          f"{'Impulse' if cs.ew_is_impulse else 'Corrective'} "
          f"{'^' if cs.ew_wave_direction > 0.5 else 'v' if cs.ew_wave_direction < -0.5 else '-'}")
    print(f"  Regime: {cs.regime} (trend={cs.regime_prob_trending:.0%})")
    print(f"  Hurst: {cs.hurst_exp:.3f}")
    print(f"  Sweet spots: {len(result.sweet_spots)} detected")
    for ss in result.sweet_spots[:5]:
        print(f"    {ss.direction} @ ${ss.price:,.2f} ({ss.timing_horizon}) "
              f"score={ss.combined_score:.2f} — {', '.join(ss.justifications[:2])}")
    if len(result.sweet_spots) > 5:
        print(f"    ... and {len(result.sweet_spots) - 5} more")
    print("=" * 70)