"""`kairon report` subcommand — generate markdown report only."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from kairon.cli.utils import console_info, console_success

report_app = typer.Typer(
    name="report",
    help="Generate analysis reports (no dashboard).",
    no_args_is_help=True,
)


@report_app.command()
def markdown(
    data: Path = typer.Argument(
        ...,
        help="Path to CSV file with OHLCV columns",
        exists=True,
    ),
    symbol: str = typer.Option("AUTO", "--symbol", "-s", help="Asset symbol"),
    timeframe: Optional[str] = typer.Option(
        None, "--timeframe", "-t",
        help="Override timeframe auto-detection",
    ),
    horizon: Optional[str] = typer.Option(
        None, "--horizon", "-h",
        help="Prediction horizon",
    ),
    pivot_scale: float = typer.Option(1.5, "--pivot-scale", help="EW zigzag sensitivity"),
    features: str = typer.Option("all", "--features", "-f", help="Feature set"),
    no_model: bool = typer.Option(False, "--no-model", help="Skip model prediction"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    threshold: float = typer.Option(0.45, "--threshold", help="Sweet spot threshold"),
    equity: float = typer.Option(10000.0, "--equity", "-e", help="Portfolio equity"),
) -> None:
    """Generate a markdown analysis report."""
    from kairon.analysis.loader import load_csv
    from kairon.analysis.engine import run_analysis
    from kairon.analysis.report import generate_markdown_report

    if symbol == "AUTO":
        symbol = data.stem.split("_")[0].upper()

    output_dir = output or Path("./reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    console_info(f"Loading data from {data}")
    load_result = load_csv(data, symbol=symbol, timeframe_override=timeframe)

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

    console_info("Generating markdown report...")
    report = generate_markdown_report(result)
    date_str = result.current_state.timestamp.strftime("%Y%m%d")
    report_path = output_dir / f"{symbol}_{load_result.timeframe.name}_{date_str}_analysis.md"
    report_path.write_text(report, encoding="utf-8")
    console_success(f"Report saved to {report_path}")