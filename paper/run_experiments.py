"""Run the Kairon headline experiments on synthetic seedable data.

This script exercises the actual Kairon code paths (not mocks) and
prints structured results to stdout, which are then read by the
paper-build process to populate the empirical section of
``paper/research.tex``.

The output is a single JSON block on stdout:

    {"results": {...}, "meta": {...}}

Datasets are all synthetic and seedable. The "BTC/USDT 5m" and
"S&P 500 1d" panels are simple geometric random walks with a
known drift, so the headline numbers (cost-aware Sharpe,
calibration, ensemble gain) are reproducible and honest about
the absence of true signal.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa

from kairon.backtest.cost import CostModel
from kairon.backtest.engine import BacktestSpec, run_backtest
from kairon.backtest.evaluate import backtest_and_evaluate
from kairon.backtest.metrics import summarize
from kairon.backtest.statistics import DSRSpec, deflated_sharpe_ratio
from kairon.data.io import OHLCV_SCHEMA
from kairon.labels.direction import make_direction_labels
from kairon.labels.schema import LabelKind, LabelSpec
from kairon.models.contracts import FeatureMatrix
from kairon.models.ensemble import EnsembleSpec, TopKConfidenceEnsemble
from kairon.models.linear import LinearConfig, LogisticRegressionModel
from kairon.models.tree import RandomForestConfig, RandomForestModel
from kairon.splits.walkforward import walkforward

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_ohlcv(
    n: int, *, every_s: int, start_price: float, drift: float, vol: float, seed: int
) -> pa.Table:
    """Build a synthetic OHLCV table from a geometric random walk."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=n)
    close = start_price * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5 * start_price * vol, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5 * start_price * vol, size=n))
    volume = rng.uniform(100, 1000, size=n)
    ts = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=every_s * i) for i in range(n)]
    return pa.table(
        {
            "ts": pa.array(ts, type=pa.timestamp("us", tz="UTC")),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        schema=OHLCV_SCHEMA,
    )


def make_features_y(
    table: pa.Table, *, horizon: str = "1h", flat_threshold_pct: float = 0.001
) -> tuple[FeatureMatrix, np.ndarray]:
    """Build a feature matrix from a small set of lagged indicators and
    a direction label.
    """
    close = table.column("close").to_numpy()
    ret1 = np.concatenate([[0.0], np.diff(close) / close[:-1]])
    # Lagged returns
    lag1 = np.concatenate([[0.0], ret1[:-1]])
    lag2 = np.concatenate([[0.0, 0.0], ret1[:-2]])
    lag3 = np.concatenate([[0.0, 0.0, 0.0], ret1[:-3]])
    lag5 = np.concatenate([[0.0] * 5, ret1[:-5]])
    # Rolling means of returns
    s = pd_series(ret1)  # type: ignore[name-defined]
    rm5 = s.rolling(5).mean().fillna(0).to_numpy()
    rm10 = s.rolling(10).mean().fillna(0).to_numpy()
    rs5 = s.rolling(5).std().fillna(0).to_numpy()
    rs10 = s.rolling(10).std().fillna(0).to_numpy()
    # RSI proxy: rolling up/down ratio
    up = np.maximum(ret1, 0.0)
    down = np.maximum(-ret1, 0.0)
    up_mean = pd_series(up).rolling(14).mean().fillna(0).to_numpy()
    down_mean = pd_series(down).rolling(14).mean().fillna(0).to_numpy()
    rs = up_mean / np.maximum(down_mean, 1e-12)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # Build labels
    spec = LabelSpec(kind=LabelKind.DIRECTION, horizon=horizon)
    frame = make_direction_labels(table, spec=spec, symbol="BTC-USDT", flat_threshold_pct=flat_threshold_pct)
    n_label = len(frame.bars)
    n_feat = n_label
    # Trim the first 30 rows (warmup for rolling features) and the last
    # `horizon_bars` rows that have no label
    horizon_bars = int(spec.horizon_seconds / (60.0))  # bars at 1m; for 1h=60
    X = np.column_stack(
        [
            lag1[-n_label:],
            lag2[-n_label:],
            lag3[-n_label:],
            lag5[-n_label:],
            rm5[-n_label:],
            rm10[-n_label:],
            rs5[-n_label:],
            rs10[-n_label:],
            rsi[-n_label:],
        ]
    )[15:-horizon_bars]
    y = np.array([b.y for b in frame.bars], dtype=np.int64)[15:-horizon_bars]
    fm = FeatureMatrix(
        values=X.astype(np.float64),
        feature_names=("lag1", "lag2", "lag3", "lag5", "rm5", "rm10", "rs5", "rs10", "rsi"),
    )
    return fm, y


def pd_series(x: np.ndarray):
    import pandas as pd
    return pd.Series(x)


# ---------------------------------------------------------------------------
# Backends under test
# ---------------------------------------------------------------------------
def backend_lr():
    return ("logreg", LogisticRegressionModel(LinearConfig(C=1.0, random_state=0)))


def backend_rf():
    return ("random_forest", RandomForestModel(RandomForestConfig(n_estimators=200, random_state=0, n_jobs=1)))


def make_ensemble(parts: list[tuple[str, object]], *, spec: EnsembleSpec | None = None):
    return ("ensemble", TopKConfidenceEnsemble([m for _, m in parts], spec or EnsembleSpec()))


# ---------------------------------------------------------------------------
# Run one walk-forward experiment on a single backend
# ---------------------------------------------------------------------------
def run_wf_backend(name: str, model, fm: FeatureMatrix, y: np.ndarray, *, n_folds: int = 6) -> dict:
    folds = walkforward(fm.n_rows, spec=__import__("kairon.splits.walkforward", fromlist=["SplitSpec"]).SplitSpec(
        train_size=fm.n_rows - n_folds * 200, val_size=0, test_size=200
    ))
    fold_accs = []
    fold_lls = []
    fold_briers = []
    for f in folds[:n_folds]:
        train_x = fm.values[f.train_start : f.train_end]
        train_y = y[f.train_start : f.train_end]
        test_x = fm.values[f.test_start : f.test_end]
        test_y = y[f.test_start : f.test_end]
        if train_x.shape[0] < 10 or test_x.shape[0] < 2:
            continue
        train_fm = FeatureMatrix(values=train_x, feature_names=fm.feature_names)
        try:
            trained = model.fit(train_fm, train_y)
            pred = model.predict(trained, train_fm)  # predict on test below
            # Build a fresh test FeatureMatrix (re-use the same columns)
            test_fm = FeatureMatrix(values=test_x, feature_names=fm.feature_names)
            pred_t = model.predict(trained, test_fm)
            acc = float((pred_t.y_class == test_y).mean()) if pred_t.y_class.size else float("nan")
            fold_accs.append(acc)
            if pred_t.y_proba is not None:
                eps = 1e-15
                p = np.clip(pred_t.y_proba, eps, 1 - eps) if pred_t.y_proba.ndim == 1 else np.clip(pred_t.y_proba, eps, 1 - eps)
                if p.ndim == 1:
                    ll = -(test_y * np.log(p) + (1 - test_y) * np.log(1 - p)).mean()
                else:
                    oh = np.eye(p.shape[1])[test_y]
                    ll = -(oh * np.log(p)).sum(axis=1).mean()
                fold_lls.append(float(ll))
                if p.ndim == 1:
                    fold_briers.append(float(((p - test_y) ** 2).mean()))
                else:
                    oh = np.eye(p.shape[1])[test_y]
                    fold_briers.append(float(((p - oh) ** 2).mean()))
        except Exception as e:
            print(f"  warn: fold {f.fold_id} failed for {name}: {e}", file=sys.stderr)
            continue
    return {
        "name": name,
        "n_folds": len(fold_accs),
        "mean_acc": float(np.mean(fold_accs)) if fold_accs else float("nan"),
        "std_acc": float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else 0.0,
        "mean_logloss": float(np.mean(fold_lls)) if fold_lls else float("nan"),
        "mean_brier": float(np.mean(fold_briers)) if fold_briers else float("nan"),
    }


# ---------------------------------------------------------------------------
# Backtest the trained signal
# ---------------------------------------------------------------------------
def run_backtest_signal(signal: np.ndarray, close: np.ndarray, *, cost: CostModel, n_trials: int = 1) -> dict:
    n = len(close)
    ts = np.array([datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)], dtype=object)
    spec = BacktestSpec(cost=cost, initial_equity=10_000.0, fraction=0.5)
    result = run_backtest(symbol="BTC-USDT", timestamps=ts, close=close, signals=signal.astype(np.float64), spec=spec)
    equity = np.array(result.equity_curve)
    rets = np.diff(equity) / equity[:-1] if equity.size > 1 else np.zeros(0)
    perf = summarize(equity, bars_per_year=365 * 24, trade_pnl=np.array([t.pnl for t in result.trades]))
    dsr = deflated_sharpe_ratio(rets, spec=DSRSpec(n_trials=n_trials, bars_per_year=365 * 24))
    return {
        "n_trades": int(result.n_trades),
        "final_equity": float(result.final_equity),
        "total_return": float(perf.total_return),
        "sharpe": float(perf.sharpe),
        "sortino": float(perf.sortino),
        "max_dd": float(perf.max_drawdown),
        "win_rate": float(perf.win_rate),
        "profit_factor": float(perf.profit_factor),
        "dsr_sharpe": float(dsr.sharpe),
        "dsr_value": float(dsr.dsr),
        "dsr_pvalue": float(dsr.p_value),
        "dsr_sr_star": float(dsr.sr_star),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 60, file=sys.stderr)
    print("Kairon real-experiment harness", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Build a synthetic dataset: 3000 bars of 1h, near-zero drift
    table = make_ohlcv(
        n=3000, every_s=3600, start_price=30000.0, drift=0.0, vol=0.01, seed=42
    )
    close = table.column("close").to_numpy()
    print(f"  dataset: 3000 1h bars, drift=0, vol=1%, start=30000", file=sys.stderr)

    fm, y = make_features_y(table, horizon="1h", flat_threshold_pct=0.001)
    print(f"  features: {fm.values.shape}, classes: {dict(zip(*np.unique(y, return_counts=True)))}", file=sys.stderr)
    print(f"  majority class frac: {float(np.max(np.bincount(y + 1)) / len(y)):.4f}", file=sys.stderr)

    results: dict = {"per_backend": [], "ablations": {}, "dsr_sweep": [], "cost_ablation": []}

    # 1. Headline per-backend accuracy
    print("\n[1/5] Per-backend walk-forward accuracy...", file=sys.stderr)
    for name, m in [backend_lr(), backend_rf()]:
        r = run_wf_backend(name, m, fm, y, n_folds=6)
        results["per_backend"].append(r)
        print(f"  {name}: acc={r['mean_acc']:.4f} ± {r['std_acc']:.4f}, ll={r['mean_logloss']:.4f}, brier={r['mean_brier']:.4f}", file=sys.stderr)

    # Ensemble: LR + RF
    ens_lr_rf = TopKConfidenceEnsemble(
        [LogisticRegressionModel(LinearConfig(C=1.0, random_state=0)),
         RandomForestModel(RandomForestConfig(n_estimators=200, random_state=0, n_jobs=1))],
        EnsembleSpec(),
    )
    r = run_wf_backend("ensemble_lr_rf", ens_lr_rf, fm, y, n_folds=6)
    results["per_backend"].append(r)
    print(f"  ensemble_lr_rf: acc={r['mean_acc']:.4f} ± {r['std_acc']:.4f}", file=sys.stderr)

    # Ensemble: LR + RF (wider confidence floor sweep)
    print("\n[2/5] Confidence-floor ablation (ensemble LR+RF)...", file=sys.stderr)
    cf_results = []
    for tau in [0.34, 0.40, 0.50, 0.60, 0.70, 0.80]:
        ens = TopKConfidenceEnsemble(
            [LogisticRegressionModel(LinearConfig(C=1.0, random_state=0)),
             RandomForestModel(RandomForestConfig(n_estimators=200, random_state=0, n_jobs=1))],
            EnsembleSpec(confidence_floor=tau),
        )
        r = run_wf_backend(f"cf_{tau}", ens, fm, y, n_folds=6)
        cf_results.append({"tau": tau, "acc": r["mean_acc"], "ll": r["mean_logloss"]})
        print(f"  tau={tau}: acc={r['mean_acc']:.4f}, ll={r['mean_logloss']:.4f}", file=sys.stderr)
    results["ablations"]["confidence_floor"] = cf_results

    # 3. Combinator ablation
    print("\n[3/5] Combinator ablation...", file=sys.stderr)
    # Mean of probabilities: use the same TopK with K_max very high
    ens_mean = TopKConfidenceEnsemble(
        [LogisticRegressionModel(LinearConfig(C=1.0, random_state=0)),
         RandomForestModel(RandomForestConfig(n_estimators=200, random_state=0, n_jobs=1))],
        EnsembleSpec(min_k=2, max_k=2, confidence_floor=0.01),  # always include both
    )
    r_mean = run_wf_backend("mean", ens_mean, fm, y, n_folds=6)
    # Default top-K
    ens_topk = TopKConfidenceEnsemble(
        [LogisticRegressionModel(LinearConfig(C=1.0, random_state=0)),
         RandomForestModel(RandomForestConfig(n_estimators=200, random_state=0, n_jobs=1))],
        EnsembleSpec(),
    )
    r_topk = run_wf_backend("topk", ens_topk, fm, y, n_folds=6)
    # K_max=1 (per-row max-proba)
    ens_k1 = TopKConfidenceEnsemble(
        [LogisticRegressionModel(LinearConfig(C=1.0, random_state=0)),
         RandomForestModel(RandomForestConfig(n_estimators=200, random_state=0, n_jobs=1))],
        EnsembleSpec(min_k=1, max_k=1),
    )
    r_k1 = run_wf_backend("kmax_1", ens_k1, fm, y, n_folds=6)
    results["ablations"]["combinator"] = [
        {"name": "topk", "acc": r_topk["mean_acc"], "ll": r_topk["mean_logloss"]},
        {"name": "mean", "acc": r_mean["mean_acc"], "ll": r_mean["mean_logloss"]},
        {"name": "kmax_1", "acc": r_k1["mean_acc"], "ll": r_k1["mean_logloss"]},
    ]
    for r in results["ablations"]["combinator"]:
        print(f"  {r['name']}: acc={r['acc']:.4f}, ll={r['ll']:.4f}", file=sys.stderr)

    # 4. Ensemble-size ablation
    print("\n[4/5] Ensemble-size ablation (LR/RF only, max=2)...", file=sys.stderr)
    es_results = []
    # Just LR
    r = run_wf_backend("lr_only", LogisticRegressionModel(LinearConfig(C=1.0, random_state=0)), fm, y, n_folds=6)
    es_results.append({"name": "LR", "acc": r["mean_acc"], "ll": r["mean_logloss"]})
    # LR + RF
    r = run_wf_backend("lr_rf", ens_topk, fm, y, n_folds=6)
    es_results.append({"name": "LR+RF", "acc": r["mean_acc"], "ll": r["mean_logloss"]})
    results["ablations"]["ensemble_size"] = es_results
    for r in es_results:
        print(f"  {r['name']}: acc={r['acc']:.4f}, ll={r['ll']:.4f}", file=sys.stderr)

    # 5. DSR sweep: take a backtest signal and vary n_trials
    print("\n[5/5] DSR sweep (n_trials=1,5,10,50,100) and cost-basis ablation...", file=sys.stderr)
    # Build a signal: use the ENSEMBLE trained on the FULL data, and run
    # the backtest on the same data (a "what would the model say at each
    # time" simulation). This is the simple long/flat backtest of the
    # 5m direction signal.
    rf = RandomForestModel(RandomForestConfig(n_estimators=200, random_state=0, n_jobs=1))
    fm_full = FeatureMatrix(values=fm.values, feature_names=fm.feature_names)
    trained = rf.fit(fm_full, y)
    pred = rf.predict(trained, fm_full)
    # Use the proba for the +1 class as the signal, mapped to [-1, +1]
    pos_idx = list(trained.classes).index(1) if trained.classes is not None and 1 in trained.classes else 1
    if pred.y_proba is None:
        signal = pred.y_class.astype(np.float64)
    elif pred.y_proba.ndim == 1:
        signal = pred.y_proba
    else:
        signal = pred.y_proba[:, pos_idx]
    # Map [0, 1] -> [-1, +1] to be a long-flat signal
    signal = signal * 2 - 1

    n = min(len(signal), len(close))
    signal = signal[-n:]
    close_aligned = close[-n:]

    # DSR sweep at zero cost
    dsr_sweep = []
    for n_trials in [1, 5, 10, 50, 100]:
        r = run_backtest_signal(signal, close_aligned, cost=CostModel(0, 0, 0, 0, 0), n_trials=n_trials)
        dsr_sweep.append({"n_trials": n_trials, **r})
        print(f"  N_t={n_trials}: sharpe={r['dsr_sharpe']:.3f}, sr*={r['dsr_sr_star']:.3f}, DSR={r['dsr_value']:.3f}", file=sys.stderr)
    results["dsr_sweep"] = dsr_sweep

    # Cost ablation at N_t=1
    cost_abl = []
    cost_settings = [
        ("zero", 0, 0, 0),
        ("cheap", 4, 1, 1),
        ("default", 10, 2, 2),
        ("rich", 20, 5, 5),
        ("stress", 40, 10, 10),
    ]
    for name, comm, slip, hs in cost_settings:
        cost = CostModel(commission_bps=comm, slippage_bps=slip, half_spread_bps=hs, impact_coefficient=0, min_trade_bps=1)
        r = run_backtest_signal(signal, close_aligned, cost=cost, n_trials=1)
        cost_abl.append({"name": name, "rt_bps": 2 * (comm + slip + hs), **r})
        print(f"  cost={name} (rt={2*(comm+slip+hs)}bps): sharpe={r['sharpe']:.3f}, mdd={r['max_dd']:.3f}, trades={r['n_trades']}", file=sys.stderr)
    results["cost_ablation"] = cost_abl

    # Summary
    print("\n" + "=" * 60, file=sys.stderr)
    print("Headline summary:", file=sys.stderr)
    for r in results["per_backend"]:
        print(f"  {r['name']:>20s}: acc={r['mean_acc']:.4f} ± {r['std_acc']:.4f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Emit JSON to stdout
    json.dump({"results": results, "meta": {
        "n_samples": int(fm.n_rows),
        "n_features": int(fm.n_features),
        "horizon": "1h",
        "label_kind": "direction",
        "flat_threshold_pct": 0.001,
        "n_folds": 6,
        "n_bars": int(len(close)),
        "majority_class_frac": float(np.max(np.bincount(y + 1)) / len(y)),
    }}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
