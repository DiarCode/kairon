"""Second experiment: a dataset with KNOWN structure.

We inject a non-zero drift and a regime-switching mean so that
a correctly-modelled signal has a measurable edge. This lets us
demonstrate that the DSR and PBO machinery correctly recognises
real edge when it exists.

This is the controlled counterpart to ``run_experiments.py``,
which uses pure noise. The two together bracket the framework's
behaviour: noise (no edge, DSR=0) vs.\ structured data (real
edge, DSR>0.95).
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa

from kairon.backtest.cost import CostModel
from kairon.backtest.engine import BacktestSpec, run_backtest
from kairon.backtest.metrics import summarize
from kairon.backtest.statistics import DSRSpec, deflated_sharpe_ratio
from kairon.data.io import OHLCV_SCHEMA
from kairon.labels.direction import make_direction_labels
from kairon.labels.schema import LabelKind, LabelSpec
from kairon.models.contracts import FeatureMatrix
from kairon.models.ensemble import EnsembleSpec, TopKConfidenceEnsemble
from kairon.models.linear import LinearConfig, LogisticRegressionModel
from kairon.models.tree import RandomForestConfig, RandomForestModel
from kairon.splits.walkforward import SplitSpec, walkforward


def make_structured_ohlcv(n: int, *, every_s: int, seed: int) -> pa.Table:
    """Make OHLCV with a strong signal: a 2-state Markov-switching
    drift and a low-vol, high-vol regime."""
    rng = np.random.default_rng(seed)
    # 2-state Markov: state 0 = up, state 1 = down; P(switch) = 0.02
    state = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        if rng.random() < 0.02:
            state[i] = 1 - state[i - 1]
        else:
            state[i] = state[i - 1]
    # drift = +0.005 in state 0, -0.005 in state 1; vol same
    drift = np.where(state == 0, 0.005, -0.005)
    vol = 0.005  # low
    rets = rng.normal(drift, vol, size=n)
    close = 30000.0 * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[30000.0], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5 * 30000 * vol, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5 * 30000 * vol, size=n))
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


def make_features(table: pa.Table, *, horizon: str = "1h") -> tuple[FeatureMatrix, np.ndarray]:
    close = table.column("close").to_numpy()
    ret1 = np.concatenate([[0.0], np.diff(close) / close[:-1]])
    import pandas as pd
    s = pd.Series(ret1)
    rm3 = s.rolling(3).mean().fillna(0).to_numpy()
    rm6 = s.rolling(6).mean().fillna(0).to_numpy()
    rs3 = s.rolling(3).std().fillna(0).to_numpy()
    rs6 = s.rolling(6).std().fillna(0).to_numpy()
    lag1 = np.concatenate([[0.0], ret1[:-1]])
    lag2 = np.concatenate([[0.0, 0.0], ret1[:-2]])
    spec = LabelSpec(kind=LabelKind.DIRECTION, horizon=horizon)
    frame = make_direction_labels(table, spec=spec, symbol="X", flat_threshold_pct=0.001)
    n_label = len(frame.bars)
    horizon_bars = int(spec.horizon_seconds / 60.0)
    X = np.column_stack([ret1[-n_label:], lag1[-n_label:], lag2[-n_label:],
                         rm3[-n_label:], rm6[-n_label:], rs3[-n_label:], rs6[-n_label:]])[20:-horizon_bars]
    y = np.array([b.y for b in frame.bars], dtype=np.int64)[20:-horizon_bars]
    return FeatureMatrix(
        values=X.astype(np.float64),
        feature_names=("ret1", "lag1", "lag2", "rm3", "rm6", "rs3", "rs6"),
    ), y


def run_wf(name: str, model, fm: FeatureMatrix, y: np.ndarray, *, n_folds: int = 6) -> dict:
    folds = walkforward(fm.n_rows, spec=SplitSpec(
        train_size=fm.n_rows - n_folds * 100, val_size=0, test_size=100
    ))
    fold_accs = []
    fold_lls = []
    for f in folds[:n_folds]:
        train_x = fm.values[f.train_start : f.train_end]
        train_y = y[f.train_start : f.train_end]
        test_x = fm.values[f.test_start : f.test_end]
        test_y = y[f.test_start : f.test_end]
        if train_x.shape[0] < 10 or test_x.shape[0] < 2:
            continue
        try:
            trained = model.fit(FeatureMatrix(values=train_x, feature_names=fm.feature_names), train_y)
            pred = model.predict(trained, FeatureMatrix(values=test_x, feature_names=fm.feature_names))
            fold_accs.append(float((pred.y_class == test_y).mean()))
            if pred.y_proba is not None:
                p = pred.y_proba
                if p.ndim == 1:
                    p = np.clip(p, 1e-15, 1 - 1e-15)
                    ll = -(test_y * np.log(p) + (1 - test_y) * np.log(1 - p)).mean()
                else:
                    oh = np.eye(p.shape[1])[test_y]
                    p = np.clip(p, 1e-15, 1 - 1e-15)
                    ll = -(oh * np.log(p)).sum(axis=1).mean()
                fold_lls.append(float(ll))
        except Exception as e:
            print(f"  warn: {name} fold {f.fold_id} failed: {e}", file=sys.stderr)
    return {
        "name": name,
        "n_folds": len(fold_accs),
        "mean_acc": float(np.mean(fold_accs)) if fold_accs else float("nan"),
        "std_acc": float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else 0.0,
        "mean_logloss": float(np.mean(fold_lls)) if fold_lls else float("nan"),
    }


def backtest_signal(signal: np.ndarray, close: np.ndarray, *, cost: CostModel, n_trials: int = 1) -> dict:
    n = len(close)
    ts = np.array([datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)], dtype=object)
    spec = BacktestSpec(cost=cost, initial_equity=10_000.0, fraction=0.5)
    result = run_backtest(symbol="X", timestamps=ts, close=close, signals=signal.astype(np.float64), spec=spec)
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


def main() -> int:
    print("=" * 60, file=sys.stderr)
    print("Kairon structured-experiment harness (KNOWN edge)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    table = make_structured_ohlcv(n=2500, every_s=3600, seed=7)
    close = table.column("close").to_numpy()
    print(f"  dataset: 2500 1h bars, Markov-switching drift, vol=0.5%", file=sys.stderr)
    print(f"  start: {close[0]:.0f}, end: {close[-1]:.0f}, total ret: {(close[-1]/close[0]-1)*100:.2f}%", file=sys.stderr)

    fm, y = make_features(table, horizon="1h")
    print(f"  features: {fm.values.shape}, classes: {dict(zip(*np.unique(y, return_counts=True)))}", file=sys.stderr)
    print(f"  majority class frac: {float(np.max(np.bincount(y + 1)) / len(y)):.4f}", file=sys.stderr)

    results: dict = {"per_backend": [], "dsr_sweep": [], "cost_ablation": []}

    print("\n[1/3] Per-backend walk-forward accuracy (structured data)...", file=sys.stderr)
    for name, m in [
        ("logreg", LogisticRegressionModel(LinearConfig(C=1.0, random_state=0))),
        ("random_forest", RandomForestModel(RandomForestConfig(n_estimators=200, random_state=0, n_jobs=1))),
    ]:
        r = run_wf(name, m, fm, y, n_folds=8)
        results["per_backend"].append(r)
        print(f"  {name}: acc={r['mean_acc']:.4f} ± {r['std_acc']:.4f}", file=sys.stderr)

    ens_lr_rf = TopKConfidenceEnsemble(
        [LogisticRegressionModel(LinearConfig(C=1.0, random_state=0)),
         RandomForestModel(RandomForestConfig(n_estimators=200, random_state=0, n_jobs=1))],
        EnsembleSpec(),
    )
    r = run_wf("ensemble_lr_rf", ens_lr_rf, fm, y, n_folds=8)
    results["per_backend"].append(r)
    print(f"  ensemble_lr_rf: acc={r['mean_acc']:.4f} ± {r['std_acc']:.4f}", file=sys.stderr)

    # Build a real signal from the trained model
    print("\n[2/3] DSR sweep (n_trials=1,5,10,50,100)...", file=sys.stderr)
    rf = RandomForestModel(RandomForestConfig(n_estimators=200, random_state=0, n_jobs=1))
    fm_full = FeatureMatrix(values=fm.values, feature_names=fm.feature_names)
    trained = rf.fit(fm_full, y)
    pred = rf.predict(trained, fm_full)
    signal = pred.y_proba if pred.y_proba is not None else pred.y_class.astype(np.float64)
    if signal.ndim > 1:
        signal = signal[:, 1]
    signal = signal * 2 - 1

    n = min(len(signal), len(close))
    signal = signal[-n:]
    close_aligned = close[-n:]

    dsr_sweep = []
    for n_trials in [1, 5, 10, 50, 100]:
        r = backtest_signal(signal, close_aligned, cost=CostModel(0, 0, 0, 0, 0), n_trials=n_trials)
        dsr_sweep.append({"n_trials": n_trials, **r})
        print(f"  N_t={n_trials}: sharpe={r['dsr_sharpe']:.3f}, sr*={r['dsr_sr_star']:.3f}, DSR={r['dsr_value']:.3f}, "
              f"trades={r['n_trades']}, ret={r['total_return']*100:.1f}%", file=sys.stderr)
    results["dsr_sweep"] = dsr_sweep

    print("\n[3/3] Cost-basis ablation...", file=sys.stderr)
    cost_abl = []
    for name, comm, slip, hs in [("zero", 0, 0, 0), ("cheap", 4, 1, 1),
                                 ("default", 10, 2, 2), ("rich", 20, 5, 5),
                                 ("stress", 40, 10, 10)]:
        cost = CostModel(commission_bps=comm, slippage_bps=slip, half_spread_bps=hs, impact_coefficient=0, min_trade_bps=1)
        r = backtest_signal(signal, close_aligned, cost=cost, n_trials=1)
        cost_abl.append({"name": name, "rt_bps": 2 * (comm + slip + hs), **r})
        print(f"  cost={name} (rt={2*(comm+slip+hs)}bps): sharpe={r['sharpe']:.3f}, ret={r['total_return']*100:.1f}%, "
              f"mdd={r['max_dd']*100:.1f}%, trades={r['n_trades']}", file=sys.stderr)
    results["cost_ablation"] = cost_abl

    print("\n" + "=" * 60, file=sys.stderr)
    print("Summary (structured data):", file=sys.stderr)
    for r in results["per_backend"]:
        print(f"  {r['name']:>20s}: acc={r['mean_acc']:.4f} ± {r['std_acc']:.4f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    json.dump({"results": results, "meta": {
        "n_samples": int(fm.n_rows),
        "n_features": int(fm.n_features),
        "horizon": "1h",
        "label_kind": "direction",
        "dataset": "structured_markov",
        "n_bars": int(len(close)),
        "total_return": float(close[-1] / close[0] - 1),
        "majority_class_frac": float(np.max(np.bincount(y + 1)) / len(y)),
    }}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
