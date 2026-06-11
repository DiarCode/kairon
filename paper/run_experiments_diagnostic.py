"""Third experiment: in-sample vs out-of-sample + coverage-accuracy.

This is the diagnostic that quant researchers actually use:

- The in-sample accuracy shows the model's capacity (and overfit ceiling).
- The out-of-sample accuracy shows the realised edge.
- The coverage-accuracy curve shows what the model is worth
  at the high-confidence threshold, which is what you'd
  actually trade.

We also compute the calibration curve (reliability diagram)
data: 10 bins of predicted probability, empirical hit rate
in each bin.

We use a *less* adversarial structured dataset: a slow random
walk with a positive drift, so the model has a real
opportunity to learn the direction signal.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa

from kairon.data.io import OHLCV_SCHEMA
from kairon.labels.direction import make_direction_labels
from kairon.labels.schema import LabelKind, LabelSpec
from kairon.models.contracts import FeatureMatrix
from kairon.models.ensemble import EnsembleSpec, TopKConfidenceEnsemble
from kairon.models.linear import LinearConfig, LogisticRegressionModel
from kairon.models.tree import RandomForestConfig, RandomForestModel
from kairon.splits.walkforward import SplitSpec, walkforward


def make_drift_ohlcv(n: int, *, every_s: int, drift: float, vol: float, seed: int) -> pa.Table:
    """Make OHLCV with a slow positive drift (trending up)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=n)
    close = 30000.0 * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[30000.0], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3 * 30000 * vol, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3 * 30000 * vol, size=n))
    volume = rng.uniform(100, 1000, size=n)
    ts = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=every_s * i) for i in range(n)]
    return pa.table(
        {
            "ts": pa.array(ts, type=pa.timestamp("us", tz="UTC")),
            "open": open_, "high": high, "low": low, "close": close, "volume": volume,
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


def calibration_curve(y_true: np.ndarray, y_proba: np.ndarray, *, n_bins: int = 10, pos_idx: int = 1) -> list[dict]:
    """Compute reliability-diagram data: per-bin, the mean predicted
    proba of the +1 class vs the empirical hit rate (y==+1)."""
    if y_proba.ndim == 1:
        p = y_proba
    else:
        p = y_proba[:, pos_idx]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if mask.sum() == 0:
            out.append({"bin_lo": float(lo), "bin_hi": float(hi),
                        "n": 0, "mean_p": float("nan"), "hit_rate": float("nan")})
            continue
        mean_p = float(p[mask].mean())
        hit_rate = float((y_true[mask] == 1).mean())
        out.append({"bin_lo": float(lo), "bin_hi": float(hi),
                    "n": int(mask.sum()), "mean_p": mean_p, "hit_rate": hit_rate})
    return out


def coverage_accuracy(y_true: np.ndarray, y_proba: np.ndarray, *, thresholds: list[float], pos_idx: int = 1) -> list[dict]:
    """At each confidence threshold, compute the coverage (fraction of rows kept)
    and the accuracy on the kept rows. y_proba is 2-D (n, k); pos_idx is the
    column of the +1 class."""
    if y_proba.ndim == 1:
        p_pos = y_proba
        p_neg = 1.0 - p_pos
    else:
        p_pos = y_proba[:, pos_idx]
        p_neg = 1.0 - p_pos
    out = []
    for t in thresholds:
        # max-proba of the predicted class
        max_p = np.maximum(p_pos, p_neg)
        mask = max_p >= t
        if mask.sum() == 0:
            out.append({"threshold": t, "coverage": 0.0, "n": 0, "acc": float("nan")})
            continue
        y_hat = np.where(p_pos >= 0.5, 1, -1)
        acc = float((y_hat[mask] == y_true[mask]).mean())
        out.append({"threshold": t, "coverage": float(mask.mean()),
                    "n": int(mask.sum()), "acc": acc})
    return out


def main() -> int:
    print("=" * 60, file=sys.stderr)
    print("Kairon: in-sample vs OOS + coverage-accuracy + calibration", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Slow drift, low vol -> the model can learn the trend
    table = make_drift_ohlcv(n=3000, every_s=3600, drift=0.002, vol=0.01, seed=42)
    close = table.column("close").to_numpy()
    print(f"  dataset: 3000 1h bars, drift=+0.2%/bar, vol=1.0%", file=sys.stderr)
    print(f"  start: {close[0]:.0f}, end: {close[-1]:.0f}, total ret: {(close[-1]/close[0]-1)*100:.1f}%", file=sys.stderr)

    fm, y = make_features(table, horizon="1h")
    print(f"  features: {fm.values.shape}, classes: {dict(zip(*np.unique(y, return_counts=True)))}", file=sys.stderr)
    print(f"  majority class frac: {float(np.max(np.bincount(y + 1)) / len(y)):.4f}", file=sys.stderr)

    # 70/30 train/test split (single split, not walk-forward)
    n = fm.n_rows
    cut = int(n * 0.7)
    Xtr, ytr = fm.values[:cut], y[:cut]
    Xte, yte = fm.values[cut:], y[cut:]

    results: dict = {"per_backend": [], "coverage_accuracy": [], "calibration": []}

    print("\n[1/3] In-sample vs OOS accuracy...", file=sys.stderr)
    for name, m in [
        ("logreg", LogisticRegressionModel(LinearConfig(C=1.0, random_state=0))),
        ("random_forest", RandomForestModel(RandomForestConfig(n_estimators=200, random_state=0, n_jobs=1))),
    ]:
        tr_fm = FeatureMatrix(values=Xtr, feature_names=fm.feature_names)
        te_fm = FeatureMatrix(values=Xte, feature_names=fm.feature_names)
        trained = m.fit(tr_fm, ytr)
        in_pred = m.predict(trained, tr_fm)
        out_pred = m.predict(trained, te_fm)
        in_acc = float((in_pred.y_class == ytr).mean())
        oos_acc = float((out_pred.y_class == yte).mean())
        in_ll = float("nan"); oos_ll = float("nan")
        if in_pred.y_proba is not None and out_pred.y_proba is not None:
            # Multi-class log loss
            for preds, ys, store in [(in_pred, ytr, "in"), (out_pred, yte, "oos")]:
                p = preds.y_proba
                if p.ndim == 1:
                    p = np.clip(p, 1e-15, 1 - 1e-15)
                    ll = -(ys * np.log(p) + (1 - ys) * np.log(1 - p)).mean()
                else:
                    p = np.clip(p, 1e-15, 1 - 1e-15)
                    oh = np.eye(p.shape[1])[ys]
                    ll = -(oh * np.log(p)).sum(axis=1).mean()
                if store == "in":
                    in_ll = float(ll)
                else:
                    oos_ll = float(ll)
        results["per_backend"].append({
            "name": name, "in_acc": in_acc, "oos_acc": oos_acc,
            "in_ll": in_ll, "oos_ll": oos_ll,
            "overfit_gap": in_acc - oos_acc,
        })
        print(f"  {name}: in_acc={in_acc:.4f} (ll={in_ll:.4f}), "
              f"oos_acc={oos_acc:.4f} (ll={oos_ll:.4f}), "
              f"gap={in_acc - oos_acc:+.4f}", file=sys.stderr)

    # Ensemble
    ens = TopKConfidenceEnsemble(
        [LogisticRegressionModel(LinearConfig(C=1.0, random_state=0)),
         RandomForestModel(RandomForestConfig(n_estimators=200, random_state=0, n_jobs=1))],
        EnsembleSpec(),
    )
    tr_fm = FeatureMatrix(values=Xtr, feature_names=fm.feature_names)
    te_fm = FeatureMatrix(values=Xte, feature_names=fm.feature_names)
    trained = ens.fit(tr_fm, ytr)
    in_pred = ens.predict(trained, tr_fm)
    out_pred = ens.predict(trained, te_fm)
    in_acc = float((in_pred.y_class == ytr).mean())
    oos_acc = float((out_pred.y_class == yte).mean())
    results["per_backend"].append({
        "name": "ensemble_lr_rf", "in_acc": in_acc, "oos_acc": oos_acc,
        "in_ll": float("nan"), "oos_ll": float("nan"),
        "overfit_gap": in_acc - oos_acc,
    })
    print(f"  ensemble_lr_rf: in_acc={in_acc:.4f}, oos_acc={oos_acc:.4f}, "
          f"gap={in_acc - oos_acc:+.4f}", file=sys.stderr)

    # 2. Coverage-accuracy curve (OOS, on the best single backend = RF)
    print("\n[2/3] Coverage-accuracy curve (RF, OOS)...", file=sys.stderr)
    rf = RandomForestModel(RandomForestConfig(n_estimators=200, random_state=0, n_jobs=1))
    trained = rf.fit(tr_fm, ytr)
    out_pred = rf.predict(trained, te_fm)
    pos_idx = list(trained.classes).index(1)
    print(f"  RF: classes={trained.classes}, pos_idx={pos_idx}", file=sys.stderr)
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    ca = coverage_accuracy(yte, out_pred.y_proba, thresholds=thresholds, pos_idx=pos_idx)
    results["coverage_accuracy"] = ca
    for row in ca:
        if row["n"] > 0 and not np.isnan(row["acc"]):
            print(f"  thresh={row['threshold']:.2f}: coverage={row['coverage']:.2%}, "
                  f"n={row['n']}, acc={row['acc']:.4f}", file=sys.stderr)
        else:
            print(f"  thresh={row['threshold']:.2f}: empty", file=sys.stderr)

    # 3. Calibration curve
    print("\n[3/3] Calibration curve (RF, OOS)...", file=sys.stderr)
    cal = calibration_curve(yte, out_pred.y_proba, n_bins=10, pos_idx=pos_idx)
    results["calibration"] = cal
    for row in cal:
        if row["n"] > 0:
            print(f"  bin=[{row['bin_lo']:.1f},{row['bin_hi']:.1f}]: n={row['n']}, "
                  f"mean_p(+1)={row['mean_p']:.3f}, hit_rate(+1)={row['hit_rate']:.3f}", file=sys.stderr)
        else:
            print(f"  bin=[{row['bin_lo']:.1f},{row['bin_hi']:.1f}]: empty", file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print("Summary:", file=sys.stderr)
    for r in results["per_backend"]:
        print(f"  {r['name']:>20s}: in={r['in_acc']:.4f}, oos={r['oos_acc']:.4f}, gap={r['overfit_gap']:+.4f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    json.dump({"results": results, "meta": {
        "n_train": int(cut),
        "n_test": int(n - cut),
        "n_features": int(fm.n_features),
        "horizon": "1h",
        "label_kind": "direction",
        "drift": 0.002,
        "vol": 0.01,
        "dataset": "drift_random_walk",
        "n_bars": int(len(close)),
        "total_return": float(close[-1] / close[0] - 1),
        "majority_class_frac": float(np.max(np.bincount(y + 1)) / len(y)),
    }}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
