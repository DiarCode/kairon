"""Background verification thread (US-004).

Polls the :class:`kairon.store.runs.RunStore` for runs whose verification
is due (``now_utc - created_at_utc >= HORIZON_PROFILES[horizon].duration_hours``),
fetches the asset's current price via
:func:`kairon.live.feed.fetch_current_price`, and writes back
``actual_pct``, ``delta_pct``, ``status`` to the store.

For v1 the verifier is started/stopped by FastAPI's lifespan and is
expected to be invoked by tests as ``run_once(run_store, fetch_price_fn,
now_utc)`` — no live thread is spun up in tests.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from kairon.analysis.contracts import RunResult, VerificationStatus
from kairon.analysis.loader import load_csv

if TYPE_CHECKING:
    from kairon.store.runs import RunStore

logger = logging.getLogger(__name__)

# A run is a "hit" if the actual % change is within 0.5% of the predicted %.
HIT_TOLERANCE_PCT: float = 0.005


def _read_base_price(run: RunResult, csv_path: Path) -> float:
    """Return the base price for a run, reading the last close of the CSV.

    The CSV path is captured at run time; we re-read it on verification to
    keep the base price consistent with what the analysis pipeline saw.
    """
    if run.base_price > 0:
        return float(run.base_price)
    try:
        result = load_csv(csv_path, symbol=run.asset)
        close = result.table.column("close").to_pylist()
        if not close:
            return 0.0
        return float(close[-1])
    except Exception as e:
        logger.warning("verifier: failed to re-read base price from %s: %s", csv_path, e)
        return 0.0


def run_once(
    run_store: RunStore,
    fetch_price_fn: Callable[[str, str], float],
    now_utc: datetime,
    *,
    base_price_reader: Callable[[RunResult, Path], float] | None = None,
) -> int:
    """Run one verification pass. Returns the number of runs verified.

    Parameters
    ----------
    run_store
        The :class:`RunStore` to walk.
    fetch_price_fn
        A callable ``(asset: str, venue: str) -> float``. In production this
        is :func:`kairon.live.feed.fetch_current_price`; in tests it is a
        stub that returns a known price.
    now_utc
        The current time (tz-aware UTC). The verifier does not call
        :func:`datetime.now` itself — tests must inject the clock.
    base_price_reader
        Optional override for reading the base price from the stored CSV.
        Defaults to :func:`_read_base_price`.
    """
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be tz-aware UTC")
    due_ids = run_store.mark_due(now_utc)
    if not due_ids:
        return 0
    reader = base_price_reader or _read_base_price
    verified = 0
    for run_id in due_ids:
        run = run_store.get(run_id)
        if run is None:
            continue
        csv_path = run_store.get_csv_path(run_id)
        if csv_path is None:
            logger.warning("verifier: no csv path for run %s; skipping", run_id)
            continue
        base_price = reader(run, csv_path)
        if base_price <= 0:
            logger.warning("verifier: no base price for run %s; skipping", run_id)
            continue
        try:
            current_price = fetch_price_fn(run.asset, "binance")
        except Exception as e:
            logger.warning("verifier: fetch_price failed for %s: %s", run.asset, e)
            continue
        actual_pct = (current_price - base_price) / base_price
        delta_pct = actual_pct - run.models[0].predicted_pct  # use trend as reference
        status: VerificationStatus = "hit" if abs(delta_pct) <= HIT_TOLERANCE_PCT else "missed"
        run_store.update_verification(
            run_id, actual_pct=actual_pct, delta_pct=delta_pct, verified_at_utc=now_utc, status=status
        )
        verified += 1
    return verified


class VerifierThread:
    """A daemon thread that calls :func:`run_once` on an interval.

    Lifecycle:
    - :meth:`start` is idempotent.
    - :meth:`stop` joins the thread with a timeout. Safe to call from
      FastAPI's lifespan shutdown.
    """

    def __init__(
        self,
        run_store: RunStore,
        fetch_price_fn: Callable[[str, str], float],
        *,
        poll_interval_seconds: float = 60.0,
    ) -> None:
        self._run_store = run_store
        self._fetch_price_fn = fetch_price_fn
        self._poll = poll_interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="kairon-verifier", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                run_once(self._run_store, self._fetch_price_fn, datetime.now(UTC))
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("verifier loop error: %s", e)
            self._stop.wait(self._poll)


__all__ = ["HIT_TOLERANCE_PCT", "VerifierThread", "run_once"]
