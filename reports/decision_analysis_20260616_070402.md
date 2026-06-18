# Kairon Decision Analysis — Testnet Session 2026-06-16 07:04:02 UTC

## Session overview

| Symbol | Orders | Local fills | Closed PnL (Bybit) | End-of-session flatten |
|---|---|---|---|---|
| BTC-USDT-PERP | 3 | 0 | -$47.96 | ✅ Flattened |
| ETH-USDT-PERP | 6 | 0 | $0.00 | ❌ Residual 0.88 short remained |

Both symbols produced exactly **one** trading decision during the 30-minute window.
The `run_testnet_symbol.py` wrapper enforces a 5-minute cooldown, but the strategy itself
also produced very few high-confidence signals because the 1-minute bars had almost
zero volume on testnet (volume_vs_avg was 0.0 for BTC and 1.0 with raw volume 0.0 for ETH).

## Root causes identified

1. **Local fills were not captured (`live_fills = 0`)**
   - Bybit WebSocket order stream messages were received, but the parser did not map
     `orderLinkId` back to the local `Order.id` and did not compute incremental fill
     quantity from `cumExecQty`.
   - Without local fills, `TradingLoop` never updated its tracked position, so the
     reconciler saw 100% drift.
   - **Fix applied**: `BybitBroker` now maintains `_intent_to_local` and
     `_order_cum_qty` maps; `parse_fill_from_order` emits incremental fills; the
     `TradingLoop` drains `stream_fills()` in a background task and updates positions.

2. **Cross-symbol reconciler drift**
   - The BTC worker saw the ETH broker position and vice-versa, causing 100% drift
     alerts on the *other* symbol.
   - **Fix applied**: `Reconciler` now accepts a `symbols` tuple and ignores broker
     positions outside the worker's managed set.

3. **ETH position not flattened at shutdown**
   - The old `_flatten` helper tried to flatten with a single market order. On testnet
     this repeatedly failed with `NoImmediateQtyToFill` / `110007` because of thin
     books and balance locking.
   - **Fix applied**: `BybitBroker.close_position()` now closes in reduce-only market
     chunks and falls back to a limit order placed slightly inside the spread.

4. **Low-quality signals entered the market**
   - BTC long was fired with `adx = null`, `volume_vs_avg = 0.0`, and mixed
     justifications (both bullish and bearish sub-conditions fired).
   - ETH short was fired with `rsi_14 = 99.0` (overbought), `volume = 0.0`, and
     conflicting structure scores.
   - **Fix applied**: `ComprehensiveStrategy` now has:
     - ADX > 22 gate (signals flattened if ADX too weak)
     - Volume gate (volume_vs_avg < 0.8 flattens)
     - Regime-aware confidence (low trending regime reduces confidence)
     - Stressed-regime kill-switch (stressed > 0.3 flattens)
     - Trend-alignment cap (confidence > 0.6 requires EMA and MACD aligned)
     - Swing-based SL/TP anchored to recent swing pivots.

## Decision-level evidence

### BTC-USDT-PERP entry decision

| Field | Value |
|---|---|
| Direction | Long (+1.0) |
| Confidence | 0.546 |
| ADX | `null` |
| Volume vs avg | 0.0 |
| RSI | 60.6 |
| MACD histogram | -0.41 (bearish) |
| Justifications | EMA continuation bullish, MACD negative, near swing low, near swing high, Williams %R overbought, OBV rising, price above VWAP |

**Diagnosis**: confluence was mixed (bullish and bearish sub-signals simultaneously). With the new gates, `adx = null` and `volume_vs_avg = 0.0` would have flattened this signal.

### ETH-USDT-PERP entry decision

| Field | Value |
|---|---|
| Direction | Short (-1.0) |
| Confidence | 0.498 |
| ADX | `null` |
| Volume vs avg | 1.0 (but raw volume = 0.0) |
| RSI | 99.0 (overbought) |
| MACD histogram | -18.6 (bearish) |
| Justifications | RSI overbought, MACD negative, near swing low, near swing high, price above VWAP |

**Diagnosis**: RSI was at an extreme overbought reading on a single flat bar with no volume. The new volume gate and ADX gate would have flattened this signal.

## Algorithm improvements implemented

1. **Strategy quality gates** (see above).
2. **Vol-aware short sizing**: `size_position_vol_aware` now accepts a `direction`
   argument and returns a signed size, so short signals are sized correctly.
3. **Cooldown-aware loop**: `CooledTradingLoop` suppresses rapid rebalancing during
   short testnet sessions.
4. **Worker subprocess `PYTHONPATH`**: `run_testnet_30min_btc_eth.py` now injects
   `src/` into the subprocess environment and supports `--help`, `--duration`,
   `--symbols`, and `--dry-run`.
5. **Incremental fill persistence**: fills from the WebSocket stream update order
   status and local position state in real time.

## Test results

- `py_compile` passes for all modified files.
- Targeted unit tests pass:
  - `tests/live/test_orchestrator.py`
  - `tests/live/test_reconciler.py`
  - `tests/live/test_strategy.py`
  - `tests/live/test_paper_broker.py`
  - `tests/policy/test_sizer.py`
  - Result: **89 passed**.
- `tests/live/test_bybit_broker.py::TestBybitBrokerIntegration::test_symbol_round_trip_on_testnet`
  fails because the testnet account currently has **zero available balance to withdraw**,
   so new orders are rejected with `ErrCode: 110007`. This is an account-state issue, not a
   code issue.

## Recommendations for next retest

1. **Resolve the testnet account balance issue first.** The account currently holds a
   residual **0.88 ETH short position** at avg entry ~14474 with an unrealized loss of
   ~$1,233. Because `availableToWithdraw` is $0, Bybit rejects any new order (including
   reduce-only close orders) with `110007`. Options:
   - Top up the demo/testnet account with more USDT, or
   - Manually close the ETH short via the Bybit testnet UI/app, or
   - Create a fresh testnet API key pair with a clean demo balance.

2. **After the account is healthy, run the smoke test first**:
   ```bash
   uv run python scripts/run_testnet_30min_btc_eth.py --duration 120 --symbols BTC-USDT-PERP
   ```
   Verify that:
   - At least one order is submitted and accepted by Bybit.
   - `live_fills` count is > 0 in the SQLite store.
   - The tracked local position matches the broker position (no reconciler drift).
   - The position is fully flattened at shutdown.

3. **Then run the full 30-minute BTC+ETH retest**:
   ```bash
   uv run python scripts/run_testnet_30min_btc_eth.py
   ```

4. **Post-session**: inspect `reports/testnet_30min_btc_eth_YYYYMMDD_HHMMSS.md` and the
   per-symbol databases. If `live_fills` is still 0 or drift alerts appear, the fix is
   incomplete.

## Remaining residual risk

- Testnet liquidity is thin; large market orders can slip or fail. The new chunked
  close + limit fallback reduces but does not eliminate this risk.
- The strategy gates should improve signal quality, but 30 minutes is a very short
  sample. A longer session or multiple sessions are needed to judge profitability.
- The current `ComprehensiveStrategy` is entirely technical-indicator based. Future
  upgrades could add:
  - trained model predictions,
  - orderbook imbalance / funding-rate filters,
  - dynamic take-profit based on realized volatility.
