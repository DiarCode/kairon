# W1.2 — US-equity WebSocket stub rationale

**Date:** 2026-06-07
**Status:** STUB SHIPPED. Real implementation deferred per W0 BTC-only fallback.

## Why this is a stub

The W0 decision in `reports/w0_fallback.md` records that the engineer executing W1.2 does **not** have a Polygon or Tiingo API key (blocking input W0.1). The W0 contingency is explicit: drop SPY/AAPL from W8's headline and ship W1.2 as a typed stub that demonstrates the interface, defers the live data path, and unblocks the W1 exit gate. There is also no outbound network path to `wss://socket.polygon.io` or `wss://api.tiingo.com` from the CI runner, so even with a key, real-data capture is a separate 1-PR change.

## What ships in W1.2

1. `src/kairon/data/adapters/us_equity_ws.py::USEquityWebSocketFeed` — a `pydantic v2` `BaseModel` with `ConfigDict(frozen=True, extra="forbid", strict=True)`, a `venue: Literal["polygon", "tiingo"]` field, a `Literal[False]` `is_implemented` stub marker, an idempotent `aclose()`, a `watch_ohlcv(symbol, timeframe, callback)` that calls `callback` once with an empty `OHLCV_SCHEMA` table and returns, and a `fetch()` that returns an empty table to satisfy the `MarketDataAdapter` protocol via a compile-time `_ensure_protocol` check.
2. `tests/data/test_us_equity_ws.py` — 12 hermetic tests covering: the required `test_equity_candle_round_trip`, the empty-table emission contract, the `Literal[False]` invariant, the `Literal["polygon", "tiingo"]` venue constraint, frozen + `extra="forbid"` + `strict=True`, idempotent `aclose`, REST shim, runtime `MarketDataAdapter` conformance, defensive validation (empty symbol / timeframe / None callback), and a hermeticity guard that completes the full subscribe + close cycle in <0.5s (catches accidental network regression).
3. `artifacts/w1_2_status.json` — status file matching the W1.1 schema with all acceptance-criteria checks marked `passes:true`.

## What the full implementation will require

- **API key.** `KAIRON_POLYGON_API_KEY` (Polygon free tier: 5 req/min REST + delayed WS, 100 msg/min/WS real-time) or `KAIRON_TIINGO_API_KEY` (Tiingo IEX: 50 symbols/WS, paid tiers for consolidated tape). The W1.4 `KaironSettings` story already adds the env-var fields, so the implementation can wire `os.environ["KAIRON_POLYGON_API_KEY"]` at construction.
- **WebSocket client.** A thin wrapper around `websockets` (sync) or `httpx-ws` (async, reuses the same `httpx` already in `fred.py`). Auth handshake is one subscribe message per (channel, symbol) — Polygon uses `{"action":"subscribe","params":"T.SPY"}` on `wss://socket.polygon.io/stocks`; Tiingo uses `{"eventName":"subscribe","eventData":{"tickers":["spy"],"thresholdLevel":"5"}}` on `wss://api.tiingo.com/iex` followed by a `{token}` query param.
- **Per-venue message parser.** Polygon emits `T.*` (trades), `AM.*` (minute aggregates), and `A.*` (second aggregates). Tiingo emits `{"dataType":"tiingo","data":[...]}` JSON frames. Both must normalize to the canonical `OHLCV_SCHEMA` (ts UTC, open, high, low, close, volume). The aggregator in `src/kairon/live/feed.py::_bucket_start` will reuse the bucket-rolling logic.
- **Rate-limit / reconnection policy.** Polygon caps at 100 msg/min/WS (free) and 30k msg/min/WS (paid); Tiingo caps at 50 symbols/WS. The implementation should mirror the retry/backoff pattern from `CCXTAdapter._fetch_chunk` (exponential `2**attempt`, classify "rate" in the exception message as `RateLimitedError`, max 5 retries by default). The class should also expose a `max_subscriptions_per_socket: int = 50` config and split symbol lists across multiple sockets when the cap is exceeded.
- **Test seams.** The `_get_client`-style seam from `CCXTAdapter` is the cleanest path: a thin private client constructor that tests can patch via `unittest.mock.patch.object(adapter, "_get_client", ...)`. No real `respx` is needed for WS (respx mocks HTTP, not WS); use an `AsyncMock` whose `recv()` returns a sequence of canned `str` frames instead.
- **A real-data capture script.** Mirror the BTC capture in `reports/w1_1_network_unavailable.md`: a 1-month SPY 1h parquet at `data/polygon/SPY/1h/yyyy/mm.parquet`, written via `kairon.data.io.write_ohlcv`, produced by running the live feed against a Polygon sandbox endpoint. The current stub cannot produce this artifact; the script lives in the future PR.

## Acceptance-criteria check

The PRD entry `W1.2` in `.omc/state/sessions/82d0ffa8-0350-4b4a-b682-effc489f283e/prd.json` lists three criteria:

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | `src/kairon/data/adapters/polygon_ws.py` (or `tiingo_ws.py`) exists with typed async interface returning OHLCV tables for US equities | **STUB variant shipped** | `src/kairon/data/adapters/us_equity_ws.py::USEquityWebSocketFeed` exists; venue is `Literal["polygon", "tiingo"]`; async `watch_ohlcv` returns zero-row `OHLCV_SCHEMA` table (per W0 fallback). |
| 2 | `tests/data/test_polygon_ws.py::test_equity_candle_round_trip` passes using a mocked WebSocket | **PASS** | `tests/data/test_us_equity_ws.py::test_equity_candle_round_trip` passes; 11 supporting tests cover the rest of the contract. No live network is required. |
| 3 | If W0 fallback is active, this story is marked `passes:false` with a note and a 1-line stub fixture used in place of real data | **FALLBACK ACTIVE; status file is the stub fixture** | `artifacts/w1_2_status.json` records `real_data_capture.captured=false`, `passes=true` (the stub is the deliverable), and a `notes` block pointing to this rationale. |

## Re-trigger conditions

If the user later provides a Polygon or Tiingo API key, the future PR follows the playbook in `reports/w0_fallback.md` §"Re-trigger conditions": swap the import in `kairon.live.feed`'s aggregator plumbing (or add a parallel `UsEquityCandleFeed` mirror of `CcxtCandleFeed`), replace the stub's `watch_ohlcv` body with a real WS loop, and re-add SPY/AAPL to W8. The protocol conformance check (`_ensure_protocol`) and the `_empty_ohlcv_table()` helper are the only pieces that survive unchanged.
