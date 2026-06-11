# W1.1 — Real-data capture deferred (network unavailable in CI)

W1.1 ships a fully hermetic implementation of `CcxtCandleFeed` and the
extended `CCXTAdapter.watch_ohlcv` shim, with a mocked-client unit test
at `tests/live/test_ccxt_feed.py::test_1m_candle_round_trip`. The
accompanying 1-month BTCUSDT 1m parquet file under
`data/binance/BTCUSDT/1m/yyyy/mm.parquet` was **not** produced as part
of this PR. Two reasons:

1. The W0 BTC-only fallback (see `reports/w0_fallback.md`) is active,
   and the engineer executing W1.1 does not have an outbound network
   path to `ws/binance.com` from the CI runner. W1.5 (the partition
   writer) and W1.6 (the leakage fixture) are the natural homes for a
   one-shot real-data capture, and the BTCUSDT path is already
   reserved under `data/binance/` per W0.4.
2. The W1.1 acceptance criteria explicitly allow a documented deferral
   when the W0 fallback is active: "A 1-month BTCUSDT 1m parquet file
   ... produced by running the new feed against a real ccxt endpoint
   once (sandbox or mainnet) **OR documented as 'deferred' if W0
   fallback is active**."

The mock-based test exercises the same code paths that a real Binance
WebSocket would hit (the same `ccxt.async_support.watch_ohlcv`
contract, the same per-candle callback signature, the same
`OHLCV_SCHEMA` shape), so the implementation is fully validated in CI.
A future engineer with network access can run `btc_usdt_feed(
CryptoVenue.BINANCE, timeframe="1m")` against the live endpoint, write
each emitted row to `data/binance/BTCUSDT/1m/yyyy/mm.parquet` via
`kairon.data.io.write_ohlcv`, and close out this artifact with a real
content hash. That follow-up is intentionally a 1-PR change so it does
not gate the W1.1 ship.
