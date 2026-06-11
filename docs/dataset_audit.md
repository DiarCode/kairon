# Dataset Audit — Kairon

**Date:** 2026-06-05
**Scope:** Each dataset listed in `datasets/DATASETS.md` and external candidates, scored for use in the Kairon system.

## Audit criteria (each scored 1-5)

| Criterion | Definition |
|-----------|------------|
| **Schema quality** | Clean columns, consistent types, documented |
| **Granularity** | Tick / 1m / 5m / 1h / 1d |
| **Coverage** | Time span, instrument count, regime diversity |
| **Missingness** | % of missing bars, gaps, handling |
| **Timezone** | Documented, normalized to UTC |
| **Survivorship** | Includes delisted assets, not just current |
| **Symbol normalization** | "BTCUSDT" / "BTC/USDT" / "BTC-USD" harmonized |
| **Leakage risk** | Can the data be joined to future info? |
| **Tradability realism** | Volume, spread, slippage information present? |
| **Merge compatibility** | Joins cleanly with news / on-chain / macro |
| **License** | Free / paid / research-only / commercial |

## 1. Crypto OHLCV

| # | Dataset | Schema | Granularity | Coverage | License | Verdict |
|---|---------|--------|-------------|----------|---------|---------|
| 1 | **Binance Full History (tick)** | Trade-level (ts, price, qty, side) | tick | 2021-, BTC + alts | Free for personal | **Include (HFT/microstructure)** |
| 2 | **Bitcoin Historical 2018-2026 (Binance)** | OHLCV | daily + intraday | 2018-2026, BTC | Kaggle, research-only OK | **Include (baseline)** |
| 3 | **BTCUSD 1-Minute (7 Exchanges)** | OHLCV | 1m | 2017- | Kaggle | **Include (cross-venue)** |
| 4 | **Integrated Crypto Data (80+ Coins)** | OHLCV + market cap | daily | 2012-2021 | Mendeley, research-only | **Include (multi-asset)** |
| 5 | **CryptoDataDownload OHLCV** | OHLCV | 1m/5m/1h/1d | 2015- | Free | **Include (regular updates)** |
| 6 | **Bitcoin Full-History Network (Graph)** | Edge list, GraphML | transaction | 2009-2024 | arXiv | **Defer (v2, GNN)** |
| 7 | **TWRR** | OHLCV + corp actions | daily | 1950-2024, 10k+ | Kaggle, research-only | **Include (stocks historical bulk)** |
| 8 | **Bitcoin Arbitrage** | price diff + timing | mixed | varies | PMC | **Defer (specialized)** |
| 9 | **Crypto Twitter/Reddit Sentiment** | ts + text + score | per-post | 2019-2024 | Kaggle | **Include (sentiment baseline)** |
| 10 | **CoinGecko API** | OHLCV | daily | 2014- | Free 2y | **Include (broad market, sanity)** |

## 2. Stocks

| Source | Granularity | Coverage | Cost | Verdict |
|--------|-------------|----------|------|---------|
| **TWRR Kaggle** | daily | 1950-2024, 10k+ | free | **Primary bulk historical** |
| **Tiingo EOD** | daily | 20y+ | $10/mo | **Primary live EOD** |
| **Polygon.io** | tick, 1m, 1h, 1d | 15y+ | $29+/mo | **Primary intraday + live** |
| **Alpha Vantage** | 1m, 5m, 1h, 1d | 20y+ | free tier | **Include (free dev)** |
| **yfinance** | 1m, 5m, 15m, 1h, 1d | varies | free | **Dev only** |

## 3. On-chain

| Source | Coverage | Cost | Verdict |
|--------|----------|------|---------|
| **Glassnode Studio** | MVRV, NUPL, SOPR, exchange flows | free API key, paid advanced | **Primary** |
| **CryptoQuant** | similar + wallet clusters | paid Pro for advanced | **Alternative** |
| **Bitquery** | multi-chain traces | free tier | **Include (traces)** |
| **Dune** | SQL on public chains | free + paid | **Include (bespoke)** |

## 4. News / sentiment

| Source | Granularity | Cost | Verdict |
|--------|-------------|------|---------|
| **CryptoPanic** | per-news + sentiment tag | free tier | **Primary crypto news** |
| **Tiingo News** | per-news + tickers | included w/ Tiingo | **Primary stock news** |
| **GDELT** | 15-min batch, 100+ languages | free | **Include (global signal)** |
| **FinBERT (ProsusAI)** | 3-class sentiment | free | **Primary sentiment model** |
| **FinGPT v3.x** | 3-class + custom | free weights | **Alternative / upgrade** |

## 5. Macro

| Source | Verdict |
|--------|---------|
| **FRED** | **Primary** |
| **World Bank** | Include |
| **ECB/BoE/IMF** | Include (FX/rates) |

## 6. Known leakage / survivorship risks

| Risk | Mitigation |
|------|------------|
| Survivorship bias in 10k+ asset lists | TWRR includes delisted; we still audit |
| Random shuffle on time series | **Forbidden** in code; CI lint rule |
| Train/test contamination | Walk-forward + purging + embargo |
| Preprocessing over full data (normalization, scaling) | Use only training window for scaler stats |
| Off-by-one in label construction | Strict unit tests on `t` and `t+H` boundaries |
| Using future bar's high/low in label | Always use `close[t+H]` for label, never high/low |
| On-chain data revisions (Glassnode/CryptoQuant) | Pin dataset version, log hash |

## 7. Tradability realism

For backtests to be honest, we need:
- Commission per trade (exchange-specific, e.g., Binance 0.1% spot default)
- Bid/ask spread (snapshot at decision time, or modeled)
- Market impact (square-root model: `impact = sigma * sqrt(order_size / adv)`)
- Funding rate (perpetuals)
- Borrow fee (shorts)
- Slippage (configurable, default 1 tick)

`backtesting.py` and `vectorbt` both accept commission and slippage parameters. We standardize on a `CostModel` pydantic schema and apply it uniformly.

## 8. Symbol normalization

We adopt a single internal symbol convention:
- Crypto spot: `BASE-QUOTE` (e.g., `BTC-USDT`)
- Crypto perp: `BASE-QUOTE-PERP` (e.g., `BTC-USDT-PERP`)
- Stocks: `TICKER` (e.g., `AAPL`); ETF: `TICKER` (e.g., `SPY`)
- Indices: `INDEX-NAME` (e.g., `SP500`, `VIX`)

All adapters must convert to/from this canonical form; the typed schema lives in `src/kairon/data/symbols.py`.

## 9. Final ranked recommendation

| Rank | Dataset | Use |
|------|---------|-----|
| 1 | TWRR (stocks, 1950-2024) | baseline historical bulk |
| 2 | CryptoDataDownload (multi-timeframe) | baseline crypto bulk |
| 3 | CCXT Binance live (REST + WS) | live crypto |
| 4 | Polygon.io live (US equities) | live stocks |
| 5 | Tiingo EOD + News | live stocks + news |
| 6 | Glassnode Studio | on-chain crypto |
| 7 | FinBERT (ProsusAI) | sentiment |
| 8 | CryptoPanic + GDELT | news |
| 9 | FRED | macro |
| 10 | Bitcoin Historical 2018-2026 (Kaggle) | cross-check |
| 11 | BTCUSD 1-Min (7 exchanges) | cross-venue |
| 12 | Integrated Crypto Data (80+ coins) | altcoin research |
| 13 | CoinGecko API | sanity / coverage |
| 14 | Crypto Twitter/Reddit Sentiment (Kaggle) | sentiment baseline |
| 15 | Bitquery / Dune | bespoke on-chain |
| 16 | Bitcoin Full-History Network Graph | v2 (GNN) |
| 17 | Bitcoin Arbitrage | v2 |
