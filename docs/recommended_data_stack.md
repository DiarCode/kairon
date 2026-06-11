# Recommended Data Stack — Kairon

**Date:** 2026-06-05
**Purpose:** Concrete data layer for the system, by use case.

## 1. Baseline modeling (research + backtest)

| Source | Role | Why |
|--------|------|-----|
| **TWRR (Kaggle)** | 10k+ assets, 1950-2024 | Bulk historical, regime coverage including 2008, dot-com |
| **CryptoDataDownload** | 50+ coins, 1m/5m/1h/1d | Clean, free, regular updates, multiple timeframes |
| **Bitcoin Historical 2018-2026 (Kaggle)** | BTC cross-check | Long history for benchmark |

## 2. High-frequency experiments

| Source | Role |
|--------|------|
| **Binance tick data** (`data.binance.vision`) | Tick / aggTrades for HFT research |
| **BTCUSD 1-Min (7 exchanges)** (Kaggle) | Cross-venue microstructure |
| **CCXT `watchOrderBook`** | Live L2 order book |

## 3. Multimodal prediction (text + on-chain + TA)

| Source | Role |
|--------|------|
| **FinBERT (ProsusAI)** | News / social sentiment (3-class) |
| **FinGPT v3.3** | Custom-domain sentiment fine-tune (4-class, more recent data) |
| **CryptoPanic + Tiingo News + GDELT** | News streams |
| **Crypto Twitter/Reddit Sentiment (Kaggle)** | Historical sentiment baseline |
| **Glassnode Studio** | MVRV, NUPL, SOPR, exchange flows, dormancy |

## 4. Macro overlay

| Source | Role |
|--------|------|
| **FRED** | US macro (CPI, rates, M2, etc.) |
| **World Bank** | Global macro |
| **ECB / BoE** | FX & rates |

## 5. Live inference

| Source | Role |
|--------|------|
| **CCXT Binance/Bybit/Coinbase** | Live crypto OHLCV + L2 order book |
| **Polygon.io** | Live US equities (real-time WebSocket) |
| **Tiingo EOD** | Live US EOD |
| **Glassnode** | Live on-chain |
| **Ollama cloud (`gpt-oss:120b-cloud`)** | LLM reasoning layer |

## 6. Ingestion cadence

| Stream | Cadence | Storage |
|--------|---------|---------|
| OHLCV (daily) | 1× per day, 30 min after close | parquet (year-partitioned) |
| OHLCV (intraday) | 1× per bar close | parquet (date-partitioned) |
| News | continuous, batched every 5 min | parquet + duckdb |
| On-chain | 1× per hour | parquet |
| Macro | 1× per day after FRED release | parquet |
| L2 order book | continuous, downsampled to 1s | parquet (rotating buffer) |

## 7. Storage layout

```
data/
  raw/
    ohlcv/
      crypto/{venue}/{symbol}/{timeframe}/{YYYY}/{MM}.parquet
      stocks/{provider}/{symbol}/{timeframe}/{YYYY}/{MM}.parquet
    onchain/{provider}/{metric}/{YYYY}.parquet
    news/{provider}/{YYYY}/{MM}/{DD}.parquet
    macro/{provider}/{series}.parquet
  processed/
    features/{symbol}/{timeframe}/{YYYY}.parquet
    labels/{symbol}/{horizon}/{YYYY}.parquet
  models/
    {name}/{version}/model.pt
    {name}/{version}/meta.json
```

All parquet files written by `pyarrow` with explicit schemas; all metadata logged via `mlflow`.

## 8. Data quality controls

- CI runs `kairon.data.diagnostics` on every parquet:
  - monotonic timestamps
  - no duplicate (ts, symbol) pairs
  - no negative prices / volumes
  - timezone uniform (UTC)
  - expected bar counts vs calendar
- All findings written to `data_quality_report.html` and surfaced in the UI.
- "Last successful ingestion" timestamp on every dataset, with SLA alerts.
