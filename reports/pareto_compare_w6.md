# W6.3 — Stacked meta CAS-dominance vs TopK

**Story:** W6.3  
**Decided at:** 2026-06-08T09:12:54Z  
**W6 FALLBACK active:** False  
**Stacked CAS dominates primary (any asset, p<0.1):** False  
**Paired t-test p-value:** 0.396226  
**N dominating assets:** 2 / 3

## Per-asset CAS comparison

| Asset | Primary CAS | Stacked CAS | Stacked > Primary |
| --- | --- | --- | --- |
| BTCUSDT | -40.989178 | -40.420430 | yes |
| ETHUSDT | -33.616364 | -33.426659 | yes |
| SOLUSDT | -24.048470 | -24.167661 | no |

## Decision

The stacked meta CAS-dominates the primary CAS on 2 / 3 assets (p=0.3962). The W6.2 stacked meta is **shipped** per the W6.3 acceptance criterion.
