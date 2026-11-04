# Data provenance

`raw/markets.csv` — 3,330 rows, of which 3,329 are usable, across 897
resolved binary Polymarket markets, 2024-01-10 to 2026-08-04. The other row
is priced at exactly 1.0, so cleaning drops it as an already-settled market.

## Where it came from

This is a **derived sample published by another open-source project**:

- Source: https://github.com/SLaffey24/prediction-market-calibration
- Path in that repo: `data/sample/markets.csv`
- Retrieved: 2026-11-04

I did not collect it. That repository obtained it from Polymarket's public
APIs (Gamma for the market list, CLOB `prices-history` for price paths) and
committed a reduced version containing only the fields needed to reproduce
a calibration analysis offline.

The analysis in this repository — every metric, every interval, every
model — is my own implementation. The data collection is not. Stating this
plainly matters more than the few lines it costs.

## Schema

| column | meaning |
| --- | --- |
| `market_id` | Polymarket market identifier; repeats across horizons |
| `platform` | always `polymarket` in this sample |
| `category` | **always `unknown`** in this derived sample — unusable |
| `price` | mid price observed `days_to_resolution` days before close |
| `outcome` | 1 if the contract resolved YES, else 0 |
| `days_to_resolution` | observation horizon: 1, 3, 7 or 30 |
| `close_time` | ISO-8601 UTC resolution timestamp |
| `volume` | final market volume, the same at every horizon |

One row is one (market, horizon) pair, so a market appears up to four times
with the same `outcome`. Those rows are not independent observations, which
is why every interval in this project clusters on `market_id`.

## Refreshing it

`make fetch` runs `scripts/fetch_polymarket.py`, which pulls directly from
the public APIs and writes the same schema with a real `category` column.
It needs network access and is deliberately rate-limited.
