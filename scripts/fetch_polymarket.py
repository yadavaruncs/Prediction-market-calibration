#!/usr/bin/env python3
"""Pull resolved binary markets from Polymarket's public APIs.

The committed sample in data/raw/markets.csv was taken from another
project's published extract (see data/README.md). This script exists so the
dataset can be rebuilt from source -- and, unlike that extract, it keeps the
`category` field, which is what the headline analysis is currently missing.

    python scripts/fetch_polymarket.py --out data/raw/markets.csv --limit 2000

Two endpoints are involved:

  Gamma   https://gamma-api.polymarket.com/markets
          market metadata, including resolution outcome and close time.

  CLOB    https://clob.polymarket.com/prices-history
          historical price series per token, which is where the price at
          N days before resolution comes from.

Be polite to both. They are free and unauthenticated, and they do not owe
anyone a scrape.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import timezone
from pathlib import Path

import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"

HORIZONS = (1, 3, 7, 30)
USER_AGENT = "pmcal-research/0.1 (academic calibration study)"
SLEEP_SECONDS = 0.25


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


def fetch_markets(session: requests.Session, limit: int, page_size: int = 100,
                  cache_dir: Path | None = None) -> list[dict]:
    """Page through resolved markets, newest first."""
    out: list[dict] = []
    offset = 0
    while len(out) < limit:
        params = {
            "closed": "true",
            "limit": min(page_size, limit - len(out)),
            "offset": offset,
            "order": "endDate",
            "ascending": "false",
        }
        r = session.get(GAMMA, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        offset += len(batch)
        print(f"  fetched {len(out)} markets", flush=True)
        time.sleep(SLEEP_SECONDS)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "gamma_markets.json").write_text(json.dumps(out))
    return out


def _is_binary_resolved(m: dict) -> bool:
    """Keep only two-outcome markets with an unambiguous settled result."""
    try:
        outcomes = json.loads(m.get("outcomes") or "[]")
        prices = json.loads(m.get("outcomePrices") or "[]")
    except (json.JSONDecodeError, TypeError):
        return False
    if len(outcomes) != 2 or len(prices) != 2:
        return False
    if not m.get("endDate") or not m.get("clobTokenIds"):
        return False
    # A settled binary market prints 1/0. Anything else is still live,
    # disputed, or void -- none of which belong in a calibration study.
    try:
        vals = sorted(float(p) for p in prices)
    except (TypeError, ValueError):
        return False
    return vals == [0.0, 1.0]


def fetch_price_history(session: requests.Session, token_id: str,
                        start_ts: int, end_ts: int) -> pd.DataFrame:
    params = {"market": token_id, "startTs": start_ts, "endTs": end_ts,
              "fidelity": 60}
    r = session.get(CLOB_HISTORY, params=params, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame(columns=["t", "p"])
    hist = r.json().get("history", [])
    time.sleep(SLEEP_SECONDS)
    return pd.DataFrame(hist)


def build_rows(session: requests.Session, markets: list[dict]) -> pd.DataFrame:
    rows = []
    kept = 0
    for i, m in enumerate(markets, 1):
        if not _is_binary_resolved(m):
            continue
        try:
            token_ids = json.loads(m["clobTokenIds"])
            prices = json.loads(m["outcomePrices"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if not token_ids:
            continue

        yes_token = token_ids[0]
        outcome = int(float(prices[0]) == 1.0)
        close = pd.to_datetime(m["endDate"], utc=True, errors="coerce")
        if pd.isna(close):
            continue

        end_ts = int(close.replace(tzinfo=timezone.utc).timestamp())
        start_ts = end_ts - 40 * 86_400
        hist = fetch_price_history(session, yes_token, start_ts, end_ts)
        if hist.empty:
            continue
        hist["ts"] = pd.to_datetime(hist["t"], unit="s", utc=True)

        for h in HORIZONS:
            target = close - pd.Timedelta(days=h)
            before = hist[hist["ts"] <= target]
            if before.empty:
                continue
            # The last quote at or before the target time -- never a later
            # one, which would be lookahead.
            px = float(before.iloc[-1]["p"])
            if not (0.0 < px < 1.0):
                continue
            rows.append({
                "market_id": m.get("id"),
                "platform": "polymarket",
                "category": m.get("category") or "unknown",
                "price": px,
                "outcome": outcome,
                "days_to_resolution": h,
                "close_time": close.isoformat(),
                "volume": float(m.get("volumeNum") or m.get("volume") or 0.0),
            })
        kept += 1
        if i % 25 == 0:
            print(f"  processed {i}/{len(markets)} markets, kept {kept}", flush=True)

    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/markets.csv")
    ap.add_argument("--limit", type=int, default=1500,
                    help="how many closed markets to consider")
    ap.add_argument("--cache", default="data/cache")
    args = ap.parse_args(argv)

    session = _session()
    print(f"fetching up to {args.limit} closed markets from Gamma ...")
    markets = fetch_markets(session, args.limit, cache_dir=Path(args.cache))
    print(f"got {len(markets)}; pulling price histories ...")

    df = build_rows(session, markets)
    if df.empty:
        print("no usable rows -- check the API response shape")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {len(df)} rows across {df.market_id.nunique()} markets "
          f"to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
