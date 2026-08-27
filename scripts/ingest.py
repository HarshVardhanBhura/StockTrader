"""Ingest NSE bhavcopy history into the Parquet bar store.

    python scripts/ingest.py --start 2024-06-01 --end 2024-07-31

Spans the July 2024 UDiFF cutover transparently: dates before it use the
legacy parser, dates after use UDiFF. Holidays and weekends are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data.nse_bhavcopy import (  # noqa: E402
    NoDataForDate, fetch_day, make_session, trading_days, url_for,
)
from backend.store.parquet_store import BarStore  # noqa: E402


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=parse_date, required=True)
    ap.add_argument("--end", type=parse_date, required=True)
    ap.add_argument("--sleep", type=float, default=0.4,
                    help="pause between requests; be polite to NSE")
    args = ap.parse_args()

    store = BarStore()
    session = make_session()
    days = trading_days(args.start, args.end)
    print(f"{len(days)} candidate trading days from {args.start} to {args.end}")

    ok = skipped = 0
    rows = 0
    for d in days:
        _, fmt = url_for(d)
        try:
            df = fetch_day(d, session)
        except NoDataForDate as exc:
            skipped += 1
            print(f"  skip {d} ({fmt}): {exc}")
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"  ERROR {d} ({fmt}): {type(exc).__name__}: {exc}")
        else:
            store.write(df)
            ok += 1
            rows += len(df)
            print(f"  {d} [{fmt}] {len(df):>5} rows")
        time.sleep(args.sleep)

    lo, hi = store.date_range()
    print(f"\ningested {ok} days ({rows} rows), skipped {skipped}")
    print(f"store now covers {lo} .. {hi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
