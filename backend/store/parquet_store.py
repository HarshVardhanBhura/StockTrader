"""Bar storage: Parquet on disk, queried through DuckDB.

Backtests are analytical scans over columns, not transactional lookups, so
Parquet + DuckDB beats a row store here and needs no server. Postgres is
reserved for run / trade / decision state.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from ..config import settings


class BarStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or settings.parquet_dir) / "bars"
        self.root.mkdir(parents=True, exist_ok=True)

    def _year_path(self, year: int) -> Path:
        return self.root / f"year={year}" / "part.parquet"

    def write(self, df: pd.DataFrame) -> int:
        """Upsert bars, partitioned by year. Dedupes on (date, symbol, series)."""
        if df.empty:
            return 0
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        total = 0
        for year, chunk in df.groupby(df["date"].dt.year):
            path = self._year_path(int(year))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                chunk = pd.concat([pd.read_parquet(path), chunk], ignore_index=True)
            chunk = (
                chunk.drop_duplicates(subset=["date", "symbol", "series"], keep="last")
                .sort_values(["date", "symbol"])
                .reset_index(drop=True)
            )
            chunk.to_parquet(path, index=False)
            total += len(chunk)
        return total

    @property
    def glob(self) -> str:
        return str(self.root / "**" / "*.parquet").replace("\\", "/")

    def is_empty(self) -> bool:
        return not any(self.root.rglob("*.parquet"))

    def query(self, sql: str) -> pd.DataFrame:
        if self.is_empty():
            return pd.DataFrame()
        con = duckdb.connect()
        try:
            source = f"read_parquet('{self.glob}')"
            return con.execute(sql.replace("{bars}", source)).df()
        finally:
            con.close()

    def read(
        self,
        symbols: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        series: str | None = "EQ",
    ) -> pd.DataFrame:
        where = []
        if symbols:
            joined = ",".join(f"'{s}'" for s in symbols)
            where.append(f"symbol IN ({joined})")
        if start:
            where.append(f"date >= DATE '{start:%Y-%m-%d}'")
        if end:
            where.append(f"date <= DATE '{end:%Y-%m-%d}'")
        if series:
            where.append(f"series = '{series}'")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        df = self.query(f"SELECT * FROM {{bars}} {clause} ORDER BY symbol, date")
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def date_range(self) -> tuple[object, object]:
        df = self.query("SELECT min(date) AS lo, max(date) AS hi FROM {bars}")
        if df.empty or pd.isna(df.loc[0, "lo"]):
            return None, None
        return df.loc[0, "lo"], df.loc[0, "hi"]

    def symbols_on(self, d: date) -> list[str]:
        """Point-in-time universe: what actually traded that day."""
        df = self.query(
            "SELECT DISTINCT symbol FROM {bars} "
            f"WHERE date = DATE '{d:%Y-%m-%d}' AND series = 'EQ' ORDER BY symbol"
        )
        return [] if df.empty else df["symbol"].tolist()
