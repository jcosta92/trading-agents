"""Historical data container: wide DataFrames per field (open/close/volume/funding/...), keyed by instrument id."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class HistoryStore:
    def __init__(self, frames: dict[str, pd.DataFrame] | None = None, meta: dict | None = None):
        self.frames: dict[str, pd.DataFrame] = {}
        self.meta: dict = dict(meta or {})
        for k, v in (frames or {}).items():
            self.frames[k] = _clean(v)

    # ------------------------------------------------------------------ building
    def set_series(self, field: str, key: str, s: pd.Series) -> None:
        s = pd.Series(s).astype("float64")
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).tz_localize(None).normalize()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        df = self.frames.get(field)
        if df is None:
            self.frames[field] = s.to_frame(key)
        else:
            df = df.drop(columns=[key], errors="ignore")
            self.frames[field] = df.join(s.rename(key), how="outer").sort_index()

    def set_frame(self, field: str, df: pd.DataFrame) -> None:
        """Add many columns at once (much faster than repeated set_series)."""
        df = _clean(df)
        cur = self.frames.get(field)
        if cur is None:
            self.frames[field] = df.sort_index()
        else:
            cur = cur.drop(columns=[c for c in df.columns if c in cur.columns])
            self.frames[field] = pd.concat([cur, df], axis=1, join="outer").sort_index()

    # ------------------------------------------------------------------ access
    def has(self, key: str, field: str = "close") -> bool:
        df = self.frames.get(field)
        return df is not None and key in df.columns and df[key].notna().any()

    def series(self, field: str, key: str) -> pd.Series:
        df = self.frames.get(field)
        if df is None or key not in df.columns:
            return pd.Series(dtype="float64")
        return df[key].dropna()

    def close(self, key: str) -> pd.Series:
        return self.series("close", key)

    def frame(self, field: str) -> pd.DataFrame:
        return self.frames.get(field, pd.DataFrame())

    def keys(self, field: str = "close") -> list[str]:
        return list(self.frames.get(field, pd.DataFrame()).columns)

    def last_date(self) -> pd.Timestamp | None:
        df = self.frames.get("close")
        return None if df is None or df.empty else df.index.max()

    def truncate(self, end: pd.Timestamp) -> "HistoryStore":
        """New store with every field cut at `end` (inclusive)."""
        end = pd.Timestamp(end)
        return HistoryStore({k: v.loc[:end] for k, v in self.frames.items()}, self.meta)

    def merge(self, other: "HistoryStore") -> None:
        for field, df in other.frames.items():
            for col in df.columns:
                self.set_series(field, col, df[col].dropna())
        self.meta.update(other.meta)

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for field, df in self.frames.items():
            out = df.astype("float32") if field in ("volume",) else df
            out.to_parquet(d / f"{field}.parquet", compression="zstd")
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))

    @classmethod
    def load(cls, directory: str | Path) -> "HistoryStore":
        d = Path(directory)
        frames = {p.stem: pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))}
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        return cls(frames, meta)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.astype("float64")


def splice_with_proxy(primary: pd.Series, proxy: pd.Series) -> pd.Series:
    """Extend `primary` backwards using `proxy` returns, scaled so the two series meet on the
    primary's first date (or the last proxy date before it)."""
    primary = primary.dropna()
    proxy = proxy.dropna()
    if primary.empty:
        return proxy
    first = primary.index[0]
    older = proxy.loc[:first]
    if older.empty:
        return primary
    scale = primary.iloc[0] / older.iloc[-1]
    back = older[older.index < first] * scale
    out = pd.concat([back, primary])
    return out[~out.index.duplicated(keep="last")].sort_index()


def to_eur(series_usd: pd.Series, eurusd: pd.Series) -> pd.Series:
    """Convert a USD price series to EUR using EURUSD (USD per EUR), forward-filled."""
    fx = eurusd.reindex(series_usd.index.union(eurusd.index)).ffill().reindex(series_usd.index)
    return (series_usd / fx).dropna()


def annualised_vol(prices: pd.Series, window: int, periods_per_year: float) -> float:
    r = np.log(prices).diff().dropna().iloc[-window:]
    if len(r) < max(10, window // 2):
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))
