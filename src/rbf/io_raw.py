"""원천 CSV(data/raw) → 연도별 wide parquet(data/processed/{year}/{var}.parquet, 행=날짜, 열=티커).

변수: open high low close volume value ret mktcap shares market(1=KOSPI,2=KOSDAQ)
      indiv_buy_val indiv_sell_val indiv_net_val foreign_net_val inst_net_val
그 밖에: names.csv(티커→종목명), etf_indiv_net.parquet(날짜×ETF), kospi.parquet(지수 OHLCV).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rbf.config import PROCESSED, RAW

PRICE_VARS = ["open", "high", "low", "close", "volume", "value", "ret", "mktcap", "market"]
CAP_VARS = ["shares"]
INV_VARS = ["indiv_buy_val", "indiv_sell_val", "indiv_net_val", "foreign_net_val", "inst_net_val"]
ALL_VARS = PRICE_VARS + CAP_VARS + INV_VARS
_OHLCV_COLS = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume",
               "거래대금": "value", "등락률": "ret", "시가총액": "mktcap"}


def trading_days(start: str | None = None, end: str | None = None) -> pd.DatetimeIndex:
    files = sorted((RAW / "index").glob("kospi_1001_*.csv"))
    idx = pd.concat([pd.read_csv(f, index_col=0, parse_dates=True, encoding="utf-8-sig") for f in files])
    days = idx.index.sort_values().unique()
    if start:
        days = days[days >= pd.Timestamp(start)]
    if end:
        days = days[days <= pd.Timestamp(end)]
    return pd.DatetimeIndex(days)


def load_kospi() -> pd.DataFrame:
    files = sorted((RAW / "index").glob("kospi_1001_*.csv"))
    df = pd.concat([pd.read_csv(f, index_col=0, parse_dates=True, encoding="utf-8-sig") for f in files]).sort_index()
    df = df.rename(columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume", "거래대금": "value"})
    return df[~df.index.duplicated()]


def _read_snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={0: str})
    df = df.rename(columns={df.columns[0]: "ticker"})
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    return df


def _year_complete(year: int, days: pd.DatetimeIndex) -> bool:
    ydays = days[days.year == year]
    for d in ydays:
        s = d.strftime("%Y%m%d")
        need = [RAW / "ohlcv" / str(year) / f"{s}_KOSPI.csv", RAW / "ohlcv" / str(year) / f"{s}_KOSDAQ.csv",
                RAW / "cap" / str(year) / f"{s}_ALL.csv"] + [RAW / "netbuy" / str(year) / f"{s}_ALL_{k}.csv" for k in ("indiv", "foreign", "inst")]
        if not all(p.exists() for p in need):
            return False
    return len(ydays) > 0


def build_year(year: int, days: pd.DatetimeIndex, force: bool = False) -> None:
    out = PROCESSED / str(year)
    if out.exists() and (out / "ret.parquet").exists() and not force:
        return
    if not _year_complete(year, days):
        print(f"[skip] {year}: 원천 파일이 아직 다 없다", file=sys.stderr)
        return
    ydays = days[days.year == year]
    longs: list[pd.DataFrame] = []
    for d in ydays:
        s = d.strftime("%Y%m%d")
        parts = []
        for mkt, code in (("KOSPI", 1), ("KOSDAQ", 2)):
            o = _read_snapshot(RAW / "ohlcv" / str(year) / f"{s}_{mkt}.csv").rename(columns=_OHLCV_COLS)
            o["market"] = code
            parts.append(o[["ticker", "market"] + [c for c in _OHLCV_COLS.values() if c in o.columns]])
        o = pd.concat(parts, ignore_index=True).drop_duplicates("ticker")
        o["ret"] = o["ret"] / 100.0
        cap = _read_snapshot(RAW / "cap" / str(year) / f"{s}_ALL.csv")[["ticker", "상장주식수"]].rename(columns={"상장주식수": "shares"})
        o = o.merge(cap, on="ticker", how="left")
        for key, pre in (("indiv", "indiv"), ("foreign", "foreign"), ("inst", "inst")):
            nb = _read_snapshot(RAW / "netbuy" / str(year) / f"{s}_ALL_{key}.csv")
            cols = {"순매수거래대금": f"{pre}_net_val"}
            if key == "indiv":
                cols.update({"매수거래대금": "indiv_buy_val", "매도거래대금": "indiv_sell_val"})
            nb = nb[["ticker"] + list(cols)].rename(columns=cols)
            o = o.merge(nb, on="ticker", how="left")
        # D4: 거래가 있었는데 투자자 기록이 없으면 0, 거래정지(거래량 0·시가 0)면 결측
        suspended = (o["volume"] == 0) & (o["open"] == 0)
        for c in INV_VARS:
            o.loc[~suspended & o[c].isna(), c] = 0.0
            o.loc[suspended, c] = np.nan
        o["date"] = d
        longs.append(o)
    long = pd.concat(longs, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    for var in ALL_VARS:
        wide = long.pivot(index="date", columns="ticker", values=var).sort_index()
        wide.columns.name = None
        wide.to_parquet(out / f"{var}.parquet")
    print(f"[built] {year}: days={len(ydays)} tickers={long['ticker'].nunique()}")


def build_names() -> pd.Series:
    """티커→종목명(마지막으로 본 이름). 개인 순매수 파일의 종목명 열에서."""
    names: dict[str, str] = {}
    for f in sorted((RAW / "netbuy").glob("*/*_ALL_indiv.csv")):
        nb = pd.read_csv(f, usecols=[0, 1], dtype=str, encoding="utf-8-sig")
        for t, n in zip(nb.iloc[:, 0].str.zfill(6), nb.iloc[:, 1]):
            names[t] = n
    s = pd.Series(names, name="name").sort_index()
    s.index.name = "ticker"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    s.to_csv(PROCESSED / "names.csv", encoding="utf-8-sig")
    return s


def build_etf() -> pd.DataFrame:
    frames = {}
    for f in sorted((RAW / "etf").glob("*.csv")):
        tk = f.name.split("_")[0]
        df = pd.read_csv(f, index_col=0, parse_dates=True, encoding="utf-8-sig")
        if len(df) == 0:
            continue
        frames.setdefault(tk, []).append(df["개인"])
    wide = pd.DataFrame({tk: pd.concat(v).sort_index() for tk, v in frames.items()})
    wide = wide[~wide.index.duplicated()].sort_index()
    wide.to_parquet(PROCESSED / "etf_indiv_net.parquet")
    return wide


def build_all(force: bool = False) -> None:
    days = trading_days()
    for y in sorted(set(days.year)):
        build_year(int(y), days, force=force)
    build_names()
    build_etf()
    load_kospi().to_parquet(PROCESSED / "kospi.parquet")


def load_wide(var: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """연도별 parquet 을 이어 붙여 하나의 wide 프레임으로. 열(티커)은 합집합."""
    years = sorted(int(p.name) for p in PROCESSED.iterdir() if p.is_dir() and p.name.isdigit())
    if start:
        years = [y for y in years if y >= pd.Timestamp(start).year]
    if end:
        years = [y for y in years if y <= pd.Timestamp(end).year]
    frames = [pd.read_parquet(PROCESSED / str(y) / f"{var}.parquet") for y in years]
    df = pd.concat(frames).sort_index()
    if start:
        df = df.loc[pd.Timestamp(start):]
    if end:
        df = df.loc[:pd.Timestamp(end)]
    return df.reindex(columns=sorted(df.columns))


def load_names() -> pd.Series:
    return pd.read_csv(PROCESSED / "names.csv", dtype=str, encoding="utf-8-sig").set_index("ticker")["name"]


if __name__ == "__main__":
    build_all(force="--force" in sys.argv)
