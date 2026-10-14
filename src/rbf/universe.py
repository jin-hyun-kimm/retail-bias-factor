"""유니버스 (설계 D8~D10). 리밸런싱일마다 제외 단계별 건수를 남긴다."""
from __future__ import annotations

import pandas as pd

from rbf import config as C
from rbf.returns import next_day

STEPS = ["listed", "common", "not_spac", "not_admin", "not_suspended", "tradable_t1", "size", "liquidity"]


def admin_intervals(events: pd.DataFrame | None, last_seen: pd.Series) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """[20013] 이벤트 → 종목별 관리종목 구간 [지정일, 해제일 전날].
    첫 이벤트가 해제·사유변경이면 그 전부터 관리종목이었던 것으로 본다(D10). 해제 없이 끝나면 마지막 관측일까지."""
    out: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    if events is None or len(events) == 0:
        return out
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"].astype(str).str.replace("/", ""), format="%Y%m%d")
    for tk, g in ev.sort_values("date").groupby("ticker"):
        start: pd.Timestamp | None = None
        ivs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for _, r in g.iterrows():
            kind = str(r["change_type"])
            if "해제" in kind:
                s = start if start is not None else pd.Timestamp("1990-01-01")
                ivs.append((s, r["date"] - pd.Timedelta(1, unit="D")))
                start = None
            elif "지정" in kind:
                if start is None:
                    start = r["date"]
            else:  # 사유변경 등: 관리종목 상태의 증거
                if start is None:
                    start = pd.Timestamp("1990-01-01")
        if start is not None:
            end = last_seen.get(tk, pd.Timestamp("2099-12-31"))
            ivs.append((start, pd.Timestamp(end)))
        out[tk] = ivs
    return out


def admin_flag(intervals: dict, tickers: pd.Index, dates: pd.DatetimeIndex) -> pd.DataFrame:
    flag = pd.DataFrame(False, index=dates, columns=tickers)
    for tk, ivs in intervals.items():
        if tk not in flag.columns:
            continue
        for s, e in ivs:
            flag.loc[s:e, tk] = True
    return flag


def is_common(tickers: pd.Index) -> pd.Series:
    return pd.Series([str(t)[-1] == "0" for t in tickers], index=tickers)


def is_spac(names: pd.Series, tickers: pd.Index) -> pd.Series:
    n = names.reindex(tickers).fillna("")
    return n.str.contains("스팩")


def one_price_day(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return (open_ > 0) & (open_ == high) & (high == low) & (low == close)


def build_universe(frames: dict[str, pd.DataFrame], names: pd.Series, admin: pd.DataFrame | None,
                   days: pd.DatetimeIndex, rebal: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """반환: (member 행=리밸런싱일 열=티커 bool, counts 행=리밸런싱일 열=단계별 남은 수)."""
    tickers = frames["close"].columns
    common = is_common(tickers)
    spac = is_spac(names, tickers)
    members, counts = [], []
    val = frames["value"]
    month_value = val.groupby([val.index.year, val.index.month]).transform("mean")
    for t in rebal:
        t1 = next_day(days, t)
        if t1 is None:
            break
        close_t = frames["close"].loc[t]
        m = close_t.notna()
        row = {"listed": int(m.sum())}
        m &= common
        row["common"] = int(m.sum())
        m &= ~spac
        row["not_spac"] = int(m.sum())
        if admin is not None and t in admin.index:
            m &= ~admin.loc[t].reindex(tickers).fillna(False).astype(bool)
        row["not_admin"] = int(m.sum())
        susp_t = (frames["open"].loc[t] == 0) & (frames["volume"].loc[t] == 0)
        m &= ~susp_t
        row["not_suspended"] = int(m.sum())
        o1, h1, l1, c1, v1 = (frames[k].loc[t1] for k in ("open", "high", "low", "close", "volume"))
        susp_t1 = (o1 == 0) & (v1 == 0)
        locked = one_price_day(o1, h1, l1, c1)
        m &= ~susp_t1 & ~locked & o1.notna()
        row["tradable_t1"] = int(m.sum())
        base = m.copy()
        mc = frames["mktcap"].loc[t].where(base)
        size_cut = mc.quantile(C.SIZE_BOTTOM_PCT)
        mv = month_value.loc[t].where(base)
        liq_cut = mv.quantile(C.VALUE_BOTTOM_PCT)
        m &= mc > size_cut
        row["size"] = int(m.sum())
        m &= mv > liq_cut
        row["liquidity"] = int(m.sum())
        members.append(m.rename(t))
        counts.append(pd.Series(row, name=t))
    return pd.DataFrame(members), pd.DataFrame(counts)
