"""보유 수익률 (설계 D6·D7): t+1 시가 → t'+1 시가. 상장폐지는 마지막 종가로 청산, t'+1 시가가 없으면 t' 종가로 청산."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rbf.signals import cum_log_return


def month_end_dates(days: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    """매월 마지막 거래일 (start~end 안)."""
    d = pd.Series(days, index=days)
    me = d.groupby([days.year, days.month]).max()
    me = pd.DatetimeIndex(me.values)
    return me[(me >= pd.Timestamp(start)) & (me <= pd.Timestamp(end))]


def next_day(days: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp | None:
    i = days.get_loc(t)
    return days[i + 1] if i + 1 < len(days) else None


def holding_returns(open_: pd.DataFrame, close: pd.DataFrame, ret: pd.DataFrame,
                    days: pd.DatetimeIndex, rebal: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """반환: (R, exit_type). R 행=리밸런싱일 t, 열=티커.
    exit_type: 'open'(t'+1 시가), 'close'(t' 종가), 'delist'(마지막 종가), 'none'(진입 불가)."""
    L = cum_log_return(ret)
    Lff = L.ffill()
    rows, kinds = [], []
    for i, t in enumerate(rebal[:-1]):
        t1 = next_day(days, t)
        tp = rebal[i + 1]
        tp1 = next_day(days, tp)
        if t1 is None or tp1 is None:
            break
        o1, c1 = open_.loc[t1], close.loc[t1]
        entry_ok = (o1 > 0) & (c1 > 0) & L.loc[t1].notna()
        entry_log = np.log((c1 / o1).where(entry_ok)) - L.loc[t1]
        o2, c2 = open_.loc[tp1], close.loc[tp1]
        exit_open = (o2 > 0) & (c2 > 0) & L.loc[tp1].notna()
        listed_tp = close.loc[tp].notna() & (close.loc[tp] > 0)
        exit_log = pd.Series(np.nan, index=open_.columns)
        exit_log[exit_open] = (L.loc[tp1] + np.log((o2 / c2).where(exit_open)))[exit_open]
        m_close = ~exit_open & listed_tp
        exit_log[m_close] = L.loc[tp][m_close]
        m_delist = ~exit_open & ~listed_tp
        exit_log[m_delist] = Lff.loc[tp][m_delist]
        R = np.expm1(entry_log + exit_log).where(entry_ok)
        kind = pd.Series("open", index=open_.columns)
        kind[m_close] = "close"
        kind[m_delist] = "delist"
        kind[~entry_ok] = "none"
        rows.append(R.rename(t))
        kinds.append(kind.rename(t))
    return pd.DataFrame(rows), pd.DataFrame(kinds)


def index_holding_returns(idx: pd.DataFrame, days: pd.DatetimeIndex, rebal: pd.DatetimeIndex) -> pd.Series:
    """KOSPI 지수의 같은 창 수익률: open(t+1) → open(t'+1)."""
    out = {}
    for i, t in enumerate(rebal[:-1]):
        t1 = next_day(days, t)
        tp1 = next_day(days, rebal[i + 1])
        if t1 is None or tp1 is None:
            break
        out[t] = idx.loc[tp1, "open"] / idx.loc[t1, "open"] - 1.0
    return pd.Series(out, name="kospi")
