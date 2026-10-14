"""D23 ④: 분모 0·결측 처리, 유니버스 제외 규칙, 보유 수익률 청산 규칙 (합성 데이터)."""
import numpy as np
import pandas as pd

from rbf import signals as S
from rbf import universe as U
from rbf.returns import holding_returns, month_end_dates


def _days(n=60):
    return pd.bdate_range("2020-01-01", periods=n)


def test_rnb_zero_denominator_is_missing():
    d = _days(10)
    net = pd.DataFrame({"A": 1.0, "B": 1.0}, index=d)
    val = pd.DataFrame({"A": 1.0, "B": 0.0}, index=d)
    susp = pd.DataFrame(False, index=d, columns=["A", "B"])
    out = S.rnb(net, val, susp)
    assert out["A"].iloc[-1] == 1.0
    assert np.isnan(out["B"].iloc[-1])


def test_rnb_excludes_suspended_days_and_requires_min_days():
    d = _days(10)
    net = pd.DataFrame({"A": [1.0] * 10}, index=d)
    val = pd.DataFrame({"A": [2.0] * 10}, index=d)
    susp = pd.DataFrame({"A": [False] * 10}, index=d)
    susp.iloc[-1, 0] = True
    out = S.rnb(net, val, susp)
    assert abs(out["A"].iloc[-1] - 0.5) < 1e-12          # 정지일 제외해도 비율은 같다
    susp.iloc[-3:, 0] = True                              # 유효일 2 < 3
    out = S.rnb(net, val, susp)
    assert np.isnan(out["A"].iloc[-1])


def test_rsh_denominator_is_twice_value():
    d = _days(25)
    buy = pd.DataFrame({"A": 3.0}, index=d)
    sell = pd.DataFrame({"A": 1.0}, index=d)
    val = pd.DataFrame({"A": 4.0}, index=d)
    susp = pd.DataFrame({"A": False}, index=d)
    out = S.rsh(buy, sell, val, susp)
    assert abs(out["A"].iloc[-1] - 0.5) < 1e-12


def test_universe_rules():
    d = _days(45)
    tickers = ["000010", "000015", "000020", "000030", "000040", "000050", "000060", "000070", "000080", "000090", "000100", "000110"]
    N = len(tickers)
    base = pd.DataFrame(100.0, index=d, columns=tickers)
    frames = {k: base.copy() for k in ["open", "close"]}
    frames["high"] = base * 1.01
    frames["low"] = base * 0.99
    frames["volume"] = pd.DataFrame(1000.0, index=d, columns=tickers)
    frames["value"] = pd.DataFrame([[1000.0 * (i + 1)] * len(d) for i in range(N)], index=tickers, columns=d).T
    frames["mktcap"] = pd.DataFrame([[1e9 * (i + 1)] * len(d) for i in range(N)], index=tickers, columns=d).T
    names = pd.Series({"000020": "어느스팩"})
    rebal = month_end_dates(d, "2020-01-01", "2020-02-29")
    t = rebal[0]
    t1 = d[d.get_loc(t) + 1]
    frames["open"].loc[t, "000030"] = 0
    frames["volume"].loc[t, "000030"] = 0                                                 # t 거래정지
    for k in ("open", "high", "low", "close"):
        frames[k].loc[t1, "000040"] = 130.0                                               # t+1 한 가격(상한가 잠김)
    admin = pd.DataFrame(False, index=d, columns=tickers)
    admin.loc[:, "000050"] = True                                                         # 관리종목
    member, counts = U.build_universe(frames, names, admin, d, rebal)
    m = member.loc[t]
    assert not m["000015"], "우선주(끝자리≠0) 제외"
    assert not m["000020"], "스팩 제외"
    assert not m["000030"], "t 거래정지 제외"
    assert not m["000040"], "t+1 한 가격 제외"
    assert not m["000050"], "관리종목 제외"
    assert m["000110"]
    c = counts.loc[t]
    assert c["listed"] == N and c["common"] == N - 1 and c["not_spac"] == N - 2 and c["not_admin"] == N - 3
    assert c["not_suspended"] == N - 4 and c["tradable_t1"] == N - 5
    assert c["size"] < c["tradable_t1"] and c["liquidity"] <= c["size"]


def test_admin_intervals_first_event_release_means_designated_before():
    ev = pd.DataFrame({"ticker": ["X", "X", "Y"], "date": ["20150301", "20160101", "20170101"],
                       "change_type": ["관리종목해제", "관리종목지정", "관리종목지정"], "reason": ["", "", ""]})
    last = pd.Series({"X": pd.Timestamp("2018-12-31"), "Y": pd.Timestamp("2017-06-30")})
    iv = U.admin_intervals(ev, last)
    assert iv["X"][0] == (pd.Timestamp("1990-01-01"), pd.Timestamp("2015-02-28"))
    assert iv["X"][1] == (pd.Timestamp("2016-01-01"), pd.Timestamp("2018-12-31"))
    assert iv["Y"] == [(pd.Timestamp("2017-01-01"), pd.Timestamp("2017-06-30"))]


def test_holding_return_open_to_open_and_delisting():
    d = _days(50)
    rebal = month_end_dates(d, "2020-01-01", "2020-03-31")
    tickers = ["A", "B"]
    close = pd.DataFrame(100.0, index=d, columns=tickers)
    ret = pd.DataFrame(0.0, index=d, columns=tickers)
    t, tp = rebal[0], rebal[1]
    t1, tp1 = d[d.get_loc(t) + 1], d[d.get_loc(tp) + 1]
    # A: t+1 시가 100 → 종가 110(+10%), 이후 보합, t'+1 시가 121 (종가 대비 +10%)
    close.loc[t1:, "A"] = 110.0
    ret.loc[t1, "A"] = 0.10
    open_ = close.copy()
    open_.loc[t1, "A"] = 100.0
    open_.loc[tp1, "A"] = 121.0
    # B: t'+1 이전에 상장폐지 (마지막 종가 50)
    mid = d[d.get_loc(t) + 10]
    close.loc[mid:, "B"] = np.nan
    ret.loc[mid:, "B"] = np.nan
    close.loc[d[d.get_loc(mid) - 1], "B"] = 50.0
    ret.loc[d[d.get_loc(mid) - 1], "B"] = -0.5
    open_.loc[mid:, "B"] = np.nan
    R, kind = holding_returns(open_, close, ret, d, rebal)
    assert abs(R.loc[t, "A"] - 0.21) < 1e-9
    assert kind.loc[t, "A"] == "open"
    assert abs(R.loc[t, "B"] - (-0.5)) < 1e-9
    assert kind.loc[t, "B"] == "delist"
