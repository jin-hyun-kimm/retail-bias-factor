"""D23 ①: 신호 함수는 asof 이후 데이터를 잘라낸 입력으로 계산해도 asof 행의 값이 같아야 한다."""
import numpy as np
import pandas as pd
import pytest

from rbf import signals as S


def make_frames(T=400, N=6, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-01-01", periods=T)
    tickers = [f"{i:05d}0" for i in range(1, N + 1)]
    ret = pd.DataFrame(rng.normal(0, 0.02, (T, N)), index=days, columns=tickers)
    close = 10000 * np.exp(np.log1p(ret).cumsum())
    open_ = close * (1 + rng.normal(0, 0.005, (T, N)))
    volume = pd.DataFrame(rng.integers(1000, 100000, (T, N)).astype(float), index=days, columns=tickers)
    value = volume * close
    shares = pd.DataFrame(5_000_000.0, index=days, columns=tickers)
    indiv_net = pd.DataFrame(rng.normal(0, 1e8, (T, N)), index=days, columns=tickers)
    buy = value * rng.uniform(0.3, 0.7, (T, N))
    sell = value * rng.uniform(0.3, 0.7, (T, N))
    foreign_net = pd.DataFrame(rng.normal(0, 1e8, (T, N)), index=days, columns=tickers)
    # 거래정지 하루, 상장 전 구간 하나
    open_.iloc[50, 0] = 0
    volume.iloc[50, 0] = 0
    ret.iloc[:100, 1] = np.nan
    close.iloc[:100, 1] = np.nan
    return {"open": open_, "high": close, "low": close, "close": close, "volume": volume, "value": value, "ret": ret,
            "mktcap": close * shares, "shares": shares, "indiv_net_val": indiv_net, "indiv_buy_val": buy,
            "indiv_sell_val": sell, "foreign_net_val": foreign_net}, days


@pytest.mark.parametrize("asof_pos", [300, 350, 399])
def test_signals_do_not_use_future(asof_pos):
    frames, days = make_frames()
    kospi = pd.Series(np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, len(days)))), index=days)
    etf = pd.DataFrame(np.random.default_rng(2).normal(0, 1e9, (len(days), 4)), index=days, columns=["122630", "123320", "114800", "123310"])
    asof = days[asof_pos]
    full = S.compute_all(frames, kospi, etf)
    cut = S.compute_all(S.cut_after(frames, asof), kospi.loc[:asof], etf.loc[:asof])
    for k in ["rnb5", "rsh20", "ret20", "cgo260", "max20", "logmcap", "mom12_1", "turn20", "frnb5"]:
        a, b = full[k].loc[asof], cut[k].loc[asof]
        assert np.allclose(a.fillna(-999), b.fillna(-999), atol=1e-10), k
    fs, cs = full["sentiment"].loc[asof], cut["sentiment"].loc[asof]
    assert (np.isnan(fs) and np.isnan(cs)) or fs == cs
    fz, cz = full["fear"].loc[asof, "z"], cut["fear"].loc[asof, "z"]
    assert (np.isnan(fz) and np.isnan(cz)) or abs(fz - cz) < 1e-10


def test_cgo_matches_slow_formula():
    frames, days = make_frames(T=320, N=2)
    out = S.cgo(frames["ret"], frames["volume"], frames["shares"])
    tk = frames["ret"].columns[0]
    r = frames["ret"][tk]
    P = np.exp(np.log1p(r).fillna(0).cumsum())
    V = (frames["volume"][tk] / frames["shares"][tk]).clip(upper=0.999)
    t = len(P) - 1
    num = den = 0.0
    for n in range(1, 261):
        w = V.iloc[t - n] * np.prod(1 - V.iloc[t - n + 1:t].values)
        num += w * P.iloc[t - n]
        den += w
    expect = (P.iloc[t] - num / den) / P.iloc[t]
    assert abs(out[tk].iloc[-1] - expect) < 1e-9
