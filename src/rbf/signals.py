"""지표 계산 (설계 D11~D15). 모든 함수는 wide 프레임(행=날짜, 열=티커)을 받아 wide 를 돌려준다.

룩어헤드 규칙(D23): 모든 계산은 과거 방향 rolling 만 쓴다. `cut_after(frames, asof)` 로 asof 이후 행을 잘라낸 입력으로
계산해도 asof 행의 값이 같아야 하며, 테스트가 이를 강제한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rbf import config as C


def suspended_mask(open_: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """거래정지일: 시가 0 이고 거래량 0 (D4). 상장돼 있지 않은 날(NaN)은 False."""
    return (open_ == 0) & (volume == 0)


def cut_after(frames: dict[str, pd.DataFrame], asof: str | pd.Timestamp) -> dict[str, pd.DataFrame]:
    """asof 이후의 행을 잘라낸 사본(룩어헤드 테스트용)."""
    ts = pd.Timestamp(asof)
    return {k: v.loc[:ts] for k, v in frames.items()}


def rnb(net_val: pd.DataFrame, value: pd.DataFrame, suspended: pd.DataFrame,
        window: int = C.RNB_WINDOW, min_days: int = C.RNB_MIN_DAYS) -> pd.DataFrame:
    """순매수 강도 = Σ 순매수거래대금 / Σ 거래대금. 거래정지일은 분자·분모에서 빠진다. 분모 0 → 결측."""
    num = net_val.mask(suspended)
    den = value.mask(suspended)
    ns = num.rolling(window, min_periods=min_days).sum()
    ds = den.rolling(window, min_periods=min_days).sum()
    out = ns / ds
    return out.where(ds > 0)


def rsh(buy_val: pd.DataFrame, sell_val: pd.DataFrame, value: pd.DataFrame, suspended: pd.DataFrame,
        window: int = C.RSH_WINDOW, min_days: int = C.RSH_MIN_DAYS) -> pd.DataFrame:
    """개인 거래 비중 = Σ(개인 매수+매도) / Σ(2×거래대금)."""
    num = (buy_val + sell_val).mask(suspended)
    den = (2.0 * value).mask(suspended)
    ns = num.rolling(window, min_periods=min_days).sum()
    ds = den.rolling(window, min_periods=min_days).sum()
    out = ns / ds
    return out.where(ds > 0)


def cum_log_return(ret: pd.DataFrame) -> pd.DataFrame:
    """L_t = Σ_{s≤t} log(1+r_s). 상장 전·후(NaN)는 NaN 유지, 중간 NaN 은 0 으로 본다(스냅샷에 있으면 등락률이 있다)."""
    lr = np.log1p(ret)
    listed = ret.notna()
    L = lr.fillna(0.0).cumsum()
    return L.where(listed)


def window_return(ret: pd.DataFrame, window: int, min_days: int | None = None) -> pd.DataFrame:
    """Π_{t-window+1..t}(1+r) − 1."""
    lr = np.log1p(ret)
    s = lr.rolling(window, min_periods=min_days or window).sum()
    return np.expm1(s)


def max_return(ret: pd.DataFrame, window: int = C.MAX_WINDOW) -> pd.DataFrame:
    return ret.rolling(window, min_periods=window).max()


def momentum_12_1(ret: pd.DataFrame, long: int = C.MOM_LONG, skip: int = C.MOM_SKIP) -> pd.DataFrame:
    """Π_{t-long+1 .. t-skip}(1+r) − 1 (12-1 개월)."""
    lr = np.log1p(ret)
    s_long = lr.rolling(long, min_periods=long).sum()
    s_skip = lr.rolling(skip, min_periods=skip).sum()
    return np.expm1(s_long - s_skip)


def turnover(volume: pd.DataFrame, shares: pd.DataFrame, cap: float = C.TURNOVER_CAP) -> pd.DataFrame:
    v = volume / shares.where(shares > 0)
    return v.clip(upper=cap)


def turnover_mean(volume: pd.DataFrame, shares: pd.DataFrame, window: int = C.TURNOVER_CTRL_WINDOW) -> pd.DataFrame:
    return (volume / shares.where(shares > 0)).rolling(window, min_periods=window).mean()


def cgo(ret: pd.DataFrame, volume: pd.DataFrame, shares: pd.DataFrame,
        window: int = C.CGO_WINDOW, min_days: int = C.CGO_MIN_DAYS) -> pd.DataFrame:
    """Grinblatt–Han 자본이득 오버행 (D13).
    RP_t = Σ_{n=1..W} w_n P_{t-n} / Σ w_n,  w_n = V_{t-n} Π_{i=1}^{n-1}(1-V_{t-n+i}) = V_{t-n} exp(S_{t-1} - S_{t-n}),
    S = Σ log(1-V). CGO_t = (P_t - RP_t)/P_t. P = 수정 가격 지수 exp(L).
    """
    P = np.exp(cum_log_return(ret).to_numpy(dtype=float))          # NaN where not listed
    V = turnover(volume, shares).to_numpy(dtype=float)
    V = np.where(np.isnan(V), 0.0, V)
    S = np.cumsum(np.log1p(-V), axis=0)                             # S_t
    T, N = P.shape
    num = np.zeros((T, N))
    den = np.zeros((T, N))
    cnt = np.zeros((T, N), dtype=np.int32)
    S_prev = np.vstack([np.zeros((1, N)), S[:-1]])                  # S_{t-1}
    for n in range(1, window + 1):
        if n >= T:
            break
        w = V[:-n] * np.exp(S_prev[n:] - S[:-n])                    # 행 t 가 t-n 행을 본다
        p = P[:-n]
        valid = ~np.isnan(p)
        w = np.where(valid, w, 0.0)
        num[n:] += w * np.where(valid, p, 0.0)
        den[n:] += w
        cnt[n:] += valid.astype(np.int32)
    with np.errstate(invalid="ignore", divide="ignore"):
        rp = num / den
        out = (P - rp) / P
    out = np.where((cnt >= min_days) & (den > 0) & ~np.isnan(P), out, np.nan)
    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


def sentiment_regime(kospi_close: pd.Series, window: int = C.SENTIMENT_WINDOW) -> pd.Series:
    """KOSPI 126거래일 수익률 부호: 1=높음(양), 0=낮음(음 또는 0). 데이터 부족이면 NaN."""
    r = kospi_close / kospi_close.shift(window) - 1.0
    out = (r > 0).astype(float)
    return out.where(r.notna())


def fear_z(etf_indiv_net: pd.DataFrame, leverage: list[str] = C.LEVERAGE_ETFS, inverse: list[str] = C.INVERSE_ETFS,
           sum_window: int = C.FEAR_SUM_WINDOW, base_window: int = C.FEAR_BASE_WINDOW,
           base_min: int = C.FEAR_BASE_MIN_DAYS) -> pd.DataFrame:
    """공포 z (D15). 기준선은 t-1 까지의 과거 base_window 일."""
    inv = etf_indiv_net.reindex(columns=inverse).sum(axis=1, min_count=1)
    lev = etf_indiv_net.reindex(columns=leverage).sum(axis=1, min_count=1)
    F = (inv - lev).rolling(sum_window, min_periods=sum_window).sum()
    base = F.shift(1).rolling(base_window, min_periods=base_min)
    z = (F - base.mean()) / base.std()
    return pd.DataFrame({"F": F, "z": z})


def compute_all(frames: dict[str, pd.DataFrame], kospi_close: pd.Series, etf_indiv_net: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """브리프 3절 지표 전부. frames 키: open high low close volume value ret mktcap shares
    indiv_net_val indiv_buy_val indiv_sell_val foreign_net_val."""
    susp = suspended_mask(frames["open"], frames["volume"])
    out = {
        "rnb5": rnb(frames["indiv_net_val"], frames["value"], susp),
        "rsh20": rsh(frames["indiv_buy_val"], frames["indiv_sell_val"], frames["value"], susp),
        "ret20": window_return(frames["ret"], C.CRASH_WINDOW),
        "cgo260": cgo(frames["ret"], frames["volume"], frames["shares"]),
        "max20": max_return(frames["ret"]),
        "logmcap": np.log(frames["mktcap"].where(frames["mktcap"] > 0)),
        "mom12_1": momentum_12_1(frames["ret"]),
        "turn20": turnover_mean(frames["volume"], frames["shares"]),
        "frnb5": rnb(frames["foreign_net_val"], frames["value"], susp),
    }
    out["sentiment"] = sentiment_regime(kospi_close)
    out["fear"] = fear_z(etf_indiv_net)
    return out
