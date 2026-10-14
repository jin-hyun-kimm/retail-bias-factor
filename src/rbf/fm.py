"""횡단면 통계 (설계 D16·D19): 윈저라이즈·z-score, Fama–MacBeth(Newey–West), 검출 가능 최소 효과."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from rbf import config as C


def winsor_z(x: pd.Series, limits: tuple[float, float] = C.WINSOR) -> pd.Series:
    """한 리밸런싱일의 횡단면: 1%·99% 윈저라이즈 후 평균 0·표준편차 1."""
    v = x.astype(float)
    lo, hi = v.quantile(limits[0]), v.quantile(limits[1])
    v = v.clip(lower=lo, upper=hi)
    sd = v.std()
    if not np.isfinite(sd) or sd == 0:
        return v * np.nan
    return (v - v.mean()) / sd


def winsor_z_wide(df: pd.DataFrame, member: pd.DataFrame | None = None) -> pd.DataFrame:
    """행(날짜)마다 winsor_z. member 가 있으면 멤버 안에서만 계산하고 나머지는 NaN."""
    x = df.where(member) if member is not None else df
    return x.apply(winsor_z, axis=1)


def nw_mean_test(x: pd.Series, lags: int = C.NW_LAGS) -> dict:
    """시계열 평균의 Newey–West t 검정."""
    v = x.dropna().astype(float)
    if len(v) < 3:
        return {"mean": np.nan, "se": np.nan, "t": np.nan, "n": len(v)}
    X = np.ones((len(v), 1))
    res = sm.OLS(v.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"mean": float(res.params[0]), "se": float(res.bse[0]), "t": float(res.tvalues[0]), "n": int(len(v))}


def fama_macbeth(panel: pd.DataFrame, y: str, xcols: list[str], date_col: str = "date",
                 lags: int = C.NW_LAGS, min_n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """월별 횡단면 OLS(절편 포함) → 계수 시계열 → NW 평균 검정.
    반환: (summary 행=변수 열=coef se t n_months avg_n, coefs 행=날짜 열=변수)."""
    rows = {}
    ns = {}
    for d, g in panel.groupby(date_col):
        g = g[[y] + xcols].dropna()
        if len(g) < min_n:
            continue
        X = np.column_stack([np.ones(len(g)), g[xcols].to_numpy(dtype=float)])
        yy = g[y].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        rows[d] = beta
        ns[d] = len(g)
    coefs = pd.DataFrame(rows, index=["const"] + xcols).T.sort_index()
    out = {}
    for c in coefs.columns:
        r = nw_mean_test(coefs[c], lags)
        out[c] = {"coef": r["mean"], "se": r["se"], "t": r["t"], "n_months": r["n"], "avg_n": float(np.mean(list(ns.values()))) if ns else np.nan}
    return pd.DataFrame(out).T, coefs


def mde_monthly(series: pd.Series, t_target: float = C.T_PREREG) -> dict:
    """검출 가능 최소 효과: 월별 초과수익 표준편차와 개월 수로 t=t_target 에 필요한 월평균."""
    v = series.dropna()
    n = len(v)
    sd = float(v.std())
    return {"n_months": n, "sd_monthly": sd, "mde_monthly": t_target * sd / np.sqrt(n) if n else np.nan}
