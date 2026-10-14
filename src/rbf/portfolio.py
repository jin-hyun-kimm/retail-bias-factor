"""포트폴리오 검정 (설계 D18·D20): 5분위, 동일가중·시총가중, 롱온리 − 벤치마크, 회전율·비용."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rbf import config as C
from rbf.fm import nw_mean_test


def assign_quantiles(signal_row: pd.Series, n: int = C.N_QUANTILES) -> pd.Series:
    """한 리밸런싱일: 신호 순위로 1..n 분위(n 이 가설 방향 상위). 결측은 NaN."""
    v = signal_row.dropna()
    if len(v) < n * 2:
        return pd.Series(np.nan, index=signal_row.index)
    ranks = v.rank(method="first")
    q = np.ceil(ranks / len(v) * n).clip(1, n)
    return q.reindex(signal_row.index)


def quantile_returns(signal: pd.DataFrame, R: pd.DataFrame, member: pd.DataFrame,
                     mktcap: pd.DataFrame | None = None, n: int = C.N_QUANTILES):
    """반환: (ew 행=리밸런싱일 열=Q1..Qn, vw 같은 모양, counts 열=Q1..Qn·n_signal_missing·n_universe, qmap 행=날짜 열=티커 분위)."""
    ew_rows, vw_rows, cnt_rows, qmaps = [], [], [], []
    for t in signal.index:
        if t not in R.index or t not in member.index:
            continue
        r = R.loc[t]
        ok = member.loc[t] & r.notna()
        s = signal.loc[t].where(ok)
        q = assign_quantiles(s, n)
        qmaps.append(q.rename(t))
        ew, vw, cnt = {}, {}, {}
        for k in range(1, n + 1):
            idx = q.index[q == k]
            ew[f"Q{k}"] = r[idx].mean() if len(idx) else np.nan
            cnt[f"Q{k}"] = len(idx)
            if mktcap is not None and len(idx):
                w = mktcap.loc[t, idx].astype(float)
                w = w / w.sum()
                vw[f"Q{k}"] = float((w * r[idx]).sum())
        cnt["n_signal_missing"] = int((ok & s.isna()).sum())
        cnt["n_universe"] = int(ok.sum())
        ew_rows.append(pd.Series(ew, name=t))
        vw_rows.append(pd.Series(vw, name=t))
        cnt_rows.append(pd.Series(cnt, name=t))
    return pd.DataFrame(ew_rows), pd.DataFrame(vw_rows), pd.DataFrame(cnt_rows), pd.DataFrame(qmaps)


def equal_weight_universe(R: pd.DataFrame, member: pd.DataFrame) -> pd.Series:
    out = {}
    for t in member.index:
        if t in R.index:
            out[t] = R.loc[t].where(member.loc[t]).mean()
    return pd.Series(out, name="univ_ew")


def long_only_weights(qmap: pd.DataFrame, k: int, mktcap: pd.DataFrame | None = None) -> pd.DataFrame:
    """Q_k 롱온리 비중 (행=리밸런싱일, 열=티커). 동일가중 또는 시총가중."""
    W = pd.DataFrame(0.0, index=qmap.index, columns=qmap.columns)
    for t in qmap.index:
        idx = qmap.columns[qmap.loc[t] == k]
        if len(idx) == 0:
            continue
        if mktcap is None:
            W.loc[t, idx] = 1.0 / len(idx)
        else:
            w = mktcap.loc[t, idx].astype(float)
            W.loc[t, idx] = (w / w.sum()).values
    return W


def tax_rate(t: pd.Timestamp) -> float:
    rate = C.TAX_SCHEDULE[0][1]
    for d, r in C.TAX_SCHEDULE:
        if pd.Timestamp(d) <= t:
            rate = r
    return rate


def turnover_and_costs(W: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """월별 회전율(Σ|w_t − w_drift|/2)과 비용 시나리오(D20). 열: turnover, cost_base, cost_low, cost_high, cost_real."""
    rows = []
    prev_w = None
    prev_t = None
    for t in W.index:
        w = W.loc[t]
        if prev_w is None:
            drift = pd.Series(0.0, index=W.columns)
        else:
            g = (1.0 + R.loc[prev_t].reindex(W.columns).fillna(0.0)) * prev_w
            drift = g / g.sum() if g.sum() > 0 else prev_w
        dw = w - drift
        buys = float(dw.clip(lower=0).sum())
        sells = float((-dw).clip(lower=0).sum())
        tax = tax_rate(t)
        rows.append(pd.Series({
            "turnover": 0.5 * float(dw.abs().sum()),
            "cost_base": (buys + sells) * C.COST_BASE,
            "cost_low": (buys + sells) * C.COST_SENS[0],
            "cost_high": (buys + sells) * C.COST_SENS[1],
            "cost_real": sells * (tax + C.COMMISSION_SLIPPAGE) + buys * C.COMMISSION_SLIPPAGE,
        }, name=t))
        prev_w, prev_t = w, t
    return pd.DataFrame(rows)


def portfolio_return(W: pd.DataFrame, R: pd.DataFrame) -> pd.Series:
    out = {}
    for t in W.index:
        r = R.loc[t].reindex(W.columns).fillna(0.0)
        out[t] = float((W.loc[t] * r).sum())
    return pd.Series(out)


def perf_stats(r: pd.Series, bench: pd.Series | None = None) -> dict:
    """월별 수익률 시계열 요약. 초과수익은 NW t 로."""
    r = r.dropna()
    n = len(r)
    out = {"n_months": n, "mean_m": r.mean(), "sd_m": r.std(),
           "ann_ret": (1 + r).prod() ** (12 / n) - 1 if n else np.nan,
           "ann_vol": r.std() * np.sqrt(12), "hit": (r > 0).mean()}
    out["sharpe"] = out["ann_ret"] / out["ann_vol"] if out["ann_vol"] else np.nan
    cum = (1 + r).cumprod()
    out["mdd"] = float((cum / cum.cummax() - 1).min())
    if bench is not None:
        ex = (r - bench.reindex(r.index)).dropna()
        tt = nw_mean_test(ex)
        out.update({"excess_m": tt["mean"], "excess_t": tt["t"], "excess_ann": tt["mean"] * 12})
    return out
