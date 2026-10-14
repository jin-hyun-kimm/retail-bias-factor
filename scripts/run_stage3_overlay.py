"""3단계 H5 공포 오버레이 (브리프 4-8, 설계 D15·D22).

1) 공포 z(일별) → z≥2 사건(20거래일 안 재발은 같은 사건) 수를 먼저 센다.
2) 사건 연구: 사건일 종가 → 이후 20·60거래일 KOSPI 수익률 표. 비교: 모든 날의 같은 창 수익률 분포.
3) 사건 10건 이상이면 오버레이: 합산 점수(s1+s2+s3) Q5 롱온리(비용 0.15%) 의 주식 비중 m = clip(1+0.25z, 0.5, 1.5).
   10건 미만이면 '통계력 부족' 이라 적고 사건 연구만 보고한다.
결과: results/stage3/overlay_*.csv, overlay_summary.md
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rbf import analysis as A  # noqa: E402
from rbf import config as C  # noqa: E402
from rbf import portfolio as P  # noqa: E402
from rbf.io_raw import load_kospi, trading_days  # noqa: E402
from rbf.signals import fear_z  # noqa: E402
from run_stage1 import git_hash, md_table  # noqa: E402

OUT = C.RESULTS / "stage3"


def events_from_z(z: pd.Series, thr: float = C.FEAR_EVENT_Z, cooldown: int = C.FEAR_EVENT_COOLDOWN) -> list[pd.Timestamp]:
    ev = []
    last = -10**9
    idx = z.index
    for i, (d, v) in enumerate(z.items()):
        if v >= thr and i - last > cooldown:
            ev.append(d)
            last = i
    return ev


def event_study(kospi: pd.DataFrame, events: list[pd.Timestamp], horizons=(20, 60)) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = kospi["close"]
    rows = []
    for d in events:
        i = close.index.get_loc(d)
        r = {"event": d.date(), "kospi_close": close.iloc[i]}
        for h in horizons:
            r[f"ret_{h}d"] = close.iloc[i + h] / close.iloc[i] - 1 if i + h < len(close) else np.nan
        rows.append(r)
    ev = pd.DataFrame(rows)
    base = pd.DataFrame({f"ret_{h}d": close.shift(-h) / close - 1 for h in horizons})
    summ = pd.DataFrame({
        "event_mean": ev[[f"ret_{h}d" for h in horizons]].mean(), "event_median": ev[[f"ret_{h}d" for h in horizons]].median(),
        "event_hit": (ev[[f"ret_{h}d" for h in horizons]] > 0).mean(), "event_n": ev[[f"ret_{h}d" for h in horizons]].count(),
        "all_days_mean": base.mean(), "all_days_sd": base.std(),
    })
    summ["t_vs_all_days"] = (summ["event_mean"] - summ["all_days_mean"]) / (summ["all_days_sd"] / np.sqrt(summ["event_n"]))
    return ev, summ


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    days = trading_days(C.LOOKBACK_START, C.EVAL_END)
    kospi = load_kospi().reindex(days)
    etf = pd.read_parquet(C.PROCESSED / "etf_indiv_net.parquet").reindex(days)
    fz = fear_z(etf)
    fz.to_csv(OUT / "overlay_fear_z_daily.csv", encoding="utf-8-sig")
    z = fz["z"].dropna()
    z = z[z.index >= pd.Timestamp(C.EVAL_START)]
    hold_start = pd.Timestamp(C.HOLDOUT_START)
    events = events_from_z(z)
    n_ex = sum(1 for d in events if d < hold_start)
    n_ho = len(events) - n_ex
    ev, summ = event_study(kospi, events)
    ev.to_csv(OUT / "overlay_events.csv", index=False, encoding="utf-8-sig")
    summ.to_csv(OUT / "overlay_event_study.csv", encoding="utf-8-sig")
    L = ["# 3단계 H5 공포 오버레이", "", f"생성 {datetime.now().isoformat(timespec='seconds')}, 커밋 {git_hash()[:10]}", "",
         f"z 정의: (인버스 {C.INVERSE_ETFS} − 레버리지 {C.LEVERAGE_ETFS}) 개인 순매수 거래대금 20일 합을 과거 250일로 표준화. z 가 있는 첫 날 {z.index[0].date()}.", "",
         f"## z ≥ 2 사건 수: 전체 {len(events)} (탐색 {n_ex}, 홀드아웃 {n_ho}; 20거래일 안 재발은 같은 사건)", ""]
    enough = len(events) >= C.FEAR_MIN_EVENTS
    if not enough:
        L += [f"**통계력 부족**: 사건이 {C.FEAR_MIN_EVENTS}건 미만이라 오버레이 성과는 보고하지 않고 사건 연구만 적는다.", ""]
    L += ["## 사건 연구: 사건일 종가 → 이후 KOSPI 수익률", "", md_table(summ, 4), "", "### 사건 목록", "", md_table(ev, 4, index=False), ""]
    if enough:
        admin = A.load_admin_events()
        ctx = A.prepare(C.LOOKBACK_START, C.EVAL_START, C.EVAL_END, admin)
        tickers = ctx.member.columns
        sig = ctx.panel.pivot(index="date", columns="ticker", values="score").reindex(columns=tickers)
        ew, _, _, qmap = P.quantile_returns(sig, ctx.R, ctx.member, None)
        W = P.long_only_weights(qmap, C.N_QUANTILES)
        costs = P.turnover_and_costs(W, ctx.R)
        port = (ew["Q5"] - costs["cost_base"]).rename("factor_net")
        zt = fz["z"].reindex(port.index)
        m = (1 + C.OVERLAY_SLOPE * zt).clip(C.OVERLAY_MIN, C.OVERLAY_MAX).fillna(1.0)
        ov = pd.DataFrame({"factor_net": port, "z": zt, "m": m, "overlay": m * port}).join(ctx.bench[["kospi", "univ_ew"]])
        ov.to_csv(OUT / "overlay_monthly.csv", encoding="utf-8-sig")
        rows = []
        for period, mask in [("explore", ov.index < hold_start - pd.Timedelta(10, unit="D")), ("holdout", ov.index >= hold_start - pd.Timedelta(10, unit="D")), ("all", ov.index == ov.index)]:
            sub = ov[mask]
            for col in ["factor_net", "overlay"]:
                st = P.perf_stats(sub[col], sub["kospi"])
                rows.append({"period": period, "portfolio": col, **{k: st[k] for k in ["n_months", "mean_m", "ann_ret", "ann_vol", "sharpe", "mdd", "excess_m", "excess_t"]},
                             "avg_m": sub["m"].mean()})
            d = (sub["overlay"] - sub["factor_net"])
            t = P.nw_mean_test(d)
            rows.append({"period": period, "portfolio": "overlay − factor", "n_months": t["n"], "mean_m": t["mean"], "excess_t": t["t"]})
        st = pd.DataFrame(rows)
        st.to_csv(OUT / "overlay_stats.csv", index=False, encoding="utf-8-sig")
        L += ["## 오버레이 (합산 점수 Q5 롱온리, 비용 0.15%; m = clip(1+0.25z, 0.5, 1.5); 현금 0%, 1 초과분은 무이자 차입 가정)", "",
              md_table(st, 4, index=False), ""]
    (OUT / "overlay_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)[:2500])


if __name__ == "__main__":
    main()
