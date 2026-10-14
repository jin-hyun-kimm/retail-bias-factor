"""가설 검정 파이프라인 (설계 D16~D20). scripts/run_stage1.py 와 run_holdout.py 가 쓴다.

흐름: 원천 parquet → 신호(일별) → 리밸런싱일 패널 → 유니버스 → 보유 수익률 → 가설별 5분위·Fama–MacBeth·소구간·플라시보·비용.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rbf import config as C
from rbf import portfolio as P
from rbf import signals as S
from rbf.fm import fama_macbeth, mde_monthly, winsor_z
from rbf.io_raw import load_kospi, load_names, load_wide, trading_days
from rbf.returns import holding_returns, index_holding_returns, month_end_dates
from rbf.universe import admin_flag, admin_intervals, build_universe

FRAME_KEYS = ["open", "high", "low", "close", "volume", "value", "ret", "mktcap", "shares",
              "indiv_net_val", "indiv_buy_val", "indiv_sell_val", "foreign_net_val", "market"]
CONTROLS = ["logmcap", "mom12_1", "ret20", "turn20", "frnb5"]
SIGNAL_KEYS = ["rnb5", "rsh20", "ret20", "cgo260", "max20", "logmcap", "mom12_1", "turn20", "frnb5"]


@dataclass
class Context:
    rebal: pd.DatetimeIndex
    member: pd.DataFrame
    exclusions: pd.DataFrame
    R: pd.DataFrame
    exit_kind: pd.DataFrame
    sig: dict[str, pd.DataFrame]
    mktcap: pd.DataFrame
    market: pd.DataFrame
    sentiment: pd.Series
    panel: pd.DataFrame
    bench: pd.DataFrame
    fear: pd.DataFrame
    notes: dict = field(default_factory=dict)


def load_admin_events(path: Path | None = None) -> pd.DataFrame | None:
    p = path or (C.RAW / "admin" / "_events.csv")
    if not p.exists():
        return None
    return pd.read_csv(p, dtype=str, encoding="utf-8-sig")


def prepare(lookback_start: str, eval_start: str, end: str, admin_events: pd.DataFrame | None = None) -> Context:
    """end = 마지막 리밸런싱일 t' 가 속한 달의 말일. 마지막 보유 월의 청산가(t'+1 시가)를 위해 데이터는 end 뒤 10일까지 더 읽는다(D6)."""
    end_load = (pd.Timestamp(end) + pd.Timedelta(10, unit="D")).strftime("%Y-%m-%d")
    frames = {k: load_wide(k, lookback_start, end_load) for k in FRAME_KEYS}
    days = trading_days(lookback_start, end_load)
    for k in FRAME_KEYS:
        frames[k] = frames[k].reindex(days)
    kospi = load_kospi().reindex(days)
    etf = pd.read_parquet(C.PROCESSED / "etf_indiv_net.parquet").reindex(days)
    names = load_names()

    sig = S.compute_all(frames, kospi["close"], etf)
    rebal = month_end_dates(days, eval_start, end)

    last_seen = frames["close"].apply(lambda s: s.last_valid_index())
    admin = None
    if admin_events is not None:
        iv = admin_intervals(admin_events, last_seen)
        admin = admin_flag(iv, frames["close"].columns, days)
    member, exclusions = build_universe(frames, names, admin, days, rebal)
    rebal = member.index  # build_universe 가 다음 날이 없는 마지막 날을 뺄 수 있다

    R, kinds = holding_returns(frames["open"], frames["close"], frames["ret"], days, rebal)
    rebal_t = R.index  # 마지막 리밸런싱일은 다음 달이 없어 빠진다
    member = member.loc[rebal_t]
    exclusions = exclusions.loc[rebal_t]

    sig_t = {k: sig[k].loc[rebal_t] for k in SIGNAL_KEYS}
    mktcap = frames["mktcap"].loc[rebal_t]
    market = frames["market"].loc[rebal_t]
    sentiment = sig["sentiment"].reindex(rebal_t)

    panel = build_panel(rebal_t, member, R, sig_t, mktcap, market, sentiment)
    bench = benchmarks(kospi, days, rebal, R, member, sig_t)
    return Context(rebal_t, member, exclusions, R, kinds, sig_t, mktcap, market, sentiment, panel, bench, sig["fear"])


def build_panel(rebal, member, R, sig_t, mktcap, market, sentiment) -> pd.DataFrame:
    """리밸런싱일 × 유니버스 멤버 long 패널. z 열은 멤버 안에서 윈저라이즈·표준화(D16)."""
    parts = []
    for t in rebal:
        m = member.loc[t]
        idx = m.index[m]
        df = pd.DataFrame({"date": t, "ticker": idx, "y": R.loc[t, idx].values})
        for k in SIGNAL_KEYS:
            df[k] = sig_t[k].loc[t, idx].values
        df["market"] = market.loc[t, idx].values
        df["sentiment"] = sentiment.loc[t]
        df["crash"] = (df["ret20"] <= df["ret20"].quantile(C.CRASH_BOTTOM_PCT)).astype(float).where(df["ret20"].notna())
        for k in SIGNAL_KEYS:
            df[f"{k}_z"] = winsor_z(df[k]).values
        df["rnb5_pct"] = df["rnb5"].rank(pct=True)
        df["max20_pct"] = df["max20"].rank(pct=True)
        parts.append(df)
    panel = pd.concat(parts, ignore_index=True)
    panel["max_x_rnb"] = panel["max20_z"] * panel["rnb5_z"]
    panel["cgo_x_rsh"] = panel["cgo260_z"] * panel["rsh20_z"]
    panel["rnb_x_crash"] = panel["rnb5_z"] * panel["crash"]
    panel["max_x_frnb"] = panel["max20_z"] * panel["frnb5_z"]
    # D17 단일 점수
    panel["s1"] = np.where(panel["crash"] == 1, -panel["rnb5_z"], 0.0)
    panel["s2"] = panel["cgo260_z"]
    s3_raw = -(panel["max20_pct"] * panel["rnb5_pct"])
    panel["s3"] = s3_raw.groupby(panel["date"]).transform(winsor_z)
    panel["score"] = panel[["s1", "s2", "s3"]].sum(axis=1, min_count=3)
    return panel


def benchmarks(kospi, days, rebal, R, member, sig_t) -> pd.DataFrame:
    """KOSPI(시가→시가), 유니버스 동일가중, 모멘텀 12-1 상위 5분위, 단기 반전(ret20 하위 5분위) 동일가중 (D18)."""
    out = pd.DataFrame(index=R.index)
    out["kospi"] = index_holding_returns(kospi, days, rebal).reindex(R.index)
    out["univ_ew"] = P.equal_weight_universe(R, member)
    mom_ew, _, _, _ = P.quantile_returns(sig_t["mom12_1"], R, member)
    out["mom_q5"] = mom_ew["Q5"]
    rev_ew, _, _, _ = P.quantile_returns(-sig_t["ret20"], R, member)
    out["rev_q5"] = rev_ew["Q5"]
    return out


# ---------------------------------------------------------------- 가설별 검정

def _wide_from_panel(panel: pd.DataFrame, col: str, tickers: pd.Index) -> pd.DataFrame:
    return panel.pivot(index="date", columns="ticker", values=col).reindex(columns=tickers)


def sort_test(ctx: Context, signal_col: str, subset_col: str | None = None) -> dict:
    """신호 5분위(EW·VW), 롱온리 Q5 − 벤치마크, 회전율·비용. subset_col 이 있으면 그 열이 1 인 종목 안에서만."""
    tickers = ctx.member.columns
    sig = _wide_from_panel(ctx.panel, signal_col, tickers)
    member = ctx.member.copy()
    if subset_col:
        sub = _wide_from_panel(ctx.panel, subset_col, tickers).fillna(0.0) == 1.0
        member = member & sub
    ew, vw, counts, qmap = P.quantile_returns(sig, ctx.R, member, ctx.mktcap)
    ew["Q5-Q1"] = ew["Q5"] - ew["Q1"]
    vw["Q5-Q1"] = vw["Q5"] - vw["Q1"]
    W = P.long_only_weights(qmap, C.N_QUANTILES)
    costs = P.turnover_and_costs(W, ctx.R)
    lo = pd.DataFrame({"gross": ew["Q5"]}).join(ctx.bench).join(costs)
    for c in ["base", "low", "high", "real"]:
        lo[f"net_{c}"] = lo["gross"] - lo[f"cost_{c}"]
    stats = {}
    for col in ["gross", "net_base", "net_low", "net_high", "net_real"]:
        for b in ["kospi", "univ_ew", "mom_q5", "rev_q5"]:
            stats[(col, b)] = P.perf_stats(lo[col], lo[b])
    stats_df = pd.DataFrame(stats).T
    stats_df.index.names = ["portfolio", "benchmark"]
    quint_summary = pd.DataFrame({
        "ew_mean": ew.mean(), "ew_t": [P.nw_mean_test(ew[c])["t"] for c in ew.columns],
        "vw_mean": vw.mean().reindex(ew.columns), "vw_t": [P.nw_mean_test(vw[c])["t"] if c in vw else np.nan for c in ew.columns],
        "avg_n": counts.reindex(columns=ew.columns).mean(),
    })
    mde = mde_monthly(lo["gross"] - lo["univ_ew"])
    return {"quintiles_ew": ew, "quintiles_vw": vw, "counts": counts, "quintile_summary": quint_summary,
            "longonly": lo, "stats": stats_df, "mde": pd.Series(mde), "qmap": qmap}


def fm_test(panel: pd.DataFrame, xcols: list[str], mask: pd.Series | None = None, label: str = ""):
    df = panel if mask is None else panel[mask]
    summ, coefs = fama_macbeth(df, "y", xcols)
    summ["spec"] = label
    summ["t_ge_2"] = summ["t"].abs() >= C.T_PREREG
    summ["t_ge_3"] = summ["t"].abs() >= C.T_HLZ
    return summ, coefs


def subperiod_fm(panel: pd.DataFrame, xcols: list[str], key: str, base_mask: pd.Series | None = None) -> pd.DataFrame:
    """사전 등록 보고 항목(D19): 연도별 / KOSPI·KOSDAQ / 시총 상·하위 반 / 2015-06-15 전후 / 2020-01 전후 / 심리 국면."""
    df = panel if base_mask is None else panel[base_mask]
    rows = []
    splits: dict[str, pd.Series] = {}
    for y in sorted(df["date"].dt.year.unique()):
        splits[f"year={y}"] = df["date"].dt.year == y
    splits["market=KOSPI"] = df["market"] == 1
    splits["market=KOSDAQ"] = df["market"] == 2
    med = df.groupby("date")["logmcap"].transform("median")
    splits["size=large"] = df["logmcap"] >= med
    splits["size=small"] = df["logmcap"] < med
    splits["limit=pre_2015-06-15"] = df["date"] < pd.Timestamp("2015-06-15")
    splits["limit=post_2015-06-15"] = df["date"] >= pd.Timestamp("2015-06-15")
    splits["retail=pre_2020"] = df["date"] < pd.Timestamp("2020-01-01")
    splits["retail=post_2020"] = df["date"] >= pd.Timestamp("2020-01-01")
    splits["sentiment=high"] = df["sentiment"] == 1
    splits["sentiment=low"] = df["sentiment"] == 0
    for name, m in splits.items():
        sub = df[m]
        if sub["date"].nunique() < 6:
            continue
        summ, _ = fama_macbeth(sub, "y", xcols)
        r = summ.loc[key]
        rows.append({"split": name, "coef": r["coef"], "t": r["t"], "n_months": r["n_months"], "avg_n": r["avg_n"]})
    return pd.DataFrame(rows)


def run_h1(ctx: Context) -> dict:
    """H1 물타기 역발상: 급락 집합 안에서 −RNB5 상위(개인 순매도 강함)가 높은 수익률. FM: 급락 집합 y ~ rnb5_z + 통제, b<0."""
    ctx.panel["h1_signal"] = -ctx.panel["rnb5"]
    out = sort_test(ctx, "h1_signal", subset_col="crash")
    crash = ctx.panel["crash"] == 1
    x = ["rnb5_z"] + [f"{c}_z" for c in CONTROLS]
    fm_primary, coefs = fm_test(ctx.panel, x, crash, "H1 primary: crash subset")
    fm_nocontrol, _ = fm_test(ctx.panel, ["rnb5_z"], crash, "H1 no controls")
    x2 = ["rnb5_z", "crash", "rnb_x_crash"] + [f"{c}_z" for c in CONTROLS]
    fm_inter, _ = fm_test(ctx.panel, x2, None, "H1 secondary: full universe interaction")
    placebo, _ = fm_test(ctx.panel, ["frnb5_z"] + [f"{c}_z" for c in ["logmcap", "mom12_1", "ret20", "turn20"]], crash,
                         "H1 placebo: foreign RNB in crash subset")
    out["fm"] = pd.concat([fm_primary, fm_nocontrol, fm_inter, placebo])
    out["fm_coefs"] = coefs
    out["subperiods"] = subperiod_fm(ctx.panel, x, "rnb5_z", crash)
    out["hypothesis"] = {"key": "rnb5_z", "expected_sign": -1, "spec": "H1 primary: crash subset"}
    return out


def run_h2(ctx: Context) -> dict:
    """H2 처분효과 압력: CGO 상위가 높은 수익률. FM: y ~ cgo_z + 통제, b>0. 보조: RSH 상·하위 반, 상호작용."""
    out = sort_test(ctx, "cgo260")
    x = ["cgo260_z"] + [f"{c}_z" for c in CONTROLS]
    fm_primary, coefs = fm_test(ctx.panel, x, None, "H2 primary")
    fm_nocontrol, _ = fm_test(ctx.panel, ["cgo260_z"], None, "H2 no controls")
    med = ctx.panel.groupby("date")["rsh20"].transform("median")
    fm_hi, _ = fm_test(ctx.panel, x, ctx.panel["rsh20"] >= med, "H2 RSH high half")
    fm_lo, _ = fm_test(ctx.panel, x, ctx.panel["rsh20"] < med, "H2 RSH low half")
    x2 = ["cgo260_z", "rsh20_z", "cgo_x_rsh"] + [f"{c}_z" for c in CONTROLS]
    fm_inter, _ = fm_test(ctx.panel, x2, None, "H2 secondary: CGO x RSH interaction")
    out["fm"] = pd.concat([fm_primary, fm_nocontrol, fm_hi, fm_lo, fm_inter])
    out["fm_coefs"] = coefs
    out["subperiods"] = subperiod_fm(ctx.panel, x, "cgo260_z")
    # 조건부 5분위: RSH 상·하위 반 각각
    tickers = ctx.member.columns
    hi = (ctx.panel["rsh20"] >= med).astype(float)
    rsh_hi = _wide_from_panel(ctx.panel.assign(hi=hi), "hi", tickers)
    sig = _wide_from_panel(ctx.panel, "cgo260", tickers)
    ew_hi, _, _, _ = P.quantile_returns(sig, ctx.R, ctx.member & (rsh_hi == 1.0), None)
    ew_lo, _, _, _ = P.quantile_returns(sig, ctx.R, ctx.member & (rsh_hi == 0.0), None)
    cond = pd.DataFrame({"RSH_high_mean": ew_hi.mean(), "RSH_high_t": [P.nw_mean_test(ew_hi[c])["t"] for c in ew_hi.columns],
                         "RSH_low_mean": ew_lo.mean(), "RSH_low_t": [P.nw_mean_test(ew_lo[c])["t"] for c in ew_lo.columns]})
    cond.loc["Q5-Q1"] = [(ew_hi["Q5"] - ew_hi["Q1"]).mean(), P.nw_mean_test(ew_hi["Q5"] - ew_hi["Q1"])["t"],
                         (ew_lo["Q5"] - ew_lo["Q1"]).mean(), P.nw_mean_test(ew_lo["Q5"] - ew_lo["Q1"])["t"]]
    out["conditional_rsh"] = cond
    out["hypothesis"] = {"key": "cgo260_z", "expected_sign": +1, "spec": "H2 primary"}
    return out


def run_h3(ctx: Context) -> dict:
    """H3 복권형 매수 압력: MAX 크고 RNB 큰 종목이 낮은 수익률. FM: y ~ max_z + rnb_z + max×rnb + 통제, b3<0. 심리 저조 국면에서 더 음수."""
    out = sort_test(ctx, "s3")
    x = ["max20_z", "rnb5_z", "max_x_rnb"] + [f"{c}_z" for c in CONTROLS]
    fm_primary, coefs = fm_test(ctx.panel, x, None, "H3 primary")
    fm_nocontrol, _ = fm_test(ctx.panel, ["max20_z", "rnb5_z", "max_x_rnb"], None, "H3 no controls")
    fm_low, _ = fm_test(ctx.panel, x, ctx.panel["sentiment"] == 0, "H3 sentiment low")
    fm_high, _ = fm_test(ctx.panel, x, ctx.panel["sentiment"] == 1, "H3 sentiment high")
    placebo, _ = fm_test(ctx.panel, ["max20_z", "frnb5_z", "max_x_frnb"] + [f"{c}_z" for c in ["logmcap", "mom12_1", "ret20", "turn20"]],
                         None, "H3 placebo: foreign RNB")
    out["fm"] = pd.concat([fm_primary, fm_nocontrol, fm_low, fm_high, placebo])
    out["fm_coefs"] = coefs
    out["subperiods"] = subperiod_fm(ctx.panel, x, "max_x_rnb")
    # 종속 이중 정렬: MAX 5분위 → 그 안에서 RNB 5분위 (EW 평균 수익률 5×5)
    tickers = ctx.member.columns
    maxw = _wide_from_panel(ctx.panel, "max20", tickers)
    rnbw = _wide_from_panel(ctx.panel, "rnb5", tickers)
    _, _, _, qmax = P.quantile_returns(maxw, ctx.R, ctx.member, None)
    grid = {}
    for k in range(1, C.N_QUANTILES + 1):
        sub = ctx.member & (qmax == k)
        ew, _, _, _ = P.quantile_returns(rnbw, ctx.R, sub, None)
        grid[f"MAX_Q{k}"] = ew.mean()
        grid[f"MAX_Q{k}_t"] = pd.Series([P.nw_mean_test(ew[c])["t"] for c in ew.columns], index=ew.columns)
    ds = pd.DataFrame(grid)
    ds.index.name = "RNB_quintile"
    out["double_sort"] = ds
    out["hypothesis"] = {"key": "max_x_rnb", "expected_sign": -1, "spec": "H3 primary"}
    return out


def verdict(fm: pd.DataFrame, hyp: dict) -> dict:
    """탐색 기간 판정 재료: 사전등록 기준은 '탐색 t≥2 & 홀드아웃 부호 일치' 이므로 여기서는 탐색 통과 여부만."""
    r = fm[fm["spec"] == hyp["spec"]].loc[hyp["key"]]
    sign_ok = np.sign(r["coef"]) == hyp["expected_sign"]
    return {"coef": float(r["coef"]), "t": float(r["t"]), "sign_as_hypothesized": bool(sign_ok),
            "explore_pass_t2": bool(sign_ok and abs(r["t"]) >= C.T_PREREG), "t_ge_3": bool(sign_ok and abs(r["t"]) >= C.T_HLZ)}


def save_results(outdir: Path, name: str, res: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(outdir / f"{name}_{k}.csv", encoding="utf-8-sig")
        elif isinstance(v, pd.Series):
            v.to_frame(k).to_csv(outdir / f"{name}_{k}.csv", encoding="utf-8-sig")
