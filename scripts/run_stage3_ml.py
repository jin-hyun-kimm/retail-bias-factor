"""3단계 ML 결합 (브리프 4-7, 설계 D21): Ridge·LightGBM walk-forward vs 세 신호 z-score 단순 합산.

전 기간 패널(탐색+홀드아웃)을 만들고, 매년 1월 재학습(확장 창)해 그해 12개월을 채점한다. 하이퍼파라미터는 기본값.
결과: results/stage3/ml_*.csv, ml_summary.md
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rbf import analysis as A  # noqa: E402
from rbf import config as C  # noqa: E402
from rbf import portfolio as P  # noqa: E402
from run_stage1 import git_hash, md_table  # noqa: E402

FEATURES = ["rnb5_z", "crash", "cgo260_z", "rsh20_z", "max20_z", "sentiment", "logmcap_z", "mom12_1_z", "ret20_z", "turn20_z", "frnb5_z"]
OUT = C.RESULTS / "stage3"


def walk_forward(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge

    df = panel.dropna(subset=["y"]).copy()
    df["y_pct"] = df.groupby("date")["y"].rank(pct=True)
    df["hold_year"] = (df["date"] + pd.offsets.MonthBegin(1)).dt.year   # 보유 월의 연도
    X_all = df[FEATURES].astype(float)
    first_test_year = pd.Timestamp(C.ML_FIRST_TRAIN_END).year + 1
    preds = {"ridge": pd.Series(np.nan, index=df.index), "lgbm": pd.Series(np.nan, index=df.index)}
    importances = []
    for yr in range(first_test_year, df["hold_year"].max() + 1):
        tr = df["hold_year"] < yr
        te = df["hold_year"] == yr
        if te.sum() == 0 or tr.sum() == 0:
            continue
        Xtr, ytr = X_all[tr].fillna(0.0), df.loc[tr, "y_pct"]
        Xte = X_all[te]
        ridge = Ridge().fit(Xtr, ytr)
        preds["ridge"][te] = ridge.predict(Xte.fillna(0.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lgbm = LGBMRegressor(verbose=-1).fit(X_all[tr], ytr)
        preds["lgbm"][te] = lgbm.predict(Xte)
        importances.append(pd.Series(lgbm.booster_.feature_importance(importance_type="gain"), index=FEATURES, name=f"lgbm_gain_{yr}"))
        importances.append(pd.Series(ridge.coef_, index=FEATURES, name=f"ridge_coef_{yr}"))
    df["pred_ridge"] = preds["ridge"]
    df["pred_lgbm"] = preds["lgbm"]
    df["pred_zsum"] = df["score"]
    return df, pd.DataFrame(importances).T


def monthly_ic(df: pd.DataFrame, col: str) -> pd.Series:
    out = {}
    for d, g in df.groupby("date"):
        g = g[[col, "y"]].dropna()
        if len(g) >= 30:
            out[d] = spearmanr(g[col], g["y"]).correlation
    return pd.Series(out, name=col)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    admin = A.load_admin_events()
    ctx = A.prepare(C.LOOKBACK_START, C.EVAL_START, C.EVAL_END, admin)
    df, imp = walk_forward(ctx.panel)
    imp.to_csv(OUT / "ml_importance.csv", encoding="utf-8-sig")
    hold_start = pd.Timestamp(C.HOLDOUT_START) - pd.Timedelta(10, unit="D")
    ics = pd.DataFrame({m: monthly_ic(df, f"pred_{m}") for m in ["zsum", "ridge", "lgbm"]})
    ics.to_csv(OUT / "ml_ic_monthly.csv", encoding="utf-8-sig")
    rows = []
    tickers = ctx.member.columns
    lo_all = {}
    for m in ["zsum", "ridge", "lgbm"]:
        sig = df.pivot(index="date", columns="ticker", values=f"pred_{m}").reindex(columns=tickers)
        ew, _, cnt, qmap = P.quantile_returns(sig, ctx.R, ctx.member, None)
        W = P.long_only_weights(qmap, C.N_QUANTILES)
        costs = P.turnover_and_costs(W, ctx.R)
        lo = pd.DataFrame({"gross": ew["Q5"], "net_base": ew["Q5"] - costs["cost_base"], "turnover": costs["turnover"]}).join(ctx.bench)
        lo_all[m] = lo
        lo.to_csv(OUT / f"ml_longonly_{m}.csv", encoding="utf-8-sig")
        for period, mask in [("explore", lo.index < hold_start), ("holdout", lo.index >= hold_start)]:
            sub = lo[mask]
            ic = ics.loc[ics.index.isin(sub.index), m]
            ict = P.nw_mean_test(ic)
            st = P.perf_stats(sub["net_base"], sub["univ_ew"])
            rows.append({"model": m, "period": period, "n_months": len(sub), "ic_mean": ic.mean(), "ic_t": ict["t"],
                         "ic_hit": (ic > 0).mean(), "q5_net_excess_m": st.get("excess_m"), "q5_net_excess_t": st.get("excess_t"),
                         "q5_sharpe": st["sharpe"], "turnover": sub["turnover"].mean()})
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "ml_summary.csv", index=False, encoding="utf-8-sig")
    # ML vs 합산 차이의 검정 (월별 IC 차이)
    diff_rows = []
    for m in ["ridge", "lgbm"]:
        for period, mask in [("explore", ics.index < hold_start), ("holdout", ics.index >= hold_start)]:
            d = (ics.loc[mask, m] - ics.loc[mask, "zsum"]).dropna()
            t = P.nw_mean_test(d)
            diff_rows.append({"model": m, "period": period, "ic_diff_mean": t["mean"], "ic_diff_t": t["t"], "n": t["n"]})
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(OUT / "ml_vs_zsum.csv", index=False, encoding="utf-8-sig")
    L = ["# 3단계 ML 결합 — Ridge·LightGBM(기본값, walk-forward 확장 창, 연 1회 재학습) vs 세 신호 z-score 합산", "",
         f"생성 {datetime.now().isoformat(timespec='seconds')}, 커밋 {git_hash()[:10]}. 첫 학습 창 보유 월 2010-02~{C.ML_FIRST_TRAIN_END[:7]}, 첫 채점 연도 {pd.Timestamp(C.ML_FIRST_TRAIN_END).year + 1}.", "",
         "## 요약 (IC = 월별 Spearman 순위 상관, Q5 = 상위 5분위 롱온리 동일가중, 비용 0.15% 반영, 초과수익은 유니버스 동일가중 대비)", "",
         md_table(summ, 4, index=False), "", "## ML − 합산 IC 차이 (NW t)", "", md_table(diff, 4, index=False), "",
         "## 특징 중요도 (LightGBM gain, Ridge 계수; 연도별 재학습)", "", md_table(imp, 3), ""]
    verdict = []
    for m in ["ridge", "lgbm"]:
        for period in ["explore", "holdout"]:
            r = diff[(diff.model == m) & (diff.period == period)].iloc[0]
            verdict.append(f"- {m} / {period}: IC 차이 {r['ic_diff_mean']:+.4f} (t={r['ic_diff_t']:.2f}) → "
                           + ("합산을 이기지 못함" if not (r["ic_diff_mean"] > 0 and r["ic_diff_t"] >= 2) else "합산보다 높음(t≥2)"))
    L += ["## 판정", ""] + verdict + [""]
    (OUT / "ml_summary.md").write_text("\n".join(L), encoding="utf-8")
    json.dump({"ts": datetime.now().isoformat(timespec="seconds"), "commit": git_hash()}, open(OUT / "ml_meta.json", "w"))
    print("\n".join(L)[:2500])


if __name__ == "__main__":
    main()
