"""4단계 결과 리포트 HTML 1장 — 결과 CSV·MD 와 그림에서만 만든다 (브리프 6절 ④). 표에는 분모(개월 수·종목 수·결측 수)를 넣는다.
사용: .venv/bin/python scripts/make_report.py  → results/report.html
"""
from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rbf import config as C  # noqa: E402

R = C.RESULTS
S1, S2, S3 = R / "stage1", R / "stage2", R / "stage3"


def read(p: Path, **k) -> pd.DataFrame | None:
    return pd.read_csv(p, index_col=0, encoding="utf-8-sig", **k) if p.exists() else None


def table(df: pd.DataFrame | None, nd: int = 4, caption: str = "", index: bool = True) -> str:
    if df is None or len(df) == 0:
        return f"<p class='missing'>{html.escape(caption)}: 결과 파일 없음</p>"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return d.to_html(index=index, border=0, escape=True).replace("<table", f"<table class='t'>{cap}", 1) if cap else d.to_html(index=index, border=0)


def fm_table(h: str, stage: Path) -> pd.DataFrame | None:
    df = read(stage / f"{h}_fm.csv")
    if df is None:
        return None
    df = df.reset_index().rename(columns={"index": "variable"})
    return df[["spec", "variable", "coef", "se", "t", "n_months", "avg_n", "t_ge_2", "t_ge_3"]]


def section_h(h: str, name: str, grades: dict | None) -> str:
    out = [f"<h2>{html.escape(name)}</h2>"]
    if grades and h in grades:
        g = grades[h]
        out.append(f"<p class='grade'>판정: <b>{g['grade']}</b> — 탐색 t={g['explore']['t']:.2f} (통과 {'예' if g['explore']['explore_pass_t2'] else '아니오'}), "
                   f"홀드아웃 coef={g['holdout']['coef']:+.4f}, t={g['holdout']['t']:.2f} (부호 {'일치' if g['holdout']['sign_as_hypothesized'] else '불일치'})</p>")
    for label, stage in [("탐색 2010-02~2020-12", S1), ("홀드아웃 2021-01~2026-09", S2)]:
        q = read(stage / f"{h}_quintile_summary.csv")
        cnt = read(stage / f"{h}_counts.csv")
        if q is not None and cnt is not None:
            q = q.copy()
            q["n_months"] = len(cnt)
            q.loc["Q1":"Q5", "avg_missing_signal"] = cnt["n_signal_missing"].mean()
        out.append(table(q, 4, f"{label} — 5분위 월평균 수익률 (EW·VW, NW t, 평균 종목 수, 개월 수, 평균 신호 결측 수)"))
        out.append(table(fm_table(h, stage), 4, f"{label} — Fama–MacBeth (설명변수 표준화, NW 시차 3)", index=False))
        st = read(stage / f"{h}_stats.csv")
        if st is not None:
            st = st.reset_index()
            st = st[["portfolio", "benchmark", "n_months", "mean_m", "excess_m", "excess_t", "excess_ann", "ann_ret", "ann_vol", "sharpe", "mdd", "hit"]]
        out.append(table(st, 4, f"{label} — 롱온리 Q5 − 벤치마크 (비용 시나리오별: gross / 0.15% / 0.10% / 0.25% / 실제 세율)", index=False))
        lo = read(stage / f"{h}_longonly.csv")
        if lo is not None:
            out.append(f"<p>월 회전율 평균 {lo['turnover'].mean():.3f}, 월 비용 평균: 기본 {lo['cost_base'].mean():.4f}, 실제 세율 {lo['cost_real'].mean():.4f}</p>")
        sp = read(stage / f"{h}_subperiods.csv")
        out.append(table(sp, 4, f"{label} — 소구간 (사전 등록 보고 항목)", index=False))
        cond = read(stage / f"{h}_conditional_rsh.csv")
        if cond is not None:
            out.append(table(cond, 4, f"{label} — RSH 상·하위 반 조건부 5분위"))
        ds = read(stage / f"{h}_double_sort.csv")
        if ds is not None:
            out.append(table(ds, 4, f"{label} — 종속 이중 정렬 MAX×RNB (EW 월평균)"))
        mde = read(stage / f"{h}_mde.csv")
        if mde is not None:
            out.append(f"<p>검출 가능 최소 효과(t=2): 월 {float(mde.loc['mde_monthly'].iloc[0]):.4f} (초과수익 월 표준편차 {float(mde.loc['sd_monthly'].iloc[0]):.4f}, {int(mde.loc['n_months'].iloc[0])}개월)</p>")
    return "\n".join(out)


def main() -> None:
    meta1 = json.loads((S1 / "run_meta.json").read_text(encoding="utf-8")) if (S1 / "run_meta.json").exists() else {}
    meta2 = json.loads((S2 / "run_meta.json").read_text(encoding="utf-8")) if (S2 / "run_meta.json").exists() else {}
    grades = meta2.get("grades")
    runs = (R / "holdout_runs.jsonl")
    n_runs = len([ln for ln in runs.read_text(encoding="utf-8").splitlines() if ln.strip()]) if runs.exists() else 0
    parts = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>retail-bias-factor 결과 리포트</title>",
             "<style>body{font-family:-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;color:#222}"
             "table{border-collapse:collapse;font-size:12px;margin:8px 0 18px}th,td{border-bottom:1px solid #ddd;padding:3px 6px;text-align:right}th{background:#f4f4f4}"
             "td:first-child,th:first-child{text-align:left}caption{text-align:left;font-weight:600;padding:4px 0}img{max-width:100%}.grade{background:#f7f3e8;padding:6px 10px}"
             ".missing{color:#a00}h2{border-top:2px solid #333;padding-top:10px;margin-top:28px}</style></head><body>",
             "<h1>개인투자자 행동 편향의 흔적으로 만든 역발상 팩터 — 결과 리포트</h1>",
             f"<p>생성 {datetime.now().isoformat(timespec='seconds')}. 투자 조언이 아니다. 모든 수치는 <code>results/</code> 의 CSV 에서 왔고 "
             f"<code>scripts/run_all.sh</code> 로 재생성된다. 탐색 커밋 {meta1.get('commit', '?')[:10]}, 홀드아웃 실행 횟수 <b>{n_runs}</b>"
             + (f" (잠금 커밋 {meta2['run']['lock_commit'][:10]})" if meta2.get("run") else "") + ".</p>"]
    # 판정 요약
    if grades:
        gt = pd.DataFrame({h: {"탐색 coef": g["explore"]["coef"], "탐색 t": g["explore"]["t"], "탐색 통과(t≥2)": g["explore"]["explore_pass_t2"],
                               "홀드아웃 coef": g["holdout"]["coef"], "홀드아웃 t": g["holdout"]["t"], "판정": g["grade"]} for h, g in grades.items()}).T
        parts.append(table(gt, 4, "판정 요약 (사전등록 기준: 탐색 |t|≥2 & 가설 부호, 홀드아웃 부호 일치 → 지지 / 뒤집힘 → 불안정 / 탐색 실패 → 기각)"))
    # 유니버스
    ex1, ex2 = read(S1 / "universe_exclusions.csv"), read(S2 / "universe_exclusions.csv")
    if ex1 is not None:
        u = pd.DataFrame({"탐색 평균": ex1.mean(), "홀드아웃 평균": ex2.mean() if ex2 is not None else None}).T
        parts.append(table(u, 1, "유니버스 제외 단계별 남은 종목 수 (월 평균): 상장 → 보통주 → 스팩 제외 → 관리종목 제외 → 거래정지 제외 → t+1 체결 가능 → 시총 하위 20% 제외 → 거래대금 하위 20% 제외"))
    b1 = read(S1 / "benchmarks.csv")
    if b1 is not None:
        parts.append(table(b1.agg(["mean", "std", "count"]).T, 4, "벤치마크 월 수익률 (탐색): KOSPI(시가→시가), 유니버스 EW, 모멘텀 12-1 Q5, 단기 반전 Q5"))
    parts.append("<h2>그림</h2>")
    for f, cap in [("fig1_quintiles.png", "그림 1 가설별 5분위"), ("fig2_cumulative.png", "그림 2 누적 수익률"), ("fig3_ml_vs_zsum.png", "그림 3 ML vs 합산"), ("fig4_fear_events.png", "그림 4 공포 z 사건 연구")]:
        p = C.FIGURES / f
        parts.append(f"<figure><img src='../figures/{f}' alt='{cap}'><figcaption>{cap}</figcaption></figure>" if p.exists() else f"<p class='missing'>{cap}: 없음</p>")
    for h, name in [("h1", "H1 물타기 역발상"), ("h2", "H2 처분효과 압력"), ("h3", "H3 복권형 매수 압력")]:
        parts.append(section_h(h, name, grades))
    parts.append("<h2>ML 결합 (3단계)</h2>")
    parts.append(table(read(S3 / "ml_summary.csv").reset_index(drop=True) if (S3 / "ml_summary.csv").exists() else None, 4, "Ridge·LightGBM(기본값, walk-forward) vs 세 신호 z-score 합산 — IC·Q5 롱온리", index=False))
    parts.append(table(read(S3 / "ml_vs_zsum.csv").reset_index(drop=True) if (S3 / "ml_vs_zsum.csv").exists() else None, 4, "ML − 합산 IC 차이 (NW t)", index=False))
    parts.append("<h2>H5 공포 오버레이 (3단계)</h2>")
    parts.append(table(read(S3 / "overlay_event_study.csv"), 4, "z≥2 사건 이후 KOSPI 수익률 (사건 수 event_n 이 분모)"))
    ov = read(S3 / "overlay_stats.csv")
    parts.append(table(ov.reset_index(drop=True) if ov is not None else None, 4, "오버레이 성과 (사건 10건 이상일 때만 보고)", index=False))
    honest = (ROOT / "docs" / "08_limitations.md")
    if honest.exists():
        parts.append("<h2>정직하게 적어야 할 것</h2><pre style='white-space:pre-wrap;font-family:inherit'>" + html.escape(honest.read_text(encoding="utf-8")) + "</pre>")
    parts.append("</body></html>")
    (R / "report.html").write_text("\n".join(parts), encoding="utf-8")
    print("->", R / "report.html")


if __name__ == "__main__":
    main()
