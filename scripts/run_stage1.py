"""1단계: 탐색 기간(보유 월 ≤ 2020-12) H1~H3 검정. 결과는 results/stage1/ 에 CSV 와 summary.md 로.

사용: .venv/bin/python scripts/run_stage1.py [--dev-start 2015-01-02 --dev-eval 2016-01-04 --dev-end 2017-12-31 --out results/dev]
dev 옵션은 파이프라인 점검용(짧은 구간). 본 실행은 옵션 없이.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rbf import config as C  # noqa: E402
from rbf import analysis as A  # noqa: E402


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def fmt(x, nd=4):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def md_table(df: pd.DataFrame, nd=4, index=True) -> str:
    d = df.copy()
    if index:
        d = d.reset_index()
    cols = list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(fmt(v, nd) if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def summarize(ctx: A.Context, res: dict[str, dict], outdir: Path, period_label: str) -> str:
    L = [f"# 1단계 탐색 기간 결과 — {period_label}", "",
         f"생성 {datetime.now().isoformat(timespec='seconds')}, 커밋 {git_hash()[:10]}", "",
         "판정 기준(브리프 4-6): 탐색 Fama–MacBeth |t|≥2.0 이고 가설 부호, 그리고 홀드아웃에서 같은 부호 → 지지. 여기서는 탐색 통과 여부만 적는다. "
         "HLZ(2016) t≥3.0 은 참고.", ""]
    L += ["## 유니버스 (월별 제외 단계별 남은 수, 평균)", "", md_table(ctx.exclusions.mean().to_frame("avg_n").T, 1), "",
          f"리밸런싱 {len(ctx.rebal)}회 ({ctx.rebal[0].date()} ~ {ctx.rebal[-1].date()}), 청산 유형: "
          + ", ".join(f"{k}={v}" for k, v in ctx.exit_kind.where(ctx.member).stack().value_counts().items()), ""]
    L += ["## 벤치마크 월평균 수익률", "", md_table(ctx.bench.agg(["mean", "std"]).T, 4), ""]
    for h, r in res.items():
        v = A.verdict(r["fm"], r["hypothesis"])
        L += [f"## {h.upper()} — 탐색 통과: {'예' if v['explore_pass_t2'] else '아니오'} (coef={v['coef']:.4f}, t={v['t']:.2f}, "
              f"가설 부호={'일치' if v['sign_as_hypothesized'] else '불일치'}, t≥3={'예' if v['t_ge_3'] else '아니오'})", ""]
        L += ["### 5분위 월평균 수익률 (Q5 = 가설 방향 상위)", "", md_table(r["quintile_summary"], 4), ""]
        if "conditional_rsh" in r:
            L += ["### RSH(개인 거래 비중) 상·하위 반 조건부 5분위", "", md_table(r["conditional_rsh"], 4), ""]
        if "double_sort" in r:
            L += ["### 종속 이중 정렬: MAX 5분위 × RNB 5분위 (EW 월평균, Q5 = 높은 값)", "", md_table(r["double_sort"], 4), ""]
        L += ["### Fama–MacBeth (y = 다음 달 보유 수익률, 설명변수 표준화; NW 시차 3)", "",
              md_table(r["fm"][["spec", "coef", "se", "t", "n_months", "avg_n", "t_ge_2", "t_ge_3"]], 4), ""]
        L += ["### 롱온리 Q5 − 벤치마크 (월, 비용 시나리오별)", "",
              md_table(r["stats"][["n_months", "mean_m", "excess_m", "excess_t", "excess_ann", "ann_ret", "ann_vol", "sharpe", "mdd", "hit"]], 4), ""]
        L += [f"회전율(월 평균) {r['longonly']['turnover'].mean():.3f}, 비용(월 평균) 기본 {r['longonly']['cost_base'].mean():.4f} / "
              f"실제세율 {r['longonly']['cost_real'].mean():.4f}", ""]
        L += ["### 소구간 (사전 등록 보고 항목; 선택용 아님)", "", md_table(r["subperiods"], 4, index=False), ""]
        m = r["mde"]
        L += [f"검출 가능 최소 효과(MDE, t=2): 월 {m['mde_monthly']:.4f} (초과수익 월 표준편차 {m['sd_monthly']:.4f}, {int(m['n_months'])}개월)", ""]
    text = "\n".join(L)
    (outdir / "summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-start")
    ap.add_argument("--dev-eval")
    ap.add_argument("--dev-end")
    ap.add_argument("--out")
    a = ap.parse_args()
    dev = bool(a.dev_start)
    lookback = a.dev_start or C.LOOKBACK_START
    eval_start = a.dev_eval or C.EVAL_START
    end = a.dev_end or C.EXPLORE_END
    outdir = Path(a.out) if a.out else (C.RESULTS / ("dev" if dev else "stage1"))
    outdir.mkdir(parents=True, exist_ok=True)
    if not dev and pd.Timestamp(end) > pd.Timestamp(C.EXPLORE_END):
        raise SystemExit("1단계는 탐색 기간까지만 본다. 홀드아웃은 scripts/run_holdout.py")

    admin = A.load_admin_events()
    print(f"period lookback={lookback} eval={eval_start} end={end} admin_events={'yes' if admin is not None else 'NO (관리종목 미적용)'}")
    ctx = A.prepare(lookback, eval_start, end, admin)
    ctx.exclusions.to_csv(outdir / "universe_exclusions.csv", encoding="utf-8-sig")
    ctx.bench.to_csv(outdir / "benchmarks.csv", encoding="utf-8-sig")
    ctx.panel.to_parquet(outdir / "panel.parquet")
    res = {"h1": A.run_h1(ctx), "h2": A.run_h2(ctx), "h3": A.run_h3(ctx)}
    for h, r in res.items():
        A.save_results(outdir, h, {k: v for k, v in r.items() if k not in ("qmap", "hypothesis")})
        r["qmap"].to_parquet(outdir / f"{h}_qmap.parquet")
    meta = {"ts": datetime.now().isoformat(timespec="seconds"), "commit": git_hash(), "lookback": lookback, "eval_start": eval_start,
            "end": end, "admin_applied": admin is not None, "n_rebal": int(len(ctx.rebal)),
            "verdicts": {h: A.verdict(r["fm"], r["hypothesis"]) for h, r in res.items()}}
    (outdir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    text = summarize(ctx, res, outdir, f"{eval_start} ~ {end}")
    print(text[:3000])
    print(f"\n-> {outdir}")


if __name__ == "__main__":
    main()
