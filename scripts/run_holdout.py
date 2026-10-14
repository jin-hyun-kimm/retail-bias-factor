"""2단계: 홀드아웃(보유 월 2021-01 ~ 2026-09) 1회 실행. 실행할 때마다 results/holdout_runs.jsonl 에 기록이 남는다(D23 ②).

실행 조건: docs/03_preregistration.md 에 '잠금 커밋: <hash>' 줄이 있어야 한다(사전등록 승인 후 잠금).
사용: .venv/bin/python scripts/run_holdout.py --note "첫 실행"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rbf import analysis as A  # noqa: E402
from rbf import config as C  # noqa: E402
from run_stage1 import git_hash, md_table, summarize  # noqa: E402

LOG = C.RESULTS / "holdout_runs.jsonl"
PREREG = ROOT / "docs" / "03_preregistration.md"


def lock_hash() -> str | None:
    if not PREREG.exists():
        return None
    m = re.search(r"잠금 커밋:\s*([0-9a-f]{7,40})", PREREG.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def restrict(ctx: A.Context, start: pd.Timestamp) -> A.Context:
    keep = ctx.rebal >= start
    ctx.rebal = ctx.rebal[keep]
    ctx.member = ctx.member.loc[ctx.rebal]
    ctx.exclusions = ctx.exclusions.loc[ctx.rebal]
    ctx.R = ctx.R.loc[ctx.rebal]
    ctx.exit_kind = ctx.exit_kind.loc[ctx.rebal]
    ctx.sig = {k: v.loc[ctx.rebal] for k, v in ctx.sig.items()}
    ctx.mktcap = ctx.mktcap.loc[ctx.rebal]
    ctx.market = ctx.market.loc[ctx.rebal]
    ctx.sentiment = ctx.sentiment.loc[ctx.rebal]
    ctx.panel = ctx.panel[ctx.panel["date"].isin(ctx.rebal)].reset_index(drop=True)
    ctx.bench = ctx.bench.loc[ctx.rebal]
    return ctx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True, help="이번 실행의 사유 (첫 실행 / 재실행 사유)")
    a = ap.parse_args()
    lock = lock_hash()
    if lock is None:
        raise SystemExit("사전등록 문서에 잠금 커밋이 없다. 승인 후 잠근 뒤 실행한다.")
    prev = [json.loads(ln) for ln in LOG.read_text(encoding="utf-8").splitlines() if ln.strip()] if LOG.exists() else []
    run_no = len(prev) + 1
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "commit": git_hash(), "lock_commit": lock, "run_no": run_no, "note": a.note}
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"홀드아웃 실행 #{run_no} 기록: {rec}")
    if run_no > 1:
        print("주의: 홀드아웃을 다시 돌렸다. 리포트에 횟수와 사유를 그대로 적는다.")

    outdir = C.RESULTS / "stage2"
    outdir.mkdir(parents=True, exist_ok=True)
    admin = A.load_admin_events()
    # 홀드아웃 리밸런싱일: 2020-12 말(보유 월 2021-01) ~ 2026-08 말(보유 월 2026-09). 룩백은 전체.
    hold_first_rebal = pd.Timestamp(C.HOLDOUT_START) - pd.Timedelta(10, unit="D")
    ctx = A.prepare(C.LOOKBACK_START, hold_first_rebal.strftime("%Y-%m-%d"), C.EVAL_END, admin)
    ctx = restrict(ctx, hold_first_rebal)

    ctx.exclusions.to_csv(outdir / "universe_exclusions.csv", encoding="utf-8-sig")
    ctx.bench.to_csv(outdir / "benchmarks.csv", encoding="utf-8-sig")
    ctx.panel.to_parquet(outdir / "panel.parquet")
    res = {"h1": A.run_h1(ctx), "h2": A.run_h2(ctx), "h3": A.run_h3(ctx)}
    for h, r in res.items():
        A.save_results(outdir, h, {k: v for k, v in r.items() if k not in ("qmap", "hypothesis")})
        r["qmap"].to_parquet(outdir / f"{h}_qmap.parquet")
    # 판정(브리프 4-6): 탐색 통과 여부는 stage1/run_meta.json, 홀드아웃 부호는 여기서
    s1 = json.loads((C.RESULTS / "stage1" / "run_meta.json").read_text(encoding="utf-8"))["verdicts"]
    grades = {}
    for h, r in res.items():
        v = A.verdict(r["fm"], r["hypothesis"])
        e = s1[h]
        if e["explore_pass_t2"] and v["sign_as_hypothesized"]:
            g = "지지"
        elif e["explore_pass_t2"] and not v["sign_as_hypothesized"]:
            g = "불안정"
        else:
            g = "기각"
        grades[h] = {"explore": e, "holdout": v, "grade": g}
    meta = {"run": rec, "grades": grades, "n_rebal": int(len(ctx.rebal))}
    (outdir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    text = summarize(ctx, res, outdir, f"홀드아웃 {ctx.rebal[0].date()} ~ {ctx.rebal[-1].date()} (실행 #{run_no})")
    grade_tbl = pd.DataFrame({h: {"탐색 t": g["explore"]["t"], "탐색 통과": g["explore"]["explore_pass_t2"],
                                  "홀드아웃 coef": g["holdout"]["coef"], "홀드아웃 t": g["holdout"]["t"], "판정": g["grade"]}
                              for h, g in grades.items()}).T
    text += "\n\n## 판정 (사전등록 기준)\n\n" + md_table(grade_tbl, 3) + f"\n\n홀드아웃 실행 횟수: {run_no}\n"
    (outdir / "summary.md").write_text(text, encoding="utf-8")
    print(text[-1500:])


if __name__ == "__main__":
    main()
