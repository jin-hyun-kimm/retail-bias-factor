"""데이터 검증 표 (설계 D3·D4·D5): 믿어도 되는 이유를 숫자로. results/data_validation.md 와 CSV.
검사: ① 거래일 달력 대비 원천 파일 누락 ② 스냅샷 종목 수의 연속성(전일 대비 급변) ③ 등락률 vs 종가비율 일치
④ 투자자별 순매수 합(개인+외국인+기관합계+기타) — 기타 없이 세 주체만 있으므로 '세 주체 합' 분포만 기록 ⑤ 거래정지·한 가격 날 수 ⑥ 시가총액 = 종가×상장주식수 일치
사용: .venv/bin/python scripts/validate_data.py [--start 2009-01-02 --end 2026-09-30]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rbf import config as C  # noqa: E402
from rbf.io_raw import load_wide, trading_days  # noqa: E402
from run_stage1 import md_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=C.LOOKBACK_START)
    ap.add_argument("--end", default=C.EVAL_END)
    a = ap.parse_args()
    days = trading_days(a.start, a.end)
    L = [f"# 데이터 검증 — {a.start} ~ {a.end}", "", f"생성 {datetime.now().isoformat(timespec='seconds')}", ""]
    # ① 원천 파일 누락
    missing = []
    for d in days:
        s = d.strftime("%Y%m%d"); y = s[:4]
        need = [C.RAW / "ohlcv" / y / f"{s}_KOSPI.csv", C.RAW / "ohlcv" / y / f"{s}_KOSDAQ.csv", C.RAW / "cap" / y / f"{s}_ALL.csv"] + \
               [C.RAW / "netbuy" / y / f"{s}_ALL_{k}.csv" for k in ("indiv", "foreign", "inst")]
        for p in need:
            if not p.exists():
                missing.append(str(p.relative_to(C.RAW)))
    L += [f"## ① 거래일 {len(days)}일 × 파일 6개 = {len(days)*6} 중 누락 {len(missing)}", ""] + ([f"- {m}" for m in missing[:30]] if missing else ["누락 없음"]) + [""]
    close = load_wide("close", a.start, a.end).reindex(days)
    ret = load_wide("ret", a.start, a.end).reindex(days)
    open_ = load_wide("open", a.start, a.end).reindex(days)
    high = load_wide("high", a.start, a.end).reindex(days)
    low = load_wide("low", a.start, a.end).reindex(days)
    vol = load_wide("volume", a.start, a.end).reindex(days)
    # ② 종목 수 연속성
    n = close.notna().sum(axis=1)
    jump = (n.diff().abs() > 50)
    L += ["## ② 날짜별 상장 종목 수 (KOSPI+KOSDAQ)", "", md_table(n.groupby(n.index.year).agg(["min", "max", "mean"]).round(0), 0),
          f"\n전일 대비 50개 넘게 변한 날: {int(jump.sum())}" + (": " + ", ".join(str(d.date()) for d in n.index[jump][:10]) if jump.any() else ""), ""]
    # ③ 등락률 vs 종가비율
    chk = (close / close.shift(1) - 1) - ret
    tot = int(chk.notna().sum().sum()); mism = int((chk.abs() > 0.002).sum().sum())
    by_year = pd.DataFrame({"obs": chk.notna().sum(axis=1).groupby(chk.index.year).sum(), "mismatch": (chk.abs() > 0.002).sum(axis=1).groupby(chk.index.year).sum()})
    by_year["share"] = by_year["mismatch"] / by_year["obs"]
    L += [f"## ③ 등락률 ≠ 종가/전일종가 − 1 (|차| > 0.2%p): {mism} / {tot} = {mism/tot:.4%} — 배당락·분할·유상증자 등 기준가 조정일", "", md_table(by_year, 4), ""]
    # ④ 투자자 세 주체 합 vs 0 (기타법인·기타외국인이 빠져 있어 0 이 아니다; 분포만 기록)
    s = sum(load_wide(k, a.start, a.end).reindex(days) for k in ("indiv_net_val", "foreign_net_val", "inst_net_val"))
    value = load_wide("value", a.start, a.end).reindex(days)
    ratio = (s / value.where(value > 0)).stack()
    L += [f"## ④ (개인+외국인+기관합계 순매수) / 거래대금 분포 — 기타법인·기타외국인을 빼서 0 이 아니지만 ±1 안이어야 한다", "",
          md_table(ratio.describe(percentiles=[.01, .05, .5, .95, .99]).to_frame("ratio").T, 4), f"\n|ratio| > 1 인 관측: {int((ratio.abs() > 1).sum())}", ""]
    # ⑤ 거래정지·한 가격 날
    susp = ((open_ == 0) & (vol == 0)).sum(axis=1)
    one = ((open_ > 0) & (open_ == high) & (high == low) & (low == close)).sum(axis=1)
    L += ["## ⑤ 하루 평균 거래정지 종목 수 / 한 가격 날 종목 수 (연도별)", "",
          md_table(pd.DataFrame({"suspended": susp.groupby(susp.index.year).mean(), "one_price": one.groupby(one.index.year).mean()}).round(1), 1), ""]
    # ⑥ 시가총액 = 종가 × 상장주식수
    mc = load_wide("mktcap", a.start, a.end).reindex(days); sh = load_wide("shares", a.start, a.end).reindex(days)
    rel = ((close * sh) / mc.where(mc > 0) - 1).abs()
    L += [f"## ⑥ |종가×상장주식수 / 시가총액 − 1| > 1%: {int((rel > 0.01).sum().sum())} / {int(rel.notna().sum().sum())}", ""]
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "data_validation.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)[:3000])


if __name__ == "__main__":
    main()
