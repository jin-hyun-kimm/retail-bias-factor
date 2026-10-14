"""4단계 그림 4장 — 결과 CSV 에서만 그린다 (브리프 6절).
1 가설별 5분위 수익률  2 누적 수익률(롱온리 vs 벤치마크, 비용 반영)  3 ML vs 합산  4 공포 z 사건 연구
사용: .venv/bin/python scripts/make_figures.py [--stage1 results/stage1 --stage2 results/stage2 --stage3 results/stage3 --out figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for cand in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
COLORS = {"Q": "#4C72B0", "bench": "#999999", "kospi": "#DD8452", "univ": "#55A868", "zsum": "#4C72B0", "ridge": "#DD8452", "lgbm": "#55A868"}
HNAMES = {"h1": "H1 물타기 역발상", "h2": "H2 처분효과 압력", "h3": "H3 복권형 매수 압력"}


def _read(p: Path, **k):
    return pd.read_csv(p, index_col=0, encoding="utf-8-sig", **k) if p.exists() else None


def fig_quintiles(s1: Path, s2: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, h in zip(axes, ["h1", "h2", "h3"]):
        q1 = _read(s1 / f"{h}_quintile_summary.csv")
        q2 = _read(s2 / f"{h}_quintile_summary.csv")
        qs = [f"Q{i}" for i in range(1, 6)]
        x = np.arange(5)
        if q1 is not None:
            ax.bar(x - 0.18, q1.loc[qs, "ew_mean"] * 100, 0.36, label="탐색 2010-02~2020-12", color=COLORS["Q"])
        if q2 is not None:
            ax.bar(x + 0.18, q2.loc[qs, "ew_mean"] * 100, 0.36, label="홀드아웃 2021-01~2026-09", color=COLORS["kospi"])
        ax.set_xticks(x, qs)
        ax.set_title(HNAMES[h])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("신호 5분위 (Q5 = 가설 방향 상위)")
    axes[0].set_ylabel("월평균 수익률 (%, 동일가중, t+1 시가 체결)")
    axes[0].legend(fontsize=8)
    fig.suptitle("그림 1. 가설별 5분위 월평균 수익률")
    fig.tight_layout()
    fig.savefig(out / "fig1_quintiles.png", dpi=150)
    plt.close(fig)


def fig_cumulative(s1: Path, s2: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, h in zip(axes, ["h1", "h2", "h3"]):
        parts = [p for p in [_read(s1 / f"{h}_longonly.csv", parse_dates=True), _read(s2 / f"{h}_longonly.csv", parse_dates=True)] if p is not None]
        if not parts:
            continue
        lo = pd.concat(parts).sort_index()
        lo = lo[~lo.index.duplicated()]
        for col, lab, c in [("net_base", "롱온리 Q5 (비용 0.15%)", COLORS["Q"]), ("net_real", "롱온리 Q5 (실제 세율)", "#8172B2"),
                            ("kospi", "KOSPI", COLORS["kospi"]), ("univ_ew", "유니버스 동일가중", COLORS["univ"])]:
            if col in lo:
                ax.plot(lo.index, (1 + lo[col].fillna(0)).cumprod(), label=lab, color=c, lw=1.4)
        if len(parts) == 2:
            ax.axvline(pd.Timestamp("2020-12-30"), color="k", ls="--", lw=0.8)
            ax.text(pd.Timestamp("2020-12-30"), ax.get_ylim()[1] * 0.95, " 홀드아웃 →", fontsize=8, va="top")
        ax.set_title(HNAMES[h])
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("누적 (로그 눈금, 시작 = 1)")
    axes[0].legend(fontsize=8)
    fig.suptitle("그림 2. 롱온리 Q5 누적 수익률 vs 벤치마크 (월별 리밸런싱, t+1 시가 체결, 비용 반영)")
    fig.tight_layout()
    fig.savefig(out / "fig2_cumulative.png", dpi=150)
    plt.close(fig)


def fig_ml(s3: Path, out: Path) -> None:
    ic = _read(s3 / "ml_ic_monthly.csv", parse_dates=True)
    if ic is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for m, lab in [("zsum", "세 신호 z-score 합산"), ("ridge", "Ridge"), ("lgbm", "LightGBM")]:
        axes[0].plot(ic.index, ic[m].rolling(12).mean(), label=lab, color=COLORS[m])
        lo = _read(s3 / f"ml_longonly_{m}.csv", parse_dates=True)
        if lo is not None:
            axes[1].plot(lo.index, (1 + lo["net_base"].fillna(0)).cumprod(), label=lab, color=COLORS[m])
    lo = _read(s3 / "ml_longonly_zsum.csv", parse_dates=True)
    if lo is not None:
        axes[1].plot(lo.index, (1 + lo["univ_ew"].fillna(0)).cumprod(), label="유니버스 동일가중", color=COLORS["bench"], ls="--")
    for ax in axes:
        ax.axvline(pd.Timestamp("2020-12-30"), color="k", ls="--", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title("월별 IC (Spearman) 12개월 이동평균")
    axes[1].set_title("상위 5분위 롱온리 누적 (비용 0.15%)")
    axes[1].set_yscale("log")
    fig.suptitle("그림 3. ML(기본값, walk-forward) vs 단순 합산")
    fig.tight_layout()
    fig.savefig(out / "fig3_ml_vs_zsum.png", dpi=150)
    plt.close(fig)


def fig_fear(s3: Path, out: Path) -> None:
    fz = _read(s3 / "overlay_fear_z_daily.csv", parse_dates=True)
    ev = pd.read_csv(s3 / "overlay_events.csv", encoding="utf-8-sig", parse_dates=["event"]) if (s3 / "overlay_events.csv").exists() else None
    es = _read(s3 / "overlay_event_study.csv")
    if fz is None or es is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    axes[0].plot(fz.index, fz["z"], lw=0.8, color=COLORS["Q"])
    axes[0].axhline(2, color="r", ls="--", lw=0.8)
    if ev is not None and len(ev):
        axes[0].scatter(ev["event"], [2.0] * len(ev), color="r", s=18, zorder=3, label=f"z≥2 사건 {len(ev)}건")
        axes[0].legend(fontsize=8)
    axes[0].axvline(pd.Timestamp("2020-12-30"), color="k", ls="--", lw=0.8)
    axes[0].set_title("공포 z (인버스 - 레버리지 ETF 개인 순매수 20일 합, 250일 표준화)")
    axes[0].grid(alpha=0.3)
    x = np.arange(len(es))
    axes[1].bar(x - 0.18, es["event_mean"] * 100, 0.36, label="사건 후 평균", color="r")
    axes[1].bar(x + 0.18, es["all_days_mean"] * 100, 0.36, label="모든 날 평균", color=COLORS["bench"])
    axes[1].set_xticks(x, [i.replace("ret_", "").replace("d", "거래일") for i in es.index])
    axes[1].set_ylabel("이후 KOSPI 수익률 (%)")
    axes[1].set_title(f"사건 연구 (n={int(es['event_n'].max())})")
    axes[1].legend(fontsize=8)
    axes[1].axhline(0, color="k", lw=0.5)
    fig.suptitle("그림 4. 공포 z 사건 연구")
    fig.tight_layout()
    fig.savefig(out / "fig4_fear_events.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=str(ROOT / "results" / "stage1"))
    ap.add_argument("--stage2", default=str(ROOT / "results" / "stage2"))
    ap.add_argument("--stage3", default=str(ROOT / "results" / "stage3"))
    ap.add_argument("--out", default=str(ROOT / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_quintiles(Path(a.stage1), Path(a.stage2), out)
    fig_cumulative(Path(a.stage1), Path(a.stage2), out)
    fig_ml(Path(a.stage3), out)
    fig_fear(Path(a.stage3), out)
    print("figures ->", out, sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
