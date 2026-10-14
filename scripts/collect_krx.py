"""KRX 원천 데이터 수집 — 날짜별 스냅샷을 data/raw 에 CSV 로 캐시한다. 재실행하면 있는 파일은 건너뛴다.

캐시 구조 (README 에도 적는다):
  data/raw/index/kospi_1001_{YYYY}.csv              KOSPI 지수 일별 OHLCV (거래일 달력 겸용)
  data/raw/ohlcv/{YYYY}/{YYYYMMDD}_{KOSPI|KOSDAQ}.csv  전 종목 시가·고가·저가·종가·거래량·거래대금·등락률·시가총액
  data/raw/netbuy/{YYYY}/{YYYYMMDD}_ALL_{indiv|foreign|inst}.csv  종목별 투자자 매도·매수·순매수 (거래량·거래대금)
  data/raw/cap/{YYYY}/{YYYYMMDD}_ALL.csv             종목별 종가·시가총액·거래량·거래대금·상장주식수
  data/raw/etf/{ticker}_{YYYY}.csv                   ETF 투자자별 일자별 순매수 거래대금
  data/raw/collect_log.jsonl                         호출 기록(시각·초·행 수·오류)

사용: .venv/bin/python scripts/collect_krx.py --start 20150102 --end 20260930 [--what index,ohlcv,netbuy,cap,etf] [--sleep 0.5]
KRX 로그인은 .env 의 KRX_ID/KRX_PW (pykrx 가 import 시점에 읽는다).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
import pandas as pd  # noqa: E402
from pykrx import stock  # noqa: E402

RAW = ROOT / "data" / "raw"
LOG = RAW / "collect_log.jsonl"
INVESTORS = {"indiv": "개인", "foreign": "외국인", "inst": "기관합계"}
ETFS = ["122630", "114800", "123320", "123310", "252670", "253230"]  # 설계 문서에서 4종 확정 전까지 6종 모두 받는다
MAX_CONSEC_FAIL = 5


def log(rec: dict) -> None:
    rec["ts"] = datetime.now().isoformat(timespec="seconds")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class Collector:
    def __init__(self, sleep: float) -> None:
        self.sleep = sleep
        self.consec_fail = 0
        self.n_call = 0
        self.n_skip = 0

    def fetch(self, label: str, path: Path, fn, *args, allow_empty: bool = False) -> pd.DataFrame | None:
        if path.exists():
            self.n_skip += 1
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, 4):
            t0 = time.time()
            try:
                df = fn(*args)
                dt = time.time() - t0
                if df is None or (len(df) == 0 and not allow_empty):
                    raise RuntimeError("empty result on a trading day")
                df.to_csv(path, encoding="utf-8-sig")
                log({"call": label, "ok": True, "sec": round(dt, 2), "rows": int(len(df)), "attempt": attempt})
                self.consec_fail = 0
                self.n_call += 1
                time.sleep(self.sleep)
                return df
            except Exception as e:  # noqa: BLE001
                dt = time.time() - t0
                log({"call": label, "ok": False, "sec": round(dt, 2), "attempt": attempt, "err": repr(e)[:300]})
                print(f"[ERR] {label} attempt {attempt}: {e!r}"[:300], file=sys.stderr)
                time.sleep(5 * attempt)
        self.consec_fail += 1
        if self.consec_fail >= MAX_CONSEC_FAIL:
            print(f"[ABORT] {MAX_CONSEC_FAIL} consecutive failures — KRX 차단 가능성. 잠시 뒤 재실행한다.", file=sys.stderr)
            sys.exit(2)
        return None


def trading_days(c: Collector, start: str, end: str) -> list[str]:
    frames = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        p = RAW / "index" / f"kospi_1001_{y}.csv"
        yend = f"{y}1231"
        # 파일이 요청 구간 끝(과거 연도는 연말)까지 덮지 못하면 다시 받는다 (시험 실행의 부분 파일 재사용 방지)
        need_until = min(yend, end)
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True, encoding="utf-8-sig")
            have_until = df.index.max().strftime("%Y%m%d") if len(df) else "00000000"
            # 연말 마지막 거래일은 12-28~31 사이이므로 12-20 이후면 완결로 본다
            complete = have_until >= (f"{y}1220" if need_until == yend else need_until)
            if not complete:
                p.unlink()
        c.fetch(f"index 1001 {y}", p, stock.get_index_ohlcv_by_date, f"{y}0101", min(yend, end), "1001")
        df = pd.read_csv(p, index_col=0, parse_dates=True, encoding="utf-8-sig")
        frames.append(df)
    idx = pd.concat(frames)
    days = [d.strftime("%Y%m%d") for d in idx.index if start <= d.strftime("%Y%m%d") <= end]
    return days


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20150102")
    ap.add_argument("--end", default="20260930")
    ap.add_argument("--what", default="index,ohlcv,netbuy,cap,etf")
    ap.add_argument("--sleep", type=float, default=0.5)
    a = ap.parse_args()
    what = set(a.what.split(","))
    c = Collector(a.sleep)
    RAW.mkdir(parents=True, exist_ok=True)

    days = trading_days(c, a.start, a.end)
    print(f"trading days {a.start}~{a.end}: {len(days)}")

    if "etf" in what:
        for tk in ETFS:
            for y in range(int(a.start[:4]), int(a.end[:4]) + 1):
                p = RAW / "etf" / f"{tk}_{y}.csv"
                if y == int(a.end[:4]) and p.exists():
                    df = pd.read_csv(p, index_col=0, parse_dates=True, encoding="utf-8-sig")
                    if len(df) and df.index.max().strftime("%Y%m%d") < min(f"{y}1231", a.end):
                        p.unlink()
                c.fetch(f"etf {tk} {y}", p, stock.get_etf_trading_volume_and_value,
                        f"{y}0101", min(f"{y}1231", a.end), tk, "거래대금", "순매수", allow_empty=True)

    t_start = time.time()
    for i, d in enumerate(days, 1):
        y = d[:4]
        if "ohlcv" in what:
            for mkt in ("KOSPI", "KOSDAQ"):
                c.fetch(f"ohlcv {d} {mkt}", RAW / "ohlcv" / y / f"{d}_{mkt}.csv", stock.get_market_ohlcv_by_ticker, d, mkt)
        if "netbuy" in what:
            for key, inv in INVESTORS.items():
                c.fetch(f"netbuy {d} ALL {key}", RAW / "netbuy" / y / f"{d}_ALL_{key}.csv",
                        stock.get_market_net_purchases_of_equities, d, d, "ALL", inv)
        if "cap" in what:
            c.fetch(f"cap {d} ALL", RAW / "cap" / y / f"{d}_ALL.csv", stock.get_market_cap_by_ticker, d, "ALL")
        if i % 20 == 0 or i == len(days):
            el = time.time() - t_start
            print(f"[{i}/{len(days)}] {d} calls={c.n_call} skipped={c.n_skip} elapsed={el/60:.1f}min", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
