"""0단계 데이터 확인 — 샘플 날짜에서 pykrx 네 데이터가 실제로 받아지는지 실측한다.
결과는 data/probe/ 에 CSV·JSON 으로 남긴다. KRX 로그인은 .env 의 KRX_ID/KRX_PW.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # pykrx 가 import 시점에 env 를 읽으므로 먼저
import pandas as pd  # noqa: E402
from pykrx import stock  # noqa: E402

OUT = ROOT / "data" / "probe"
OUT.mkdir(parents=True, exist_ok=True)
SLEEP = 0.4
SAMPLE_DATES = ["20150102", "20150105", "20180103", "20200320", "20230102", "20260930"]
ETFS = {
    "122630": "KODEX 레버리지",
    "252670": "KODEX 200선물인버스2X",
    "114800": "KODEX 인버스",
    "123320": "TIGER 레버리지",
    "253230": "TIGER 200선물인버스2X",
    "123310": "TIGER 인버스",
}
log: list[dict] = []


def timed(label: str, fn, *a, **k):
    t0 = time.time()
    try:
        r = fn(*a, **k)
        dt = time.time() - t0
        n = len(r) if hasattr(r, "__len__") else None
        cols = list(map(str, r.columns)) if hasattr(r, "columns") else None
        log.append({"call": label, "ok": True, "sec": round(dt, 2), "rows": n, "cols": cols})
        print(f"[OK ] {label}: rows={n} sec={dt:.2f} cols={cols}")
        time.sleep(SLEEP)
        return r
    except Exception as e:  # noqa: BLE001
        dt = time.time() - t0
        log.append({"call": label, "ok": False, "sec": round(dt, 2), "err": repr(e)[:300]})
        print(f"[ERR] {label}: {e!r}"[:400])
        time.sleep(SLEEP)
        return None


def main() -> None:
    print("KRX_ID set:", bool(os.getenv("KRX_ID")), "| KRX_PW set:", bool(os.getenv("KRX_PW")))

    # 1) 가격·거래량 스냅샷 (ALL 이 되는지, KOSPI/KOSDAQ 분리가 필요한지)
    for d in SAMPLE_DATES:
        for mkt in ["ALL", "KOSPI", "KOSDAQ"]:
            df = timed(f"ohlcv_by_ticker {d} {mkt}", stock.get_market_ohlcv_by_ticker, d, market=mkt)
            if df is not None and len(df):
                df.to_csv(OUT / f"ohlcv_{d}_{mkt}.csv", encoding="utf-8-sig")

    # 2) 종목별 투자자 순매수 — 하루 단위 (개인·외국인·기관합계), ALL 시도
    for d in SAMPLE_DATES:
        for inv in ["개인", "외국인", "기관합계"]:
            for mkt in ["ALL", "KOSPI", "KOSDAQ"]:
                df = timed(
                    f"net_purchases {d} {mkt} {inv}",
                    stock.get_market_net_purchases_of_equities, d, d, mkt, inv,
                )
                if df is not None and len(df):
                    df.to_csv(OUT / f"netbuy_{d}_{mkt}_{inv}.csv", encoding="utf-8-sig")

    # 3) 시가총액·상장주식수
    for d in SAMPLE_DATES:
        df = timed(f"market_cap_by_ticker {d} ALL", stock.get_market_cap_by_ticker, d, market="ALL")
        if df is not None and len(df):
            df.to_csv(OUT / f"cap_{d}_ALL.csv", encoding="utf-8-sig")

    # 4) ETF 투자자별 일자별 순매수 거래대금 — 종목별, 2015 초부터 받아지는지
    for tk, nm in ETFS.items():
        df = timed(
            f"etf_invest_by_date {tk} {nm} 2015H1",
            stock.get_etf_trading_volume_and_value, "20150102", "20150630", tk, "거래대금", "순매수",
        )
        if df is not None and len(df):
            df.to_csv(OUT / f"etf_{tk}_2015H1.csv", encoding="utf-8-sig")
        df = timed(
            f"etf_invest_by_date {tk} {nm} 2016H2",
            stock.get_etf_trading_volume_and_value, "20160901", "20161231", tk, "거래대금", "순매수",
        )
        if df is not None and len(df):
            df.to_csv(OUT / f"etf_{tk}_2016H2.csv", encoding="utf-8-sig")
        df = timed(
            f"etf_invest_by_date {tk} {nm} 2026Q3",
            stock.get_etf_trading_volume_and_value, "20260701", "20260930", tk, "거래대금", "순매수",
        )
        if df is not None and len(df):
            df.to_csv(OUT / f"etf_{tk}_2026Q3.csv", encoding="utf-8-sig")

    # 5) KOSPI 지수 (벤치마크·심리 국면)
    df = timed("index_ohlcv KOSPI 2015", stock.get_index_ohlcv_by_date, "20150101", "20151231", "1001")
    if df is not None and len(df):
        df.to_csv(OUT / "kospi_2015.csv", encoding="utf-8-sig")
    df = timed("index_ohlcv KOSPI 2026", stock.get_index_ohlcv_by_date, "20260101", "20260930", "1001")
    if df is not None and len(df):
        df.to_csv(OUT / "kospi_2026.csv", encoding="utf-8-sig")

    # 6) 상장폐지 종목 포함 여부 — 한진해운(117930, 2017-03 상폐), 삼성물산(000830, 2015-09 합병소멸)
    df = pd.read_csv(OUT / "ohlcv_20150102_ALL.csv", dtype={"티커": str}) if (OUT / "ohlcv_20150102_ALL.csv").exists() else None
    if df is not None:
        col = df.columns[0]
        s = set(df[col].astype(str).str.zfill(6))
        print("delisted check 2015-01-02:", {t: (t in s) for t in ["117930", "000830", "005930"]})
        log.append({"call": "delisted_check_20150102", "ok": True, "detail": {t: (t in s) for t in ["117930", "000830", "005930"]}})

    (OUT / "probe_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for r in log if r.get("ok"))
    print(f"\nDONE: {ok}/{len(log)} ok. log -> {OUT/'probe_log.json'}")


if __name__ == "__main__":
    main()
