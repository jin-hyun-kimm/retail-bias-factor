"""관리종목 지정·해제 이력 수집 — KRX [20013] 관리종목 지정 내역(개별종목), bld=MDCSTAT21501.

pykrx 에 없는 화면이라 pykrx 의 로그인 세션으로 직접 호출한다.
- isuCd 는 ISIN(KR7+티커+00+체크숫자) 만 받는다. 6자리 코드는 빈 결과.
- 조회 기간은 호출당 최대 2년(730일). 그래서 2년 창으로 나눠 부른다.
- designDdYn=Y 면 이벤트(관리종목지정·해제·사유변경)만 돌려준다.

대상 티커: data/raw/ohlcv 의 KOSPI·KOSDAQ 스냅샷에 한 번이라도 나온 보통주(끝자리 '0', 스팩 제외).
창: 2010-01-04 이전부터 상장돼 있던 종목은 2006-01-01 부터(2010 이전 지정분이 2010 이후 해제되는 경우를 잡기 위해),
    그 뒤 상장한 종목은 처음 보인 날부터, 마지막으로 보인 날까지.

출력: data/raw/admin/{ticker}_{strt}_{end}.csv (이벤트 0건이면 헤더만 있는 빈 파일 — 재실행 시 건너뛰기 위함)
      data/raw/admin/_events.csv  (전부 합친 이벤트 표: ticker, date, change_type, reason)
사용: .venv/bin/python scripts/collect_admin_issue.py [--sleep 0.5] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
from pykrx.website.comm.webio import get_session  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = RAW / "admin"
LOG = RAW / "collect_admin_log.jsonl"
URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
BLD = "dbms/MDC/STAT/issue/MDCSTAT21501"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC02020702",
    "X-Requested-With": "XMLHttpRequest",
}
BACKFILL_FROM = "20060101"
PERIOD_START = "20100104"
WINDOW_DAYS = 729  # 2년 한도 안쪽


def isin_kr(ticker6: str) -> str:
    base = "KR7" + ticker6 + "00"
    digits = "".join(str(ord(c) - 55) if c.isalpha() else c for c in base)
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return base + str((10 - total % 10) % 10)


def ticker_spans() -> pd.DataFrame:
    """ohlcv 스냅샷에서 티커별 처음·마지막 관측일. 보통주(끝자리 0)만."""
    files = sorted((RAW / "ohlcv").glob("*/*_KOSPI.csv")) + sorted((RAW / "ohlcv").glob("*/*_KOSDAQ.csv"))
    first: dict[str, str] = {}
    last: dict[str, str] = {}
    names: dict[str, str] = {}
    for f in files:
        d = f.name[:8]
        tk = pd.read_csv(f, usecols=[0], dtype=str, encoding="utf-8-sig").iloc[:, 0].str.zfill(6)
        for t in tk:
            if t[-1] != "0":
                continue
            if t not in first or d < first[t]:
                first[t] = d
            if t not in last or d > last[t]:
                last[t] = d
    # 이름(스팩 제외용)은 개인 순매수 파일의 종목명에서
    for f in sorted((RAW / "netbuy").glob("*/*_ALL_indiv.csv"))[::40]:
        nb = pd.read_csv(f, usecols=[0, 1], dtype=str, encoding="utf-8-sig")
        for t, n in zip(nb.iloc[:, 0].str.zfill(6), nb.iloc[:, 1]):
            names.setdefault(t, n)
    rows = [{"ticker": t, "first": first[t], "last": last[t], "name": names.get(t, "")} for t in first]
    df = pd.DataFrame(rows)
    df = df[~df["name"].str.contains("스팩", na=False)].reset_index(drop=True)
    return df


def windows(first: str, last: str) -> list[tuple[str, str]]:
    start = BACKFILL_FROM if first <= PERIOD_START else first
    s = pd.Timestamp(start)
    end = pd.Timestamp(last)
    out = []
    while s <= end:
        e = min(s + pd.Timedelta(WINDOW_DAYS, unit="D"), end)
        out.append((s.strftime("%Y%m%d"), e.strftime("%Y%m%d")))
        s = e + pd.Timedelta(1, unit="D")
    return out


def log(rec: dict) -> None:
    rec["ts"] = datetime.now().isoformat(timespec="seconds")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="시험용: 티커 수 제한")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spans = ticker_spans()
    if a.limit:
        spans = spans.head(a.limit)
    jobs = [(r.ticker, s, e) for r in spans.itertuples() for (s, e) in windows(r.first, r.last)]
    print(f"tickers={len(spans)} windows(calls)={len(jobs)}")
    n_call = n_skip = consec_fail = 0
    t0 = time.time()
    for i, (tk, s, e) in enumerate(jobs, 1):
        p = OUT / f"{tk}_{s}_{e}.csv"
        if p.exists():
            n_skip += 1
            continue
        data = {"bld": BLD, "locale": "ko_KR", "isuCd": isin_kr(tk), "strtDd": s, "endDd": e,
                "designDdYn": "Y", "share": "1", "money": "1", "csvxls_isNo": "false"}
        ok = False
        for attempt in range(1, 4):
            try:
                sess = get_session()  # 1시간 만료 시 pykrx 가 자동 재로그인하도록 매 요청마다 얻는다
                r = sess.post(URL, data=data, headers=HEADERS, timeout=30)
                j = r.json()
                rows = j.get("output", [])
                df = pd.DataFrame(rows, columns=["TRD_DD", "ISU_CD", "ISU_NM", "MKT_NM", "RSN_CHG_TP_NM", "LIST_BZ_RSN_NM"]) if not rows else pd.DataFrame(rows)[["TRD_DD", "ISU_CD", "ISU_NM", "MKT_NM", "RSN_CHG_TP_NM", "LIST_BZ_RSN_NM"]]
                df.to_csv(p, index=False, encoding="utf-8-sig")
                log({"call": f"admin {tk} {s} {e}", "ok": True, "rows": len(df), "attempt": attempt})
                ok = True
                n_call += 1
                consec_fail = 0
                time.sleep(a.sleep)
                break
            except Exception as ex:  # noqa: BLE001
                log({"call": f"admin {tk} {s} {e}", "ok": False, "attempt": attempt, "err": repr(ex)[:300]})
                print(f"[ERR] {tk} {s}-{e} attempt {attempt}: {ex!r}"[:200], file=sys.stderr)
                time.sleep(5 * attempt)
        if not ok:
            consec_fail += 1
            if consec_fail >= 5:
                print("[PAUSE] 5 consecutive failures — KRX 점검·차단 가능성. 10분 뒤 계속", file=sys.stderr)
                time.sleep(600)
                consec_fail = 0
                pauses = globals().get('_pauses', 0) + 1
                globals()['_pauses'] = pauses
                if pauses >= 12:
                    print("[ABORT] 2시간 넘게 실패", file=sys.stderr)
                    sys.exit(2)
        if i % 200 == 0 or i == len(jobs):
            print(f"[{i}/{len(jobs)}] calls={n_call} skipped={n_skip} elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    # 합치기
    parts = []
    for f in OUT.glob("*_*_*.csv"):
        df = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
        if len(df):
            parts.append(df)
    if parts:
        ev = pd.concat(parts, ignore_index=True)
        ev = ev.rename(columns={"TRD_DD": "date", "ISU_CD": "ticker", "ISU_NM": "name", "MKT_NM": "market",
                                "RSN_CHG_TP_NM": "change_type", "LIST_BZ_RSN_NM": "reason"})
        ev["date"] = ev["date"].str.replace("/", "")
        ev = ev.drop_duplicates().sort_values(["ticker", "date"]).reset_index(drop=True)
        ev.to_csv(OUT / "_events.csv", index=False, encoding="utf-8-sig")
        print(f"events: {len(ev)} rows, tickers={ev['ticker'].nunique()}; types={ev['change_type'].value_counts().to_dict()}")
    print("DONE")


if __name__ == "__main__":
    main()
