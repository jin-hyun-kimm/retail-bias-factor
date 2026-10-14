# retail-bias-factor — 개인투자자 행동 편향의 흔적으로 만든 역발상 팩터

개인투자자의 행동 편향(물타기·처분효과·복권형 선호·공포)이 가격에 남긴 흔적을 KRX 공개 데이터로 종목별로 측정해, 역발상 팩터로 쓸 수 있는지 **사전 등록한 절차**로 검증한다. 투자 조언이 아니다.

- 브리프(사전 등록 원문): [`mirae_brief.md`](mirae_brief.md)
- 0단계 데이터 확인 보고: [`docs/00_stage0_data_check.md`](docs/00_stage0_data_check.md)
- 설계 결정 로그 / 사전등록 문서 / 결과 리포트: 단계별로 `docs/` 에 추가

## 재현

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # KRX 정보데이터시스템 ID/PW (무료 회원가입). 캐시가 있으면 로그인 불필요
.venv/bin/python scripts/collect_krx.py --start 20150102 --end 20260930    # 원천 수집 (약 4.5시간, 재실행하면 이어받기)
.venv/bin/python scripts/collect_krx.py --start 20140102 --end 20141230 --what index,ohlcv,cap   # 2014 룩백용 가격·시총
.venv/bin/python scripts/collect_admin_issue.py                             # 관리종목 지정·해제 이력 (KRX [20013])
```

## 데이터 캐시 구조 (`data/raw/`, git 에 넣지 않음)

| 경로 | 내용 | 출처 (pykrx 1.2.9 / KRX 화면) |
|---|---|---|
| `index/kospi_1001_{YYYY}.csv` | KOSPI 지수 일별 OHLCV. 거래일 달력 겸용 | `get_index_ohlcv_by_date` |
| `ohlcv/{YYYY}/{YYYYMMDD}_{KOSPI\|KOSDAQ}.csv` | 그날 상장된 전 종목 시가·고가·저가·종가·거래량·거래대금·등락률·시가총액 (상장폐지 종목 포함) | `get_market_ohlcv_by_ticker` [12001] |
| `netbuy/{YYYY}/{YYYYMMDD}_ALL_{indiv\|foreign\|inst}.csv` | 종목별 개인·외국인·기관합계 매도·매수·순매수 거래량·거래대금 | `get_market_net_purchases_of_equities` [12010] |
| `cap/{YYYY}/{YYYYMMDD}_ALL.csv` | 종목별 종가·시가총액·거래량·거래대금·상장주식수 | `get_market_cap_by_ticker` |
| `etf/{ticker}_{YYYY}.csv` | ETF 투자자별 일자별 순매수 거래대금 (레버리지·인버스 6종) | `get_etf_trading_volume_and_value` |
| `admin/{ticker}_{strt}_{end}.csv`, `admin/_events.csv` | 관리종목 지정·해제·사유변경 이벤트 | KRX [20013] `MDCSTAT21501` (pykrx 미지원, 직접 호출) |
| `collect_log.jsonl`, `collect_admin_log.jsonl` | 호출 기록(시각·초·행 수·오류) | — |

`data/probe/` 는 0단계 실측 결과(샘플 날짜 스냅샷, `probe_log.json`)로 git 에 포함한다.

## 데이터의 함정과 처리 (실측으로 생긴 규칙)

| 함정 | 처리 |
|---|---|
| `market="ALL"` 은 KONEX 를 포함 | KOSPI·KOSDAQ 스냅샷의 티커 집합으로 거른다 |
| 거래정지일은 시가=고가=저가=0, 거래량=0, 등락률=0 | 거래정지 플래그로 쓰고 그날은 유니버스에서 뺀다 |
| [12010] 은 그날 그 투자자가 거래한 종목만 돌려준다 | 없는 종목은 거래 0 으로 본다(결측 아님). 거래정지 종목은 별도 |
| 우선주는 이름 규칙으로 놓치는 것이 있다(전환우선주 `K`/`L` 접미) | 티커 끝자리 ≠ `0` 이면 우선주 |
| 관리종목은 pykrx 에 없다 | KRX [20013] 을 종목별·2년 창으로 호출해 이벤트에서 구간을 복원 |
| 주식분할·배당락 날 가격 불연속 | 수익률은 KRX `등락률`(기준가 조정 반영)로 만들고, 수정 가격은 그 누적으로 만든다 |
| KRX 로그인 세션 1시간 만료 | pykrx 가 자동 재로그인. 받은 CSV 를 캐시해 재현에는 로그인이 필요 없게 |

## 라이선스

MIT
