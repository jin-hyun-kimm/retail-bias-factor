# retail-bias-factor — 개인투자자 행동 편향의 흔적으로 만든 역발상 팩터

개인투자자의 행동 편향(물타기·처분효과·복권형 선호·공포)이 가격에 남긴 흔적을 KRX 공개 데이터로 종목별로 측정해, 역발상 팩터로 쓸 수 있는지 **사전 등록한 절차**로 검증한다. 투자 조언이 아니다.

- 브리프(사전 등록 원문): [`mirae_brief.md`](mirae_brief.md)
- 0단계 데이터 확인 보고: [`docs/00_stage0_data_check.md`](docs/00_stage0_data_check.md)
- 리서치 요약(문헌 3편·데이터 가용성): [`docs/01_research.md`](docs/01_research.md)
- 설계 결정 로그 D1~D24: [`docs/02_design.md`](docs/02_design.md)
- 사전등록 문서(가설·판정 기준·잠금 커밋): [`docs/03_preregistration.md`](docs/03_preregistration.md)
- 결과: `results/stage1/summary.md`(탐색), `results/stage2/summary.md`(홀드아웃), `results/stage3/`(ML·오버레이), `results/report.html`, 그림 `figures/`

## 파이프라인

| 단계 | 명령 | 산출물 |
|---|---|---|
| 수집 | `scripts/collect_krx.py`, `scripts/collect_admin_issue.py` (순차 실행은 `scripts/run_collection_chain.sh`) | `data/raw/` |
| 가공 | `python -m rbf.io_raw` | `data/processed/{연도}/{변수}.parquet` |
| 검증 | `scripts/validate_data.py`, `pytest` | `results/data_validation.md`, 테스트 12개 |
| 1단계 탐색 | `scripts/run_stage1.py` | `results/stage1/` (5분위·FM·비용·소구간·플라시보·MDE) |
| 2단계 홀드아웃 | `scripts/run_holdout.py --note ...` (잠금 후 1회; 실행마다 `results/holdout_runs.jsonl` 기록) | `results/stage2/` + 판정 |
| 3단계 | `scripts/run_stage3_ml.py`, `scripts/run_stage3_overlay.py` | `results/stage3/` |
| 4단계 | `scripts/make_figures.py`, `scripts/make_report.py` | `figures/`, `results/report.html` |
| 전부 | `scripts/run_all.sh` | 위 전체 |

코드는 `src/rbf/`: `config`(상수) · `io_raw`(원천→parquet) · `signals`(지표) · `returns`(t+1 시가 체결 보유 수익률) · `universe`(제외 규칙·관리종목 복원) · `fm`(윈저라이즈·Fama–MacBeth·MDE) · `portfolio`(5분위·비용) · `analysis`(가설별 검정).

## 결과 한눈에 (사전등록 판정, 홀드아웃 1회)

| 가설 | 탐색(131개월) | 홀드아웃(69개월) | 판정 |
|---|---|---|---|
| H1 물타기 역발상 (급락 종목 안 개인 순매수 강도 → 수익률 ↓) | 계수 +0.0046, t 4.51 (부호 반대) | +0.0031, t 2.02 | **기각** |
| H2 처분효과 압력 (CGO ↑ → 수익률 ↑) | +0.0008, t 0.64 | +0.0010, t 0.43 | **기각** |
| H3 복권형 매수 압력 (MAX × 개인 순매수 → 수익률 ↓) | +0.0013, t 1.82 (부호 반대) | +0.0013, t 0.77 | **기각** |
| H5 공포 오버레이 | z≥2 사건 20건, 20일 후 KOSPI +2.85% vs 평소 +0.99% (t 1.41) | 오버레이 −0.24%p/월 (t −1.02) | 지지되지 않음 |

ML(Ridge·LightGBM, walk-forward)은 세 신호 합산을 IC 기준으로 크게 이겼지만, 그 우위는 회전율·규모·MAX·개인 거래 비중과 **가설과 반대 부호의 개인 순매수**에서 나왔다. 사전등록한 팩터의 성공이 아니다. 자세한 것은 [`docs/08_limitations.md`](docs/08_limitations.md), 결과 리포트 `results/report.html`, 단계 보고 [`docs/04_stage1_report.md`](docs/04_stage1_report.md).

## 재현

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # KRX 정보데이터시스템 ID/PW (무료 회원가입). 캐시가 있으면 로그인 불필요
.venv/bin/python scripts/collect_krx.py --start 20150102 --end 20260930    # 원천 수집 (약 4.5시간, 재실행하면 이어받기)
.venv/bin/python scripts/collect_krx.py --start 20140102 --end 20141230 --what index,ohlcv,cap   # 2014 룩백용 가격·시총
.venv/bin/python scripts/collect_admin_issue.py                             # 관리종목 지정·해제 이력 (KRX [20013])
```

## 데이터 출처와 KRX 이용약관 (중요)

이 연구의 원천 데이터는 KRX 정보데이터시스템(data.krx.co.kr)에서 pykrx 로 받았다. 수집 중 KRX 가 "자동화 수단을 통한 비정상 대량 조회"를 이유로 IP 를 1일 제한했고, 안내문은 이용약관 제10조 제2호가 자동화 수단을 이용한 정보 수집을 금지한다고 밝혔다. 따라서 **이 저장소의 수집 스크립트를 대량으로 실행하는 것을 권하지 않는다.** 재현에는 (1) 공개 릴리스로 제공하는 가공 캐시(`data/processed/`)를 쓰거나, (2) KRX 가 안내한 공식 경로(화면 다운로드, 데이터 상품, [KRX Open API](https://openapi.krx.co.kr))를 쓴다. 관리종목 이력은 이 제한으로 받지 못해 유니버스에 적용하지 못했다(설계 D25).

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
