"""고정 상수. 설계 문서 docs/02_design.md 의 D번호와 대응한다. 격자 탐색 금지 — 값을 바꾸면 결정 번호를 새로 받는다."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# D2 기간
LOOKBACK_START = "2009-01-02"   # 룩백 전용(신호 계산에만 씀)
EVAL_START = "2010-01-04"       # 평가 시작(첫 리밸런싱일은 2010-01 말)
EXPLORE_END = "2020-12-31"      # 탐색 = 보유 월 ≤ 2020-12 (리밸런싱일 ≤ 2020-11-30)
HOLDOUT_START = "2021-01-01"    # 홀드아웃 = 보유 월 2021-01 ~ 2026-09
EVAL_END = "2026-09-30"

# D11~D15 지표 윈도우 (하나씩만)
RNB_WINDOW = 5
RNB_MIN_DAYS = 3
RSH_WINDOW = 20
RSH_MIN_DAYS = 10
CRASH_WINDOW = 20
CRASH_BOTTOM_PCT = 0.20
CGO_WINDOW = 260
CGO_MIN_DAYS = 130
TURNOVER_CAP = 0.999
MAX_WINDOW = 20
SENTIMENT_WINDOW = 126
FEAR_SUM_WINDOW = 20
FEAR_BASE_WINDOW = 250
FEAR_BASE_MIN_DAYS = 120
MOM_LONG = 252
MOM_SKIP = 21
TURNOVER_CTRL_WINDOW = 20

# D15 ETF 4종
LEVERAGE_ETFS = ["122630", "123320"]   # KODEX 레버리지, TIGER 레버리지
INVERSE_ETFS = ["114800", "123310"]    # KODEX 인버스, TIGER 인버스
ETF_NAMES = {"122630": "KODEX 레버리지", "123320": "TIGER 레버리지", "114800": "KODEX 인버스", "123310": "TIGER 인버스"}

# D9 유니버스
SIZE_BOTTOM_PCT = 0.20
VALUE_BOTTOM_PCT = 0.20

# D16 표준화
WINSOR = (0.01, 0.99)

# D18 분위
N_QUANTILES = 5

# D19 Newey–West
NW_LAGS = 3
T_PREREG = 2.0
T_HLZ = 3.0

# D20 비용 (편도, 소수)
COST_BASE = 0.0015
COST_SENS = [0.0010, 0.0025]
COMMISSION_SLIPPAGE = 0.0005   # 실제 세율 시나리오의 매수·매도 각 가정
TAX_SCHEDULE = [            # (적용 시작일, 매도 시 증권거래세+농특세)
    ("2000-01-01", 0.0030),
    ("2019-06-03", 0.0025),
    ("2021-01-01", 0.0023),
    ("2023-01-01", 0.0020),
    ("2024-01-01", 0.0018),
    ("2025-01-01", 0.0015),
]

# D22 오버레이
OVERLAY_SLOPE = 0.25
OVERLAY_MIN, OVERLAY_MAX = 0.5, 1.5
FEAR_EVENT_Z = 2.0
FEAR_EVENT_COOLDOWN = 20
FEAR_MIN_EVENTS = 10

# D21 ML
ML_FIRST_TRAIN_END = "2012-12-31"

# D23 어휘 게이트
FORBIDDEN_WORDS = ["예측", "추천", "전망", "유망", "목표가"]
