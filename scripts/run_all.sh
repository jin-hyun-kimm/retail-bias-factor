#!/bin/bash
# 수집 → 가공 → 테스트 → 1단계(탐색) → [잠금 후] 2단계(홀드아웃) → 3단계(ML·오버레이) → 4단계(그림·리포트).
# 캐시가 있으면 수집은 건너뛴다. 홀드아웃은 사전등록 문서에 잠금 커밋이 있어야 돈다.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/collect_krx.py --start 20090102 --end 20260930
$PY scripts/collect_admin_issue.py
$PY -m rbf.io_raw
$PY -m pytest -q
$PY scripts/run_stage1.py
if grep -q "잠금 커밋: [0-9a-f]" docs/03_preregistration.md; then
  $PY scripts/run_holdout.py --note "run_all.sh"
  $PY scripts/run_stage3_ml.py
  $PY scripts/run_stage3_overlay.py
  $PY scripts/make_figures.py
  $PY scripts/make_report.py
fi
