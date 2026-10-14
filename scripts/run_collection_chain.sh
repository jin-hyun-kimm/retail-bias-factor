#!/bin/bash
# 수집 대기열 — KRX 동시 호출을 피하려고 순차 실행한다. 재실행해도 받은 파일은 건너뛴다.
# 1) 2015-01-02~2026-09-30 본 수집 (이미 돌고 있으면 끝날 때까지 기다린다)
# 2) 2009-01-02~2014-12-30 확장 수집 (평가 기간을 2010-01 부터로 늘림; 2009 는 CGO·모멘텀 룩백용)
# 3) 관리종목 지정·해제 이력 (KRX [20013])
cd "$(dirname "$0")/.."
while pgrep -f "collect_krx.py --start 20150102" >/dev/null; do sleep 60; done
.venv/bin/python scripts/collect_krx.py --start 20090102 --end 20141230 --sleep 0.5 > data/raw/collect_2009_2014.out 2>&1
.venv/bin/python scripts/collect_admin_issue.py --sleep 0.5 > data/raw/collect_admin.out 2>&1
echo "CHAIN DONE $(date)" >> data/raw/collect_chain.out
