#!/bin/bash
# KRX 가 점검·차단으로 응답하지 않을 때: 10분마다 건강 확인 후 관리종목 수집기를 다시 돌린다(받은 파일은 건너뜀). DONE 이면 끝.
cd "$(dirname "$0")/.."
while pgrep -f "collect_admin_issue.py" >/dev/null; do sleep 60; done   # 이전 프로세스가 끝날 때까지
for i in $(seq 1 60); do
  if curl -s -m 20 -A "Mozilla/5.0" -X POST -d "bld=dbms/MDC/STAT/standard/MDCSTAT01501&locale=ko_KR&mktId=STK&trdDd=20261014&share=1&money=1&csvxls_isNo=false" \
       "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd" | head -c 200 | grep -q "OutBlock_1\|output"; then
    echo "$(date '+%F %T') KRX ok — run collector (try $i)"
    .venv/bin/python scripts/collect_admin_issue.py --sleep 0.5 >> data/raw/collect_admin.out 2>&1 && { echo "$(date '+%F %T') ADMIN DONE"; echo "CHAIN DONE $(date)" > data/raw/collect_chain.out; exit 0; }
    echo "$(date '+%F %T') collector exited non-zero; retry after 10 min"
  else
    echo "$(date '+%F %T') KRX not healthy (try $i)"
  fi
  sleep 600
done
echo "$(date '+%F %T') GAVE UP after 60 tries"
