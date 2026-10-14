"""D23 ②: 홀드아웃 실행 기록 파일의 형식을 검사한다(막지는 않는다 — 다시 돌리면 그 사실이 남아야 한다)."""
import json

from rbf.config import RESULTS

LOG = RESULTS / "holdout_runs.jsonl"


def test_holdout_run_log_format():
    if not LOG.exists():
        return  # 아직 홀드아웃을 돌리지 않았다
    lines = [ln for ln in LOG.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for ln in lines:
        rec = json.loads(ln)
        assert {"ts", "commit", "note"} <= set(rec), rec
        assert len(rec["commit"]) >= 7
    assert len(lines) >= 1
