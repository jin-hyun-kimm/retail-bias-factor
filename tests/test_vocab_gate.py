"""D23 ③ 어휘 게이트: 수익률 전망성 낱말이 코드·문서·결과에 나오면 실패. 브리프 원문과 이 파일은 제외."""
from pathlib import Path

from rbf.config import FORBIDDEN_WORDS, ROOT

SCAN_DIRS = ["src", "scripts", "docs", "results", "tests"]
SCAN_FILES = ["README.md"]
EXCLUDE = {"mirae_brief.md", "test_vocab_gate.py", "config.py"}
EXTS = {".py", ".md", ".txt", ".html", ".csv", ".sh", ".toml", ".json"}


def _files():
    for d in SCAN_DIRS:
        p = ROOT / d
        if p.exists():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix in EXTS and f.name not in EXCLUDE:
                    yield f
    for f in SCAN_FILES:
        p = ROOT / f
        if p.exists():
            yield p


def test_no_forbidden_words():
    hits = []
    for f in _files():
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for w in FORBIDDEN_WORDS:
                if w in line:
                    hits.append(f"{f.relative_to(ROOT)}:{i}: {w}")
    assert not hits, "금지 낱말:\n" + "\n".join(hits)
