"""CI helper: fail if src/kairon/ui/web/ contains forbidden escape hatches.

This mirrors the autouse fixture in ``tests/conftest.py`` so the same
check can be run as a pre-commit / CI command (without running pytest).

Usage:
    uv run python -m scripts.check_no_any_in_web
    ! uv run python -c "import scripts.check_no_any_in_web as m; m.main()"
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_WEB_TREE = Path("src/kairon/ui/web")
_FORBIDDEN = re.compile(r"\b(Any|cast)\b|#\s*type:\s*ignore")


def main() -> int:
    if not _WEB_TREE.is_dir():
        print(f"INFO: {_WEB_TREE} not present; nothing to check.")
        return 0
    offenders: list[str] = []
    for path in _WEB_TREE.rglob("*.py"):
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if _FORBIDDEN.search(src):
            offenders.append(path.as_posix())
    if offenders:
        print("FAIL: forbidden escape hatches (Any/cast/type:ignore) in:", file=sys.stderr)
        for o in offenders:
            print(f"  - {o}", file=sys.stderr)
        return 1
    print("OK: src/kairon/ui/web/ is clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
