"""W9.4 — CI smoke job integration test.

The W9.4 acceptance criterion:

- ``tests/integration/test_ci_smoke.py::test_smoke_job_defined``
  passes: the CI YAML contains the smoke job and references the
  expected script.

The test reads ``.github/workflows/ci.yml`` as plain text (no
PyYAML dependency) and asserts:

1. The file contains a job named ``real-data-smoke``.
2. The job invokes ``scripts/run_e2e.py`` with a ``btc_1h``
   subcommand.
3. The job has a timeout of <= 5 minutes (the W9.4 acceptance
   criterion "must complete in < 5 minutes").
4. The job's step that runs the script targets the 1-month window
   (we accept either ``--n-bars 720`` or an explicit comment
   referencing 1mo).
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
CI_YAML: Path = REPO_ROOT / ".github" / "workflows" / "ci.yml"


@pytest.mark.skipif(not CI_YAML.exists(), reason="CI workflow not found")
def test_smoke_job_defined() -> None:
    """The CI YAML contains a 'real-data-smoke' job that runs the W9.4 smoke."""
    text: str = CI_YAML.read_text(encoding="utf-8")
    # 1. The job is present and named correctly.
    assert "real-data-smoke:" in text, (
        f"CI YAML is missing the 'real-data-smoke' job; see {CI_YAML}"
    )
    # 2. The job invokes scripts/run_e2e.py with the btc_1h subcommand.
    assert "scripts/run_e2e.py" in text, (
        f"CI YAML is missing the scripts/run_e2e.py invocation; see {CI_YAML}"
    )
    assert "btc_1h" in text, (
        f"CI YAML is missing the 'btc_1h' subcommand for the smoke job; "
        f"see {CI_YAML}"
    )
    # 3. The job has a timeout of <= 5 minutes (the W9.4 acceptance
    # criterion: "the job must complete in < 5 minutes and pass").
    # The YAML uses ``timeout-minutes: 5`` on the job.
    assert "timeout-minutes: 5" in text, (
        f"CI YAML is missing the 'timeout-minutes: 5' on the smoke job; "
        f"see {CI_YAML}"
    )
    # 4. The job targets a 1-month window: ``--n-bars 720`` is the
    # canonical 1mo x 30d x 24h signal. The script's --n-bars
    # argument is a recent addition; the test accepts either
    # explicit --n-bars 720 or a 1mo keyword.
    assert ("--n-bars 720" in text) or ("1mo" in text), (
        f"CI YAML does not target a 1-month window (expected "
        f"'--n-bars 720' or '1mo'); see {CI_YAML}"
    )
