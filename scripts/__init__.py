"""Runnable entry-point scripts for the kairon plan.

Each module in this package is runnable as ``uv run python -m
scripts.<name>`` and writes its headline artifact to
``reports/`` and any sidecar to ``artifacts/``. The scripts
follow the project's "no silent defaults" rule (every
config field has either a YAML value, an env value, or a
required ``Field(...)``) and write a JSON sidecar so
downstream consumers (W2.5 GO/NO-GO gate, W3-4 cost-ML
re-work loop) can parse the output without re-running the
script.

The package is intentionally empty of business logic; the
heavy lifting lives in :mod:`kairon.evaluation` and the
runner is a thin IO + serialisation layer.
"""
