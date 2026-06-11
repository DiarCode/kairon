"""Aggregate all real experiment results into a single JSON for the paper."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
data = {}
for name in ["experiments.json", "experiments_structured.json", "experiments_diagnostic.json"]:
    p = ROOT / name
    if not p.exists():
        continue
    with p.open() as f:
        data[name.removesuffix(".json")] = json.load(f)

json.dump(data, sys.stdout, indent=2)
