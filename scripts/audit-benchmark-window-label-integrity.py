#!/usr/bin/env python3
"""Audit whether configurable benchmark windows are represented faithfully."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
bench = (root / "scripts/bench-serial.sh").read_text()
score = (root / "scripts/score.py").read_text()
env = (root / "env.sh").read_text()

payload = {
    "policy": {
        "window_is_environment_configurable": all(x in env for x in ('LX_PP="${LX_PP:-512}"', 'LX_TG="${LX_TG:-128}"')),
        "invocations_use_configurable_window": all(x in bench for x in ('-p "$LX_PP" -n 0', '-p 0 -n "$LX_TG"')),
        "metrics_window_records_configured_values": '"window": {"pp": int("$LX_PP"), "tg": int("$LX_TG")' in bench,
        "throughput_keys_are_hardcoded": '"pp512": float("$PP_TS")' in bench and '"tg128": float("$TG_TS")' in bench,
        "score_consumes_hardcoded_keys": all(re.search(p, score) is not None for p in (r'\["pp512"\]', r'\["tg128"\]')),
        "window_key_consistency_guard": False,
    },
    "risk": "Overriding LX_PP or LX_TG runs a different workload but stores throughput under pp512/tg128 keys consumed by score.py.",
}
assert all(payload["policy"][k] for k in (
    "window_is_environment_configurable", "invocations_use_configurable_window",
    "metrics_window_records_configured_values", "throughput_keys_are_hardcoded",
    "score_consumes_hardcoded_keys"))
out = root / "benchmark/results/benchmark-window-label-integrity-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
