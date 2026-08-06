#!/usr/bin/env python3
"""Audit oneDNN backend dispatch policy in active Laguna configuration and results."""
import json
import os
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
pattern = re.compile(r"GGML_SYCL_DISABLE_DNN")
active_assignment = re.search(r"(?:export\s+)?GGML_SYCL_DISABLE_DNN=([^\s#]+)", env_text)
active_unset = bool(re.search(r"^\s*unset\s+GGML_SYCL_DISABLE_DNN(?:\s|$)", env_text, re.M))
counts = {"disabled": 0, "enabled": 0, "other": 0}
files_scanned = 0
records = 0
for path in sorted((root / "results").rglob("*.json")):
    if path.name == "onednn-dispatch-policy-audit-20260807.json":
        continue
    files_scanned += 1
    try:
        obj = json.loads(path.read_text())
    except Exception:
        continue
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for key, value in cur.items():
                if key == "GGML_SYCL_DISABLE_DNN":
                    records += 1
                    if value is None:
                        counts["other"] += 1
                    else:
                        normalized = str(value).strip().lower()
                        if normalized in {"1", "true", "yes", "on"}:
                            counts["disabled"] += 1
                        elif normalized in {"0", "false", "no", "off"}:
                            counts["enabled"] += 1
                        else:
                            counts["other"] += 1
                else:
                    stack.append(value)
        elif isinstance(cur, list):
            stack.extend(cur)

payload = {
    "audit": "onednn-dispatch-policy",
    "active_policy": "unset/default" if active_unset else (active_assignment.group(1) if active_assignment else "unspecified"),
    "serial_harness_records_policy": bool(pattern.search(bench_text)),
    "json_files_scanned": files_scanned,
    "historical_records": records,
    "historical_counts": counts,
    "process_value": os.environ.get("GGML_SYCL_DISABLE_DNN"),
}
out = root / "results" / "onednn-dispatch-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
