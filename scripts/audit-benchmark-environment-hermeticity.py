#!/usr/bin/env python3
"""Audit whether serial benchmark runs are insulated from inherited runtime knobs."""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text()
BENCH = (ROOT / "scripts/bench-serial.sh").read_text()

# Runtime families capable of changing device selection, dispatch, compilation, or threading.
family = re.compile(r"^(GGML_SYCL_|SYCL_|ZE_|ONEAPI_|UR_|DNNL_|MKL_|OMP_|KMP_)")
inherited = {k: v for k, v in os.environ.items() if family.match(k)}

# Names explicitly assigned/unset by env.sh and names retained in metrics JSON.
managed = set(re.findall(r"(?:export|unset)\s+([A-Z][A-Z0-9_]*)", ENV))
recorded = set(re.findall(r'os\.environ\.get\("([A-Z][A-Z0-9_]*)"\)', BENCH))
unmanaged_active = sorted(set(inherited) - managed)
unrecorded_active = sorted(set(inherited) - recorded)

payload = {
    "audit": "serial-benchmark-environment-hermeticity",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "sources": ["env.sh", "scripts/bench-serial.sh"],
    "runtime_variable_families": ["GGML_SYCL_", "SYCL_", "ZE_", "ONEAPI_", "UR_", "DNNL_", "MKL_", "OMP_", "KMP_"],
    "active_runtime_variables": inherited,
    "managed_by_env_sh": sorted(set(inherited) & managed),
    "recorded_by_metrics": sorted(set(inherited) & recorded),
    "unmanaged_active_variables": unmanaged_active,
    "unrecorded_active_variables": unrecorded_active,
    "counts": {
        "active": len(inherited),
        "managed": len(set(inherited) & managed),
        "recorded": len(set(inherited) & recorded),
        "unmanaged": len(unmanaged_active),
        "unrecorded": len(unrecorded_active),
    },
    "assertions": {
        "bench_inherits_process_environment": "env -i" not in BENCH and "unsetenv" not in BENCH,
        "metrics_do_not_capture_complete_runtime_environment": bool(unrecorded_active),
    },
}
out = ROOT / "results" / "benchmark-environment-hermeticity-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(out)
print(json.dumps(payload["counts"], sort_keys=True))
if not all(payload["assertions"].values()):
    raise SystemExit("audit assertion failed")
