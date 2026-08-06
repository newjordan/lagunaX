#!/usr/bin/env python3
"""Audit per-process transparent-huge-page disablement and Laguna provenance."""
import ctypes
import json
import pathlib
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
PR_GET_THP_DISABLE = 42

libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
libc.prctl.restype = ctypes.c_int
thp_disabled = libc.prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0)
if thp_disabled < 0:
    errno = ctypes.get_errno()
    raise OSError(errno, "PR_GET_THP_DISABLE failed")

status = pathlib.Path("/proc/self/status").read_text()
status_value = next(
    (line.split(":", 1)[1].strip() for line in status.splitlines() if line.startswith("THP_enabled:")),
    None,
)
env_text = (ROOT / "env.sh").read_text().lower()
harness_text = (ROOT / "scripts/bench-serial.sh").read_text().lower()
terms = ("pr_set_thp_disable", "pr_get_thp_disable", "thp_enabled", "thp_disable")

report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "process-transparent-hugepage-disable-policy",
    "live": {
        "pr_get_thp_disable": thp_disabled,
        "transparent_hugepages_disabled_for_auditor": bool(thp_disabled),
        "proc_status_thp_enabled": status_value,
    },
    "provenance": {
        "env_controls_or_records_process_thp": any(term in env_text for term in terms),
        "serial_harness_controls_or_records_process_thp": any(term in harness_text for term in terms),
    },
}
out = ROOT / "benchmark/results/process-transparent-hugepage-disable-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
