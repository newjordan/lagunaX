#!/usr/bin/env python3
"""Audit Linux CPU core-scheduling capability/state and Laguna provenance."""
import ctypes
import errno
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PR_SCHED_CORE = 62
PR_SCHED_CORE_GET = 0
PR_SCHED_CORE_SCOPE_THREAD = 0


def read_optional(path: pathlib.Path):
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


libc = ctypes.CDLL(None, use_errno=True)
cookie = ctypes.c_ulonglong()
rc = libc.prctl(
    PR_SCHED_CORE,
    PR_SCHED_CORE_GET,
    0,
    PR_SCHED_CORE_SCOPE_THREAD,
    ctypes.byref(cookie),
)
err = ctypes.get_errno() if rc else 0
cmdline = read_optional(pathlib.Path("/proc/cmdline")) or ""
env_text = (ROOT / "env.sh").read_text()
harness_text = (ROOT / "scripts/bench-serial.sh").read_text()
terms = ("PR_SCHED_CORE", "sched_core", "core scheduling", "core-scheduling")
report = {
    "angle": "linux-cpu-core-scheduling-policy-and-provenance",
    "pr_sched_core_get": {
        "supported": rc == 0,
        "return_code": rc,
        "errno": err,
        "error": None if rc == 0 else errno.errorcode.get(err, os.strerror(err)),
        "calling_thread_cookie": cookie.value if rc == 0 else None,
    },
    "smt": {
        "active": read_optional(pathlib.Path("/sys/devices/system/cpu/smt/active")),
        "control": read_optional(pathlib.Path("/sys/devices/system/cpu/smt/control")),
    },
    "boot_overrides": [term for term in ("nosmt", "sched_core") if term in cmdline],
    "provenance": {
        "env_controls_core_scheduling": any(term in env_text for term in terms),
        "serial_harness_records_core_scheduling": any(term in harness_text for term in terms),
    },
}
out = ROOT / "benchmark/results/cpu-core-scheduling-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
