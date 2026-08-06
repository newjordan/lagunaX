#!/usr/bin/env python3
"""Audit process I/O scheduling priority and Laguna benchmark provenance."""
import ctypes
import json
import os
import pathlib
import platform
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark/results/process-io-priority-policy-audit-20260807.json"
ENV = (ROOT / "env.sh").read_text()
HARNESS = (ROOT / "scripts/bench-serial.sh").read_text()

# Linux ioprio_get(IOPRIO_WHO_PROCESS, 0). Python exposes no portable wrapper.
syscall_numbers = {"x86_64": 252, "amd64": 252, "aarch64": 30}
machine = platform.machine().lower()
if machine not in syscall_numbers:
    raise RuntimeError(f"unsupported architecture for ioprio_get: {machine}")
libc = ctypes.CDLL(None, use_errno=True)
raw = libc.syscall(syscall_numbers[machine], 1, 0)
if raw < 0:
    errno = ctypes.get_errno()
    raise OSError(errno, os.strerror(errno))

class_names = {0: "none", 1: "realtime", 2: "best-effort", 3: "idle"}
io_class = raw >> 13
io_data = raw & ((1 << 13) - 1)
source = ENV + "\n" + HARNESS
control_pattern = re.compile(r"\b(?:ionice|ioprio_set|IOPRIO_CLASS_)\b", re.I)
record_pattern = re.compile(r"\b(?:ioprio_get|io_priority|ioprio_class)\b", re.I)
report = {
    "angle": "process-io-priority-policy",
    "process_policy": {
        "raw_ioprio": raw,
        "class_id": io_class,
        "class_name": class_names.get(io_class, "unknown"),
        "class_data": io_data,
        "effective_semantics": (
            "inherits scheduler-derived best-effort priority"
            if io_class == 0
            else "explicit I/O priority"
        ),
    },
    "laguna": {
        "env_or_serial_harness_controls_io_priority": bool(control_pattern.search(source)),
        "serial_harness_records_io_priority": bool(record_pattern.search(HARNESS)),
    },
}
assert io_class in class_names, report
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
