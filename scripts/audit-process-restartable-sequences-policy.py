#!/usr/bin/env python3
"""Audit Linux restartable-sequence ABI availability and Laguna provenance."""
import ctypes
import json
import pathlib
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark/results/process-restartable-sequences-policy-audit-20260807.json"
AT_RSEQ_FEATURE_SIZE = 27
AT_RSEQ_ALIGN = 28

libc = ctypes.CDLL(None)
libc.getauxval.argtypes = [ctypes.c_ulong]
libc.getauxval.restype = ctypes.c_ulong
feature_size = int(libc.getauxval(AT_RSEQ_FEATURE_SIZE))
alignment = int(libc.getauxval(AT_RSEQ_ALIGN))

env_text = (ROOT / "env.sh").read_text(errors="replace").lower()
harness_text = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace").lower()
terms = ("rseq", "restartable sequence", "glibc.pthread.rseq")

report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "process-restartable-sequences-policy",
    "live": {
        "at_rseq_feature_size": feature_size,
        "at_rseq_align": alignment,
        "kernel_advertises_rseq_abi": feature_size > 0 and alignment > 0,
    },
    "provenance": {
        "env_controls_or_records_rseq": any(term in env_text for term in terms),
        "serial_harness_controls_or_records_rseq": any(term in harness_text for term in terms),
    },
    "sources": {
        "abi": "getauxval(AT_RSEQ_FEATURE_SIZE/AT_RSEQ_ALIGN)",
        "policy_control": "GLIBC_TUNABLES=glibc.pthread.rseq=0 disables glibc registration",
    },
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
