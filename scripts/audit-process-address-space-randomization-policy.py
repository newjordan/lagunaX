#!/usr/bin/env python3
"""Audit ASLR/personality state and Laguna benchmark provenance."""
import json
import os
import pathlib
import re
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text()
BENCH = (ROOT / "scripts/bench-serial.sh").read_text()

randomize_va_space = int(pathlib.Path("/proc/sys/kernel/randomize_va_space").read_text().strip())
personality_raw = pathlib.Path("/proc/self/personality").read_text().strip()
personality = int(personality_raw, 16)
addr_no_randomize = 0x0040000
source = ENV + "\n" + BENCH
controls = ("randomize_va_space", "setarch", "ADDR_NO_RANDOMIZE", "personality")

payload = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "process-address-space-randomization-policy",
    "live_state": {
        "kernel_randomize_va_space": randomize_va_space,
        "kernel_mode": {0: "disabled", 1: "conservative", 2: "full"}.get(randomize_va_space, "unknown"),
        "audit_process_personality_hex": personality_raw,
        "audit_process_addr_no_randomize": bool(personality & addr_no_randomize),
    },
    "canonical_policy": {
        "env_or_harness_controls_aslr": any(re.search(rf"\b{re.escape(item)}\b", source, re.I) for item in controls),
        "metrics_record_aslr_state": any(item in BENCH for item in ("randomize_va_space", "process_personality", "aslr_state")),
    },
}
out = ROOT / "results/process-address-space-randomization-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(out)
assert randomize_va_space == 2
assert not payload["live_state"]["audit_process_addr_no_randomize"]
assert not any(payload["canonical_policy"].values())
