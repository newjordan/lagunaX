#!/usr/bin/env python3
"""Audit host polling and scheduling controls used by the canonical serial benchmark."""
import json, os, re, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
cmd = ["bash", "-lc", f'source {root}/env.sh; "$LX_LLAMA_BENCH" --help 2>&1']
help_text = subprocess.run(cmd, text=True, capture_output=True, check=True).stdout

def help_default(flag):
    line = next((x for x in help_text.splitlines() if flag in x), "")
    m = re.search(r"default:\s*([^\)]+)", line)
    return {"line": line.strip(), "default": m.group(1).strip() if m else None}

sources = env_text + "\n" + bench_text
prio = help_default("--prio")
poll = help_default("--poll")
artifact = {
    "audit": "host-polling-scheduling-policy",
    "executable_support": {"priority": prio, "poll": poll},
    "canonical_policy": {
        "passes_priority": bool(re.search(r"(^|\s)--prio(?:\s|$)", sources)),
        "passes_poll": bool(re.search(r"(^|\s)--poll(?:\s|$)", sources)),
        "records_priority": '"priority"' in bench_text or '"prio"' in bench_text,
        "records_poll": '"poll"' in bench_text,
    },
    "process_environment": {
        "LX_PRIORITY": os.environ.get("LX_PRIORITY"),
        "LX_POLL": os.environ.get("LX_POLL"),
    },
}
out = root / "benchmark/results/host-polling-scheduling-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(artifact, indent=2) + "\n")
assert prio["default"] == "0", artifact
assert poll["default"] == "50", artifact
assert not any(artifact["canonical_policy"].values()), artifact
assert all(v is None for v in artifact["process_environment"].values()), artifact
print(out)
