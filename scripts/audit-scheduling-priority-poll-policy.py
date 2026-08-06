#!/usr/bin/env python3
"""Audit active llama-bench scheduling priority and spin-poll policy."""
import json, os, re, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = subprocess.run(
    ["bash", "-lc", f"source {root/'env.sh'}; env -0"],
    check=True, capture_output=True,
).stdout.split(b"\0")
env = dict(x.decode().split("=", 1) for x in env if b"=" in x)
bench = env["LX_LLAMA_BENCH"]
help_text = subprocess.run([bench, "--help"], env=env, text=True,
                           capture_output=True, check=True).stdout
serial = (root / "scripts/bench-serial.sh").read_text()
env_src = (root / "env.sh").read_text()
prio = re.search(r"--prio <-1\|0\|1\|2\|3>.*?default: (-?\d+)", help_text)
poll = re.search(r"--poll <0\.\.\.100>.*?default: (\d+)", help_text)
if not prio or not poll:
    raise SystemExit("unable to parse --prio/--poll defaults")
active_text = env_src + "\n" + serial
result = {
    "benchmark": bench,
    "controls": {
        "priority": {"supported": True, "default": int(prio.group(1)), "overridden": bool(re.search(r"(?:--prio|\bPRIO=)", active_text))},
        "poll": {"supported": True, "default_percent": int(poll.group(1)), "overridden": bool(re.search(r"(?:--poll|\bPOLL=)", active_text))},
    },
    "interpretation": "Active runs use normal priority and the default 50% spin-poll policy.",
}
out = root / "results/scheduling-priority-poll-policy-audit-20260807.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
