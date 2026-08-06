#!/usr/bin/env python3
"""Audit CPU affinity controls independently of NUMA and priority policy."""
import json, os, re, subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[1]
env = (root / "env.sh").read_text()
harness = (root / "scripts/bench-serial.sh").read_text()
bench = os.environ["LX_LLAMA_BENCH"]
help_text = subprocess.run([bench, "--help"], text=True, capture_output=True, check=True).stdout
mask = re.search(r"--cpu-mask <hex,hex>.*?default: ([^)]+)", help_text)
threads = re.search(r"--threads <n>.*?default: ([^)]+)", help_text)
assert mask and threads
active = bool(re.search(r"(?:--cpu-mask|-C)\s", "\n".join((env, harness))))
data = {
  "angle": "CPU core affinity binding (distinct from NUMA mode and process priority)",
  "supported": True,
  "executable_default_cpu_mask": mask.group(1),
  "executable_default_threads": int(threads.group(1)),
  "canonical_threads": int(os.environ["THREADS"]),
  "canonical_cpu_mask_override": active,
  "effective_policy": "OS-scheduled across allowed CPUs" if not active and mask.group(1) == "0x0" else "explicit",
}
out = root / "benchmark/results/cpu-affinity-policy-audit-20260807.json"
out.write_text(json.dumps(data, indent=2) + "\n")
print(out)
