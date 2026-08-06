#!/usr/bin/env python3
"""Audit exact-name GPU lock coverage against Linux TASK_COMM_LEN truncation."""
import json
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lock_source = (root / "scripts/lib-gpu-lock.sh").read_text()
requested = ["llama-bench", "llama-perplexity", "llama-cli", "llama-server"]
with tempfile.TemporaryDirectory() as td:
    link = Path(td) / "llama-perplexity"
    link.symlink_to("/bin/sleep")
    proc = subprocess.Popen([str(link), "10"])
    try:
        comm = subprocess.check_output(["ps", "-p", str(proc.pid), "-o", "comm="], text=True).strip()
        exact = subprocess.run(["pgrep", "-x", "llama-perplexity"], capture_output=True, text=True).returncode == 0
        truncated = subprocess.run(["pgrep", "-x", comm], capture_output=True, text=True).returncode == 0
    finally:
        proc.terminate()
        proc.wait()

result = {
    "audit": "gpu-lock-process-name-coverage",
    "requested_exact_names": requested,
    "linux_comm_observed": comm,
    "requested_name_length": len("llama-perplexity"),
    "observed_comm_length": len(comm),
    "pgrep_exact_full_name_matches": exact,
    "pgrep_exact_observed_comm_matches": truncated,
    "lock_uses_exact_full_name": "pgrep -a -x llama-perplexity" in lock_source,
    "llama_perplexity_covered": exact,
}
out = root / "results/gpu-lock-process-name-coverage-audit-20260807.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
assert result["lock_uses_exact_full_name"]
assert comm == "llama-perplexit"
assert not exact and truncated
