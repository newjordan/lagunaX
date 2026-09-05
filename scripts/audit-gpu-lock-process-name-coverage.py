#!/usr/bin/env python3
"""Check GPU lock detection using real Linux process names."""
import json
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lock_script = root / "scripts/lib-gpu-lock.sh"
requested = ["llama-bench", "llama-perplexity", "llama-cli", "llama-server", "test-backend-ops"]
results = []
with tempfile.TemporaryDirectory() as td:
    for name in requested:
        link = Path(td) / name
        link.symlink_to("/bin/sleep")
        proc = subprocess.Popen([str(link), "10"])
        try:
            comm = subprocess.check_output(["ps", "-p", str(proc.pid), "-o", "comm="], text=True).strip()
            detected = subprocess.check_output(
                ["bash", "-c", 'source "$1"; lx_gpu_foreign_procs', "audit", str(lock_script)], text=True
            )
            covered = str(proc.pid) in {line.split()[0] for line in detected.splitlines() if line.strip()}
            results.append({"name": name, "comm": comm, "covered": covered})
        finally:
            proc.terminate()
            proc.wait()

print(json.dumps({
    "audit": "gpu-lock-process-name-coverage",
    "results": results,
}, indent=2))
assert all(result["covered"] for result in results), "GPU process missing from lock detection"
