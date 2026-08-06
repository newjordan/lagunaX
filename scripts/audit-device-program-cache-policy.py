#!/usr/bin/env python3
"""Audit persistent SYCL/Level Zero compilation-cache policy and provenance."""
import json, os, pathlib, re, subprocess

root = pathlib.Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
harness_text = (root / "scripts/bench-serial.sh").read_text()
keys = ("SYCL_CACHE_PERSISTENT", "SYCL_CACHE_DIR", "SYCL_CACHE_THRESHOLD", "UR_L0_ENABLE_PROGRAM_CACHE")

def sourced_env():
    cmd = "source ./env.sh >/dev/null 2>&1; env -0"
    raw = subprocess.check_output(["bash", "-c", cmd], cwd=root)
    return dict(item.split("=", 1) for item in raw.decode(errors="replace").split("\0") if "=" in item)

env = sourced_env()
cache_dirs = [pathlib.Path.home()/".cache"/"libsycl_cache", pathlib.Path.home()/".cache"/"oneapi"]
report = {
    "angle": "persistent-device-program-cache-policy",
    "active_policy": {k: env.get(k) for k in keys},
    "source_mentions": {k: {"env_sh": bool(re.search(rf"\\b{re.escape(k)}\\b", env_text)), "serial_harness": bool(re.search(rf"\\b{re.escape(k)}\\b", harness_text))} for k in keys},
    "cache_directories": [{"path": str(p), "exists": p.exists(), "files": sum(1 for x in p.rglob("*") if x.is_file()) if p.exists() else 0} for p in cache_dirs],
    "provenance": {"serial_harness_records_any_cache_control": any(k in harness_text for k in keys)}
}
out = root / "benchmark/results/device-program-cache-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
