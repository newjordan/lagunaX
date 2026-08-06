#!/usr/bin/env python3
"""Audit host swap/zswap policy and Laguna benchmark provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")

SYSFS = {
    "zswap_enabled": "/sys/module/zswap/parameters/enabled",
    "zswap_compressor": "/sys/module/zswap/parameters/compressor",
    "zswap_zpool": "/sys/module/zswap/parameters/zpool",
    "zswap_max_pool_percent": "/sys/module/zswap/parameters/max_pool_percent",
    "zswap_accept_threshold_percent": "/sys/module/zswap/parameters/accept_threshold_percent",
}
TOKENS = ("swappiness", "zswap", "swapoff", "SwapTotal", "/proc/swaps")


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def parse_swaps(text):
    lines = text.splitlines()
    return [dict(zip(("filename", "type", "size_kib", "used_kib", "priority"), line.split()))
            for line in lines[1:] if line.split()]


meminfo = {}
for line in (read("/proc/meminfo") or "").splitlines():
    key, _, value = line.partition(":")
    if key in ("SwapTotal", "SwapFree"):
        meminfo[key] = value.strip()

raw_swaps = read("/proc/swaps") or ""
report = {
    "angle": "host-swap-zswap-policy",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "live": {
        "swappiness": read("/proc/sys/vm/swappiness"),
        "page_cluster": read("/proc/sys/vm/page-cluster"),
        "swaps": parse_swaps(raw_swaps),
        "meminfo": meminfo,
        "zswap": {name.removeprefix("zswap_"): read(path) for name, path in SYSFS.items()},
    },
    "laguna": {
        "env_policy_hits": [token for token in TOKENS if token in ENV],
        "bench_policy_hits": [token for token in TOKENS if token in BENCH],
        "metrics_record_policy": any(f'"{token}"' in BENCH for token in TOKENS),
    },
    "finding": (
        "The audit captures active swap devices, VM swap clustering and swappiness, "
        "zswap configuration, and whether Laguna controls or records this policy."
    ),
}
out = ROOT / "results/host-swap-zswap-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
