#!/usr/bin/env python3
"""Audit active SYCL/Level Zero JIT and persistent-cache policy."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "env.sh"
BENCH = ROOT / "scripts/bench-serial.sh"
RESULTS = ROOT / "results"
KEYS = (
    "SYCL_CACHE_PERSISTENT",
    "SYCL_CACHE_DIR",
    "SYCL_CACHE_MAX_SIZE",
    "SYCL_CACHE_MIN_DEVICE_IMAGE_SIZE",
    "SYCL_CACHE_TRACE",
    "SYCL_PROGRAM_COMPILE_OPTIONS",
    "IGC_CACHE_DIR",
    "IGC_EnableKernelNames",
)
active_text = ENV.read_text() + "\n" + BENCH.read_text()
active = {key: os.environ.get(key) for key in KEYS}
source_mentions = {key: active_text.count(key) for key in KEYS}
artifacts = 0
historical_mentions = {key: 0 for key in KEYS}
for path in RESULTS.rglob("*.json"):
    try:
        text = path.read_text()
        json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    artifacts += 1
    for key in KEYS:
        historical_mentions[key] += text.count(key)
out = {
    "audit": "sycl-persistent-cache-policy",
    "environment": active,
    "active_source_mentions": source_mentions,
    "effective_explicit_policy": any(v is not None for v in active.values()) or any(source_mentions.values()),
    "json_artifacts_parsed": artifacts,
    "historical_mentions": historical_mentions,
}
print(json.dumps(out, indent=2, sort_keys=True))
