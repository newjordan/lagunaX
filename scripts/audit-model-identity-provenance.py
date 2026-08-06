#!/usr/bin/env python3
"""Audit whether serial benchmark evidence binds results to exact model bytes."""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_path = root / "scripts/bench-serial.sh"
bench_text = bench_path.read_text()

m = re.search(r'export LX_MODEL="\$\{LX_MODEL:-([^}]+)\}"', env_text)
if not m:
    raise SystemExit("cannot resolve LX_MODEL default")
model = Path(os.environ.get("LX_MODEL", m.group(1)))
if not model.is_file():
    raise SystemExit(f"model missing: {model}")

h = hashlib.sha256()
with model.open("rb") as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
        h.update(chunk)

model_sha_configured = bool(re.search(r'MODEL_SHA256=', bench_text))
model_sha_emitted = bool(re.search(r'["\']model_sha256["\']\s*:', bench_text))
model_stat = model.stat()
artifact = {
    "audit": "model-identity-provenance",
    "model": str(model),
    "model_size_bytes": model_stat.st_size,
    "model_sha256": h.hexdigest(),
    "harness_records_model_path": '"model": "$LX_MODEL"' in bench_text,
    "harness_computes_model_sha256": model_sha_configured,
    "harness_emits_model_sha256": model_sha_emitted,
    "exact_model_bytes_bound_to_metrics": model_sha_configured and model_sha_emitted,
    "bench_binary_sha256_bound_to_metrics": "BENCH_SHA256=" in bench_text and '"binary_sha256"' in bench_text,
}
out = root / "results/model-identity-provenance-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
assert artifact["harness_records_model_path"]
assert artifact["bench_binary_sha256_bound_to_metrics"]
assert not artifact["exact_model_bytes_bound_to_metrics"]
print(json.dumps(artifact, indent=2))
