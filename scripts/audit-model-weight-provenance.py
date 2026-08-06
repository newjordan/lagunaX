#!/usr/bin/env python3
"""Audit whether serial results bind performance claims to exact model bytes."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text()
BENCH = (ROOT / "scripts" / "bench-serial.sh").read_text()
match = re.search(r'export LX_MODEL="\$\{LX_MODEL:-([^}]+)\}"', ENV)
model = Path(match.group(1)) if match else None
model_exists = bool(model and model.is_file())
model_sha256 = None
if model_exists:
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    model_sha256 = digest.hexdigest()

report_name = "model-weight-provenance-audit-20260807.json"
files = sorted(path for path in (ROOT / "results").rglob("*.json") if path.name != report_name)
sha_records = 0
path_records = 0
for path in files:
    try:
        text = path.read_text()
        payload = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    lowered = text.lower()
    path_records += int("model" in lowered and ".gguf" in lowered)
    sha_records += int(any(key in lowered for key in ("model_sha256", "model_hash", "gguf_sha256")))

report = {
    "model_path": str(model) if model else None,
    "model_exists": model_exists,
    "model_sha256": model_sha256,
    "serial_harness_passes_model_path": '-m "$LX_MODEL"' in BENCH,
    "serial_harness_records_model_sha256": any(key in BENCH for key in ("model_sha256", "MODEL_SHA256", "sha256sum")),
    "json_artifacts_scanned": len(files),
    "artifacts_with_model_path": path_records,
    "artifacts_with_model_sha256": sha_records,
}
out = ROOT / "results" / report_name
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
assert model_exists and model_sha256 and len(model_sha256) == 64
assert report["serial_harness_passes_model_path"]
assert not report["serial_harness_records_model_sha256"]
assert len(files) > 0
print(out)
