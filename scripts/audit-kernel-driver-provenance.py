#!/usr/bin/env python3
"""Audit kernel and Intel xe driver provenance retained by serial benchmark metrics."""
import json
import pathlib
import platform

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "scripts/bench-serial.sh"
BENCH = BENCH_PATH.read_text(errors="replace")

os_release = {}
for line in pathlib.Path("/etc/os-release").read_text(errors="replace").splitlines():
    if "=" in line and not line.startswith("#"):
        key, value = line.split("=", 1)
        os_release[key] = value.strip().strip('"')

module = pathlib.Path("/sys/module/xe")
parameters = {}
if (module / "parameters").is_dir():
    for path in sorted((module / "parameters").iterdir()):
        try:
            parameters[path.name] = path.read_text(errors="replace").strip()
        except OSError:
            pass

report = {
    "audit": "kernel-driver-provenance",
    "kernel_release": platform.release(),
    "kernel_version": platform.version(),
    "os_pretty_name": os_release.get("PRETTY_NAME"),
    "xe_driver_loaded": module.exists(),
    "xe_driver_parameters": parameters,
    "serial_metrics_records_kernel_release": "kernel_release" in BENCH,
    "serial_metrics_records_kernel_version": "kernel_version" in BENCH,
    "serial_metrics_records_os_release": "os_release" in BENCH,
    "serial_metrics_records_xe_driver": "xe_driver" in BENCH,
}
out = ROOT / "benchmark/results/kernel-driver-provenance-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
assert report["kernel_release"]
assert report["os_pretty_name"]
assert report["xe_driver_loaded"]
assert parameters
assert not any(report[k] for k in (
    "serial_metrics_records_kernel_release",
    "serial_metrics_records_kernel_version",
    "serial_metrics_records_os_release",
    "serial_metrics_records_xe_driver",
))
print(out)
