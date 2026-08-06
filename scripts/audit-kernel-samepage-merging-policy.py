#!/usr/bin/env python3
"""Audit Kernel Samepage Merging policy, activity, and Laguna provenance."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
KSM = pathlib.Path("/sys/kernel/mm/ksm")


def read_optional(path: pathlib.Path):
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


fields = (
    "run",
    "pages_to_scan",
    "sleep_millisecs",
    "pages_scanned",
    "pages_shared",
    "pages_sharing",
    "pages_unshared",
    "pages_volatile",
    "full_scans",
    "stable_node_chains",
    "stable_node_dups",
    "use_zero_pages",
    "merge_across_nodes",
    "max_page_sharing",
    "smart_scan",
    "advisor_mode",
)
values = {field: read_optional(KSM / field) for field in fields}
env_text = (ROOT / "env.sh").read_text()
harness_text = (ROOT / "scripts/bench-serial.sh").read_text()
terms = ("/sys/kernel/mm/ksm", "MADV_MERGEABLE", "PR_SET_MEMORY_MERGE", "ksm")
report = {
    "angle": "kernel-samepage-merging-policy-and-activity",
    "ksm_sysfs_present": KSM.is_dir(),
    "policy_and_activity": values,
    "ksm_running": values["run"] not in (None, "0"),
    "provenance": {
        "env_controls_ksm": any(term in env_text for term in terms),
        "serial_harness_controls_or_records_ksm": any(term in harness_text for term in terms),
    },
}
out = ROOT / "benchmark/results/kernel-samepage-merging-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
