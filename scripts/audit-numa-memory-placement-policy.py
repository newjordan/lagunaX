#!/usr/bin/env python3
"""Audit host NUMA topology, memory policy, and Laguna provenance."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
node_root = Path("/sys/devices/system/node")
node_ids = (node_root / "online").read_text().strip()
nodes = []
for path in sorted(node_root.glob("node[0-9]*")):
    nodes.append({
        "node": int(path.name[4:]),
        "cpulist": (path / "cpulist").read_text().strip(),
        "mem_total_kb": int(re.search(r"MemTotal:\s+(\d+) kB", (path / "meminfo").read_text()).group(1)),
    })
combined = env_text + "\n" + bench_text
configured = bool(re.search(r"(?:numactl|--membind|--cpunodebind|--interleave|set_mempolicy|mbind|NUMA)", combined, re.I))
recorded = bool(re.search(r'["\'](?:numa|numa_nodes|memory_policy|membind)["\']', bench_text, re.I))
artifact = {
    "angle": "host NUMA topology and memory-placement policy",
    "online_nodes": node_ids,
    "node_count": len(nodes),
    "nodes": nodes,
    "laguna_configures_numa_policy": configured,
    "metrics_record_numa_policy": recorded,
    "optimization_applicability": "no cross-node placement is possible on this single-node host",
}
assert len(nodes) == 1 and nodes[0]["node"] == 0
assert configured is False
assert recorded is False
out = root / "results/numa-memory-placement-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
