#!/usr/bin/env python3
"""Audit whether serial Laguna benchmarks control CPU affinity."""
from pathlib import Path
import json, re
root = Path(__file__).resolve().parents[1]
bench = (root / 'scripts/bench-serial.sh').read_text()
env = (root / 'env.sh').read_text()
controls = {
    'taskset': bool(re.search(r'(^|\s)taskset(\s|$)', bench, re.M)),
    'numactl_physcpubind': '--physcpubind' in bench,
    'sched_setaffinity': 'sched_setaffinity' in bench,
    'omp_places': 'OMP_PLACES' in bench or 'OMP_PLACES' in env,
    'kmp_affinity': 'KMP_AFFINITY' in bench or 'KMP_AFFINITY' in env,
    'ze_affinity_mask': 'ZE_AFFINITY_MASK' in bench or 'ZE_AFFINITY_MASK' in env,
}
out = {
    'audit': 'cpu-affinity-control',
    'host_cpu_affinity_explicit': any(controls[k] for k in ('taskset','numactl_physcpubind','sched_setaffinity','omp_places','kmp_affinity')),
    'controls': controls,
    'distinction': 'ZE_AFFINITY_MASK selects GPU devices/subdevices; it does not bind host benchmark threads to CPU cores.',
}
print(json.dumps(out, indent=2))
