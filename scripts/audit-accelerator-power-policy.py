#!/usr/bin/env python3
"""Audit accelerator energy counters and configured power limits without changing them."""
import glob, json, os
from pathlib import Path

hwmons = sorted(glob.glob('/sys/class/drm/card*/device/hwmon/hwmon*'))
devices = []
for root in hwmons:
    def read(name):
        p = Path(root, name)
        try: return p.read_text().strip()
        except OSError: return None
    energies = []
    for p in sorted(Path(root).glob('energy*_input')):
        n = p.name.removesuffix('_input')
        energies.append({'channel': n, 'label': read(n + '_label'), 'microjoules': int(read(p.name))})
    caps = []
    for p in sorted(Path(root).glob('power*_cap')):
        n = p.name.removesuffix('_cap')
        caps.append({'channel': n, 'label': read(n + '_label'), 'cap_microwatts': int(read(p.name)),
                     'critical_microwatts': int(read(n + '_crit')) if read(n + '_crit') else None,
                     'interval_microseconds': int(read(n + '_cap_interval')) if read(n + '_cap_interval') else None})
    if energies or caps:
        devices.append({'hwmon': root, 'name': read('name'), 'energy_counters': energies, 'power_caps': caps})
source = Path('env.sh').read_text() + '\n' + Path('scripts/bench-serial.sh').read_text()
keys = ('power1_cap', 'power1_crit', 'energy1_input', 'ZES_POWER', 'SYSMAN')
out = {'devices': devices, 'explicit_power_policy_in_active_sources': [k for k in keys if k in source],
       'explicit_power_policy_in_environment': {k:v for k,v in os.environ.items() if 'POWER' in k or 'SYSMAN' in k}}
outfile = Path('results/accelerator-power-policy-audit-20260807.json')
outfile.write_text(json.dumps(out, indent=2) + '\n')
print(outfile)
