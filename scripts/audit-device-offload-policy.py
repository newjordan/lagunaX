#!/usr/bin/env python3
import json, os, re, subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[1]
env_text = (root/'env.sh').read_text()
bench_text = (root/'scripts/bench-serial.sh').read_text()
cmd = f'source {root}/env.sh && "$LX_LLAMA_BENCH" --help 2>&1'
help_text = subprocess.run(['bash','-lc',cmd], text=True, capture_output=True, check=True).stdout

def default(flag):
    m = re.search(rf'{re.escape(flag)}[^\n]*\(default:\s*([^\)]+)\)', help_text)
    if not m: raise SystemExit(f'missing help contract for {flag}')
    return m.group(1).strip()

payload = {
  'angle': 'device_offload_policy',
  'live_help': {
    'no_kv_offload_supported': '--no-kv-offload' in help_text,
    'no_kv_offload_default': default('--no-kv-offload'),
    'no_op_offload_supported': '--no-op-offload' in help_text,
    'no_op_offload_default': default('--no-op-offload'),
  },
  'canonical_sources': {
    'kv_offload_override': bool(re.search(r'(^|\s)(-nkvo|--no-kv-offload)(\s|$)', env_text+'\n'+bench_text)),
    'op_offload_override': bool(re.search(r'(^|\s)(-nopo|--no-op-offload)(\s|$)', env_text+'\n'+bench_text)),
    'metrics_record_kv_offload': 'kv_offload' in bench_text,
    'metrics_record_op_offload': 'op_offload' in bench_text,
  }
}
assert payload['live_help'] == {'no_kv_offload_supported': True, 'no_kv_offload_default': '0', 'no_op_offload_supported': True, 'no_op_offload_default': '0'}
assert not any(payload['canonical_sources'].values())
out = root/'benchmark/results/device-offload-policy-audit-20260807.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2)+'\n')
print(out)
