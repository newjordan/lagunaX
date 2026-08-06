#!/usr/bin/env python3
import hashlib, json, os, pathlib, re, subprocess
root=pathlib.Path(__file__).resolve().parents[1]
env=os.environ.copy()
# Resolve exactly as env.sh does, without trusting caller defaults.
p=subprocess.run(['bash','-lc','source ./env.sh; printf %s "$LX_LLAMA_BENCH"'],cwd=root,text=True,capture_output=True,check=True)
bin=pathlib.Path(p.stdout)
def run(*args):
 return subprocess.run(args,text=True,capture_output=True,check=True).stdout
sections=run('readelf','-SW',str(bin))
syms=run('readelf','-Ws',str(bin))
debug=sorted(set(re.findall(r'\.debug_[A-Za-z0-9_.]+',sections)))
rows=[]
for line in syms.splitlines():
 m=re.match(r'\s*\d+:\s+[0-9a-fA-F]+\s+\d+\s+\w+\s+\w+\s+\w+\s+(\w+)\s+(.+)',line)
 if m and m.group(1) != 'UND': rows.append(m.group(2).strip())
data={'audit':'executable-footprint','binary':str(bin),'sha256':hashlib.sha256(bin.read_bytes()).hexdigest(),'file_bytes':bin.stat().st_size,'debug_sections':debug,'defined_symbol_rows':len(rows),'stripped':len(rows)==0 and not debug}
out=root/'results'/'executable-footprint-audit-20260807.json'
out.write_text(json.dumps(data,indent=2)+'\n')
print(out)
