#!/usr/bin/env python3
"""Audit backing-storage and kernel read-ahead provenance for model loading."""
import json, os, pathlib, subprocess
root = pathlib.Path(__file__).resolve().parents[1]
env = (root / 'env.sh').read_text()
bench = (root / 'scripts/bench-serial.sh').read_text()
model = os.environ.get('LX_MODEL')
if not model:
    p = subprocess.run(['bash','-lc',f'source {root}/env.sh >/dev/null 2>&1; printf %s "$LX_MODEL"'],text=True,capture_output=True,check=True)
    model=p.stdout
st = os.stat(model)
fs = subprocess.run(['findmnt','-T',model,'-J','-o','TARGET,SOURCE,FSTYPE,OPTIONS'],text=True,capture_output=True,check=True)
mount = json.loads(fs.stdout)['filesystems'][0]
source = mount['source']
lsblk = subprocess.run(['lsblk','-J','-o','NAME,PATH,TYPE,PKNAME,ROTA,RA,SCHED'],text=True,capture_output=True,check=True)
blocks=json.loads(lsblk.stdout)['blockdevices']
flat=[]
def walk(xs):
  for x in xs:
    flat.append({k:v for k,v in x.items() if k!='children'})
    walk(x.get('children') or [])
walk(blocks)
matched=[x for x in flat if source.startswith(x.get('path') or '/nonexistent')]
terms=('read_ahead','readahead','blockdev','findmnt','lsblk','mount_options','filesystem')
out={
 'model':model,'model_size_bytes':st.st_size,'mount':mount,'matching_block_devices':matched,
 'laguna_configures_storage_read_ahead':any(t in env.lower() or t in bench.lower() for t in terms),
 'metrics_record_storage_read_ahead':any(t in bench[bench.find('payload = {'):].lower() for t in terms),
}
outpath=root/'results'/'model-storage-read-ahead-policy-audit-20260807.json'
outpath.write_text(json.dumps(out,indent=2)+'\n')
print(outpath)
