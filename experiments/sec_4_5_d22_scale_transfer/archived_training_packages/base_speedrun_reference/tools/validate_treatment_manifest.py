#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--manifest',type=Path,required=True)
ap.add_argument('--ratio',type=float,required=True)
ap.add_argument('--clamp',type=float,required=True)
ap.add_argument('--k',type=int,required=True)
a=ap.parse_args(); m=json.loads(a.manifest.read_text())
assert m.get('complete'), 'manifest incomplete'
assert int(m['snapshot_k']) == a.k, (m.get('snapshot_k'), a.k)
uc=m.get('user_config',{})
sr=float(uc.get('record_schedule_param_data_ratio'))
cl=float(uc.get('record_terminal_lr_clamp_frac'))
assert math.isclose(sr,a.ratio,rel_tol=0,abs_tol=1e-9), (sr,a.ratio)
assert math.isclose(cl,a.clamp,rel_tol=0,abs_tol=1e-9), (cl,a.clamp)
ends=list(m['endpoints'].values())
assert len(ends)==1, len(ends)
e=ends[0]
assert math.isclose(float(e['ratio']),a.ratio,rel_tol=0,abs_tol=1e-9), e['ratio']
steps=[int(x) for x in e['snapshot_steps']]
assert len(steps)==a.k and steps[-1]==int(e['endpoint_step'])
print(f"treatment manifest PASS ratio={a.ratio:g} endpoint={e['endpoint_step']} spacing={m['snapshot_spacing_steps']} steps={steps}")
