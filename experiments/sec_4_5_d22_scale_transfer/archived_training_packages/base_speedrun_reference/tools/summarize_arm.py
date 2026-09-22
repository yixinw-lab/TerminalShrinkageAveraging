#!/usr/bin/env python3
import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--arm',required=True); ap.add_argument('--result-dir',type=Path,required=True); ap.add_argument('--threshold',type=float,required=True); a=ap.parse_args()
rows=[]
for name in ('raw_core.json','tsa_core.json'):
    p=a.result_dir/name
    if p.exists():
        x=json.loads(p.read_text()); rows.append((name,x))
lines=[f"arm={a.arm}",f"core_threshold={a.threshold:.6f}"]
for name,x in rows:
    prefix='tsa' if name.startswith('tsa') else 'raw'
    lines += [f"{prefix}_training_minutes={x['training_minutes']:.6f}",f"{prefix}_val_bpb={x['val_bpb']:.6f}",f"{prefix}_core={x['core_metric']:.6f}",f"{prefix}_qualifies={'YES' if x['core_metric']>a.threshold else 'NO'}"]
if len(rows)==2:
    d={('tsa' if n.startswith('tsa') else 'raw'):x for n,x in rows}
    lines += [f"tsa_minus_raw_core={d['tsa']['core_metric']-d['raw']['core_metric']:+.6f}",f"tsa_minus_raw_val_bpb={d['tsa']['val_bpb']-d['raw']['val_bpb']:+.6f}"]
out=a.result_dir/'DIGEST.txt'; out.write_text('\n'.join(lines)+'\n'); print(out.read_text())
