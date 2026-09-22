#!/usr/bin/env python3
"""Merge live Muon-aware filter results and quantify durability."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt


def load(root:Path):
    out=[]
    for p in root.rglob('result.json'):
        try:r=json.loads(p.read_text());r['result_path']=str(p);out.append(r)
        except Exception as e:print('skip',p,e)
    if not out:raise SystemExit('no results')
    return out

def paired(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);d=a-b
    m=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else 0.;z=m/se if se else 0.
    return m,se,z

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rs=load(a.root); exact=[r for r in rs if r['arm'].startswith('exact_lr:')]
    if not exact:raise SystemExit('need exact baseline')
    best=min(exact,key=lambda r:float(r['trace'][-1]['bpb']))
    rows=[]
    for r in rs:
        f=r['trace'][-1]; d,se,z=paired(f['bpb_values'],best['trace'][-1]['bpb_values'])
        stop=r.get('filter_stop_row')
        stop_delta=float('nan'); retain=float('nan')
        if stop:
            # Interpolate exact baseline at the nearest charged FLOP point.
            e=min(best['trace'],key=lambda x:abs(float(x['charged_flop_equiv'])-float(stop['charged_flop_equiv'])))
            sd,_,_=paired(stop['bpb_values'],e['bpb_values']); stop_delta=sd
            if sd<0: retain=(-d)/(-sd)
        rows.append({'arm':r['arm'],'final_bpb':f['bpb'],'delta_vs_best_exact':d,'paired_se':se,'paired_z':z,
                     'delta_at_filter_stop':stop_delta,'retained_fraction':retain,'actions':r['final_ledger']['synthetic_steps'],
                     'charged_flop_equiv':f['charged_flop_equiv'],'charged_seconds':f['charged_seconds'],'result_path':r['result_path']})
    rows.sort(key=lambda x:float(x['final_bpb']))
    with (a.out/'leaderboard.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('best exact',best['arm'],best['trace'][-1]['bpb'])
    for r in rows[:30]:print(f"{r['arm'][:55]:55} {float(r['final_bpb']):.6f} {float(r['delta_vs_best_exact']):+.6f} z={float(r['paired_z']):+.2f} stop={r['delta_at_filter_stop']} retain={r['retained_fraction']}")
    fig,ax=plt.subplots(figsize=(10,max(4,.32*min(len(rows),30))))
    rr=rows[:30];y=np.arange(len(rr));v=np.array([x['delta_vs_best_exact'] for x in rr]);err=2*np.array([x['paired_se'] for x in rr])
    ax.barh(y,v,xerr=err);ax.axvline(0,linewidth=1);ax.set_yticks(y,[x['arm'] for x in rr]);ax.invert_yaxis();ax.set_xlabel('final BPB delta vs tuned exact');ax.set_title('Muon-aware live filters after exact continuation');fig.tight_layout();fig.savefig(a.out/'fig_final_delta.png',dpi=180);fig.savefig(a.out/'fig_final_delta.pdf');plt.close(fig)
    # retention scatter
    pts=[x for x in rows if math.isfinite(float(x['delta_at_filter_stop']))]
    if pts:
        fig,ax=plt.subplots(figsize=(7,6));ax.scatter([x['delta_at_filter_stop'] for x in pts],[x['delta_vs_best_exact'] for x in pts]);ax.axhline(0,linewidth=1);ax.axvline(0,linewidth=1)
        for x in pts:ax.annotate(x['arm'][:25],(x['delta_at_filter_stop'],x['delta_vs_best_exact']),fontsize=6)
        ax.set_xlabel('delta vs exact when filter turns off');ax.set_ylabel('delta vs exact after continuation');ax.set_title('Transient smoothing or durable optimization?');fig.tight_layout();fig.savefig(a.out/'fig_durability.png',dpi=180);fig.savefig(a.out/'fig_durability.pdf');plt.close(fig)
    (a.out/'validation.json').write_text(json.dumps({'best_exact_arm':best['arm'],'equal_compute_spread':max(x['charged_flop_equiv'] for x in rows)-min(x['charged_flop_equiv'] for x in rows)},indent=2))
if __name__=='__main__':main()
