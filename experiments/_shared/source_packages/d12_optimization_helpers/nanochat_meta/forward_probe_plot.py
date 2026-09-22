#!/usr/bin/env python3
"""Merge forward-probe fast-forward results."""
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
    a=np.asarray(a,float);b=np.asarray(b,float);d=a-b;m=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else 0.;z=m/se if se else 0.;return m,se,z

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rs=load(a.root)
    exact=[r for r in rs if r['arm'].startswith('exact_lr:')]
    if not exact:raise SystemExit('need exact_lr')
    best_exact=min(exact,key=lambda r:float(r['trace'][-1]['bpb']))
    standard=[r for r in rs if r['arm'].startswith(('exact_lr:','eval_lawa_lr:'))]
    best_std=min(standard,key=lambda r:float(r['trace'][-1]['bpb']))
    rows=[]
    for r in rs:
        f=r['trace'][-1];ref=best_std['trace'][-1]
        vals=f.get('bpb_values',f.get('raw_bpb_values'))
        rvals=ref.get('bpb_values',ref.get('raw_bpb_values'))
        d,se,z=paired(vals,rvals)
        rows.append({'arm':r['arm'],'family':'standard' if r in standard else 'forward-probe',
                     'final_bpb':f['bpb'],'delta_vs_best_standard':d,'paired_se':se,'paired_z':z,
                     'charged_flop_equiv':f['charged_flop_equiv'],'charged_seconds':f['charged_seconds'],
                     'exact_backward_calls':r['final_ledger']['exact_backward_calls'],'virtual_steps':r.get('virtual_steps'),
                     'jump_actions':r.get('jump_actions',0),'accepted':r.get('accepted',0),'rejected':r.get('rejected',0),
                     'acceptance_rate':r.get('acceptance_rate',0),'skipped_tokens':r.get('skipped_tokens',0),'result_path':r['result_path']})
    rows.sort(key=lambda x:float(x['final_bpb']))
    with (a.out/'leaderboard.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('best exact',best_exact['arm'],best_exact['trace'][-1]['bpb']);print('best standard',best_std['arm'],best_std['trace'][-1]['bpb'])
    for r in rows[:40]:print(f"{r['arm'][:70]:70} {float(r['final_bpb']):.6f} {float(r['delta_vs_best_standard']):+.6f} z={float(r['paired_z']):+.2f} acc={float(r['acceptance_rate']):.2f} bw={r['exact_backward_calls']} v={r['virtual_steps']}")
    # final delta
    rr=rows[:35];fig,ax=plt.subplots(figsize=(11,max(4,.3*len(rr))));y=np.arange(len(rr));v=np.array([x['delta_vs_best_standard'] for x in rr]);e=2*np.array([x['paired_se'] for x in rr]);ax.barh(y,v,xerr=e);ax.axvline(0,linewidth=1);ax.set_yticks(y,[x['arm'] for x in rr]);ax.invert_yaxis();ax.set_xlabel('BPB delta vs strongest exact/LAWA standard');ax.set_title('Forward-verified causal fast-forward');fig.tight_layout();fig.savefig(a.out/'fig_final_delta.png',dpi=180);fig.savefig(a.out/'fig_final_delta.pdf');plt.close(fig)
    # acceptance vs gain
    ff=[x for x in rows if x['family']=='forward-probe']
    if ff:
        fig,ax=plt.subplots(figsize=(7,6));ax.scatter([x['acceptance_rate'] for x in ff],[-x['delta_vs_best_standard'] for x in ff])
        for x in ff:ax.annotate(x['arm'][:30],(x['acceptance_rate'],-x['delta_vs_best_standard']),fontsize=6)
        ax.axhline(0,linewidth=1);ax.set_xlabel('acceptance rate');ax.set_ylabel('improvement over best standard (-Δ BPB)');ax.set_title('Does forward verification monetize causal proposals?');fig.tight_layout();fig.savefig(a.out/'fig_acceptance_vs_gain.png',dpi=180);fig.savefig(a.out/'fig_acceptance_vs_gain.pdf');plt.close(fig)
    (a.out/'validation.json').write_text(json.dumps({'best_exact':best_exact['arm'],'best_standard':best_std['arm'],'equal_compute_spread':max(x['charged_flop_equiv'] for x in rows)-min(x['charged_flop_equiv'] for x in rows)},indent=2))
if __name__=='__main__':main()
