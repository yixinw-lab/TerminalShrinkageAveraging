#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from statistics import mean, stdev

TCRIT_N3 = 4.302652729911275  # two-sided 95%, df=2

def ci(xs):
    xs=list(map(float,xs)); m=mean(xs)
    if len(xs)<2: return m, float('nan'), float('nan')
    sd=stdev(xs)
    t=TCRIT_N3 if len(xs)==3 else 1.96
    return m,sd,t*sd/math.sqrt(len(xs))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); args=ap.parse_args()
    rows=[]
    for p in sorted((args.root/'runs').glob('*/metrics.json')):
        d=json.loads(p.read_text()); d['_path']=str(p); rows.append(d)
    if len(rows)!=6:
        raise SystemExit(f"expected 6 metrics.json files, found {len(rows)}")
    def arm(d): return 'control' if float(d['terminal_lr_floor']) < 0 else 'treatment'
    for d in rows: d['arm']=arm(d)
    seeds=sorted({int(d['experiment_seed']) for d in rows})
    if seeds != [42,43,44]: raise SystemExit(f"unexpected seeds: {seeds}")
    by={(d['arm'],int(d['experiment_seed'])):d for d in rows}
    keys=sorted(set.intersection(*[set(d['metrics_bpb']) for d in rows]))
    out=args.root/'analysis'; out.mkdir(parents=True,exist_ok=True)

    with (out/'arm_summary.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['arm','estimator','n','mean_bpb','sd','ci95_half'])
        for a in ['control','treatment']:
            for k in keys:
                xs=[by[(a,s)]['metrics_bpb'][k] for s in seeds]
                m,sd,h=ci(xs); w.writerow([a,k,len(xs),f'{m:.9f}',f'{sd:.9f}',f'{h:.9f}'])

    contrasts=[]
    for s in seeds:
        c=by[('control',s)]['metrics_bpb']; t=by[('treatment',s)]['metrics_bpb']
        raw_effect=t['raw']-c['raw']
        for est in ['tsa_alpha_0.55','tsa_alpha_0.587','tsa_alpha_1']:
            est_effect=t[est]-c[est]
            contrasts.append((s,est,raw_effect,est_effect,raw_effect-est_effect,c['raw']-t[est],t['raw']-t[est],c['raw']-c[est]))
    with (out/'paired_effects.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['seed','estimator','raw_schedule_effect','estimator_schedule_effect','interaction_raw_minus_estimator','full_recipe_gain_control_raw_minus_treatment_estimator','treatment_same_trajectory_gain','control_same_trajectory_gain'])
        for r in contrasts: w.writerow([r[0],r[1],*[f'{x:.9f}' for x in r[2:]]])

    lines=['D22 DeltaAI paired replication','================================','Hardware: 2 DeltaAI nodes x 4 GH200/H100 GPUs per training job','Timing is hardware-specific and must not be compared to the single-node NanoChat leaderboard.','']
    for est in ['tsa_alpha_0.55','tsa_alpha_0.587','tsa_alpha_1']:
        rr=[r for r in contrasts if r[1]==est]
        labels=[('raw schedule effect',2),('estimator schedule effect',3),('interaction raw-estimator',4),('full recipe gain',5),('treatment same-trajectory gain',6),('control same-trajectory gain',7)]
        lines.append(f'[{est}]')
        for label,idx in labels:
            xs=[r[idx] for r in rr]; m,sd,h=ci(xs)
            lines.append(f'{label}: {m:+.9f} +/- {h:.9f} (sd={sd:.9f}; signs +={sum(x>0 for x in xs)}/3)')
        lines.append('')
    # Hardware consistency diagnostic: seed 42 treatment vs historical RunPod treatment BPB.
    hist=0.720050
    if ('treatment',42) in by and 'tsa_alpha_0.587' in by[('treatment',42)]['metrics_bpb']:
        v=by[('treatment',42)]['metrics_bpb']['tsa_alpha_0.587']
        lines += [f'seed42 treatment TSA(.587) DeltaAI BPB: {v:.6f}',f'historical RunPod TSA(.587) BPB: {hist:.6f}',f'difference DeltaAI-RunPod: {v-hist:+.6f}','This hardware comparison is descriptive only; do not pool it into the paired estimate.']
    (out/'DIGEST.txt').write_text('\n'.join(lines)+'\n')
    print((out/'DIGEST.txt').read_text())

if __name__=='__main__': main()
