#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

TCRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}

def ci_half(vals):
    a=np.asarray(vals,dtype=float); n=len(a)
    if n < 2: return 0.0
    return float(TCRIT.get(n,1.96)*a.std(ddof=1)/math.sqrt(n))

def write_csv(path, rows):
    rows=list(rows); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text(""); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)

def load(root):
    out=[]
    for p in sorted((root/'runs').glob('seed*/endpoint_results.csv')):
        seed=int(p.parent.name.replace('seed',''))
        rows=list(csv.DictReader(p.open()))
        raw=next(r for r in rows if r['recipe']=='raw')
        raw_bpb=float(raw['bpb'])
        for r in rows:
            rr=dict(r); rr['seed']=seed; rr['bpb']=float(r['bpb']); rr['gain_vs_raw']=raw_bpb-float(r['bpb'])
            out.append(rr)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--repo',type=Path,required=True); a=ap.parse_args()
    root=a.root.resolve(); fig=root/'figures'; fig.mkdir(parents=True,exist_ok=True)
    rows=load(root)
    if not rows: raise SystemExit('no endpoint results found')
    by=defaultdict(list)
    for r in rows: by[r['recipe']].append(r)
    summary=[]
    for recipe,rs in sorted(by.items()):
        bpbs=[r['bpb'] for r in rs]; gains=[r['gain_vs_raw'] for r in rs]
        x=rs[0]
        summary.append({
            'recipe':recipe,'method':x['method'],'k':x.get('k'),'spacing':x.get('spacing'),'alpha':x.get('alpha'),'beta':x.get('beta'),
            'n':len(rs),'mean_bpb':float(np.mean(bpbs)),'bpb_ci95_half':ci_half(bpbs),
            'mean_gain_vs_raw':float(np.mean(gains)),'gain_ci95_half':ci_half(gains),
        })
    write_csv(fig/'averaging_family_summary.csv',summary)
    write_csv(fig/'averaging_family_per_seed.csv',rows)
    ks=[r for r in summary if r['recipe'].startswith('tsa:k')]
    write_csv(fig/'ks_summary.csv',ks)

    # Copy/summarize existing optimizer/adaptive experiment artifacts if present.
    old=a.repo/'outputs/optimization_paper/averaging_curves_d12_s5_v1/figures'
    copied=[]
    for name in ('final_summary.csv','final_seed_results.csv','DIGEST.txt'):
        src=old/name
        if src.exists():
            dst=fig/f'optimizer_adaptive_{name}'; dst.write_bytes(src.read_bytes()); copied.append(str(dst))

    # Cheap empirical check of the quadratic alpha model on the clean baseline holdout sweep.
    alpha_csv=a.repo/'outputs/optimization_paper/paper_main_d12_v1/runs/floor_05p00pct/recipe_evaluations.csv'
    fit_txt=[]
    if alpha_csv.exists():
        rr=list(csv.DictReader(alpha_csv.open()))
        h=[x for x in rr if x.get('eval_kind')=='holdout' and int(float(x.get('step',-1)))==3000]
        raw=next((x for x in h if x.get('recipe')=='raw'),None)
        pts=[]
        if raw:
            rb=float(raw['bpb'])
            for x in h:
                rec=x.get('recipe','')
                if rec.startswith('tsa:'):
                    alpha=float(rec.split(':')[1]); gain=rb-float(x['bpb']); pts.append((alpha,gain))
            if pts:
                X=np.asarray([[aa,aa*aa] for aa,_ in pts],float); y=np.asarray([g for _,g in pts],float)
                coef=np.linalg.lstsq(X,y,rcond=None)[0]; pred=X@coef
                ssres=float(((y-pred)**2).sum()); sst=float(((y-y.mean())**2).sum()); r2=1-ssres/sst if sst>0 else float('nan')
                b,c=map(float,coef); vertex=-b/(2*c) if c<0 else float('nan')
                write_csv(fig/'alpha_quadratic_fit.csv',[{'linear':b,'quadratic':c,'r2':r2,'predicted_alpha_vertex':vertex,'n':len(pts)}])
                write_csv(fig/'alpha_quadratic_fit_points.csv',[{'alpha':aa,'observed_gain':gg,'predicted_gain':float(b*aa+c*aa*aa),'residual':float(gg-(b*aa+c*aa*aa))} for aa,gg in pts])
                fit_txt=[f'quadratic fit R2={r2:.6f}',f'predicted alpha vertex={vertex:.4f}',f'linear={b:.8g} quadratic={c:.8g}']

    # Plain-text digest.
    primary=next((x for x in summary if x['recipe'].startswith('tsa:k8:s32:a0.550')),None)
    ema=next((x for x in summary if x['recipe']=='ema:0.95:16:16'),None)
    swa=next((x for x in summary if x['recipe']=='swastyle:32:16'),None)
    lines=['D12 appendix averaging sensitivity','==================================',f'n seeds: {max(x["n"] for x in summary)}']
    if primary: lines.append(f'TSA K8 s32 alpha .55 gain vs raw: {primary["mean_gain_vs_raw"]:+.6f} +/- {primary["gain_ci95_half"]:.6f}')
    if ema: lines.append(f'Checkpoint EMA gain vs raw: {ema["mean_gain_vs_raw"]:+.6f} +/- {ema["gain_ci95_half"]:.6f}')
    if swa: lines.append(f'SWA-style gain vs raw: {swa["mean_gain_vs_raw"]:+.6f} +/- {swa["gain_ci95_half"]:.6f}')
    lines += fit_txt
    lines.append(f'copied optimizer/adaptive artifacts: {len(copied)}')
    (fig/'DIGEST.txt').write_text('\n'.join(lines)+'\n')
    print((fig/'DIGEST.txt').read_text())

if __name__=='__main__': main()
