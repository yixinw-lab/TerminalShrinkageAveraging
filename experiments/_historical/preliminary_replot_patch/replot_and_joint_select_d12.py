#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from typing import Any


def read_env(path: Path) -> dict[str,str]:
    out={}
    if not path.exists(): return out
    for line in path.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k,v=line.split('=',1); out[k.strip()]=v.strip()
    return out

def alpha_key(a: float) -> str:
    return f"tsa:{a:.3f}"

def endpoint_rows(result: dict[str,Any], kind: str):
    T=int(result['train_steps'])
    return [r for r in result['recipe_evaluations'] if r.get('eval_kind')==kind and int(r.get('step',-1))==T]

def curve_rows(result: dict[str,Any]):
    return [r for r in result['recipe_evaluations'] if r.get('eval_kind')=='curve']

def row_for_alpha(rows, a: float):
    recipe='raw' if abs(a)<1e-12 else alpha_key(a)
    m=[r for r in rows if r.get('recipe')==recipe]
    if len(m)!=1: raise RuntimeError(f"expected one {recipe}, got {len(m)}")
    return m[0]

def load_results(root: Path):
    out={}
    for p in sorted(root.glob('runs/floor_*pct/result.json')):
        r=json.loads(p.read_text()); out[float(r['floor'])]=r
    if not out: raise RuntimeError(f'no result.json files under {root}/runs')
    return out

def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def paired_ci(row: dict) -> float:
    try: return 1.96*float(row.get('paired_se_vs_raw',0) or 0)
    except: return float('nan')

def get_curve(result: dict[str,Any], a: float):
    rows=curve_rows(result); pts=[]
    for step in sorted({int(r['step']) for r in rows}):
        sr=[r for r in rows if int(r['step'])==step]
        rr=row_for_alpha(sr,a); pts.append((step,float(rr['bpb'])))
    return pts

def auto_ylim(series, xlo=None):
    vals=[]
    for pts in series:
        vals.extend(y for x,y in pts if xlo is None or x>=xlo)
    lo,hi=min(vals),max(vals); span=hi-lo
    pad=max(span*0.18, 0.00045)
    return lo-pad,hi+pad

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('outputs/optimization_paper/paper_main_d12_v1'))
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args()
    root=args.root.resolve(); out=(args.out or (root/'paper_figures_v2')).resolve(); out.mkdir(parents=True,exist_ok=True)
    results=load_results(root)
    baseline_floor=min(results,key=lambda x:abs(x-.05)); baseline=results[baseline_floor]
    env=read_env(root/'FROZEN.env')
    base_alpha=float(env.get('SELECTED_ALPHA','nan'))
    if not math.isfinite(base_alpha):
        ars=endpoint_rows(baseline,'alpha_select'); cand=[]
        for a in baseline['alpha_grid']:
            cand.append((float(row_for_alpha(ars,float(a))['bpb']),float(a)))
        base_alpha=min(cand)[1]

    # Joint selection uses ONLY floor_select, over all predeclared floors and alphas.
    joint=[]
    for floor,r in sorted(results.items()):
        er=endpoint_rows(r,'floor_select')
        for a in r['alpha_grid']:
            rr=row_for_alpha(er,float(a))
            joint.append({'floor':float(floor),'floor_percent':100*float(floor),'alpha':float(a),'selection_bpb':float(rr['bpb'])})
    best=min(joint,key=lambda z:(z['selection_bpb'],z['floor'],z['alpha']))
    joint_floor=float(best['floor']); joint_alpha=float(best['alpha']); joint_result=results[joint_floor]
    for r in joint: r['selected']=int(abs(r['floor']-joint_floor)<1e-12 and abs(r['alpha']-joint_alpha)<1e-12)
    write_csv(out/'joint_selection.csv',joint)
    (out/'JOINT_FROZEN.env').write_text(f'SELECTED_ALPHA={joint_alpha:.6f}\nSELECTED_FLOOR={joint_floor:.6f}\nPRIMARY_K=8\nPRIMARY_SPACING=32\nBASELINE_FLOOR=0.050000\n')

    # Export actual curve data so future plotting does not require result.json.
    curve_export=[]
    wanted={(baseline_floor,0.0,'5% Raw'),(baseline_floor,base_alpha,f'5% TSA alpha={base_alpha:.2f}'),(baseline_floor,1.0,'5% LAWA'),
            (joint_floor,0.0,f'{100*joint_floor:g}% Raw'),(joint_floor,joint_alpha,f'{100*joint_floor:g}% TSA alpha={joint_alpha:.2f}')}
    for floor,a,label in sorted(wanted):
        for step,bpb in get_curve(results[floor],a): curve_export.append({'floor':floor,'alpha':a,'label':label,'step':step,'bpb':bpb})
    write_csv(out/'main_curve_data.csv',curve_export)

    # Main tables.
    hold=endpoint_rows(baseline,'holdout'); raw=row_for_alpha(hold,0.0)
    methods=[('Raw iterate','raw'),(f'TSA (alpha={base_alpha:.2f})',alpha_key(base_alpha)),('Uniform LAWA',alpha_key(1.0)),
             ('Checkpoint EMA','ema:0.95:16:16'),('SWA-style late average','swastyle:32:16')]
    t1=[]
    for label,recipe in methods:
        rr=next(x for x in hold if x['recipe']==recipe)
        improvement=float(raw['bpb'])-float(rr['bpb'])
        t1.append({'estimator':label,'holdout_bpb':float(rr['bpb']),'improvement_vs_raw':improvement,'paired_ci95':paired_ci(rr) if recipe!='raw' else 0.0})
    write_csv(out/'table1_averaging_v2.csv',t1)
    with (out/'table1_averaging_v2.tex').open('w') as f:
        f.write('\\begin{tabular}{lrr}\n\\toprule\nEstimator & Holdout BPB $\\downarrow$ & Improvement vs. raw $\\uparrow$ \\\\\n\\midrule\n')
        bestb=min(r['holdout_bpb'] for r in t1)
        for r in t1:
            b=f"\\textbf{{{r['holdout_bpb']:.6f}}}" if abs(r['holdout_bpb']-bestb)<1e-12 else f"{r['holdout_bpb']:.6f}"
            g='---' if r['estimator']=='Raw iterate' else f"{r['improvement_vs_raw']:+.6f} $\\pm$ {r['paired_ci95']:.6f}"
            name=r['estimator'].replace('alpha=', '$\\alpha=').replace(')', '$)') if r['estimator'].startswith('TSA') else r['estimator']
            f.write(f'{name} & {b} & {g} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

    # Joint headline table.
    hb=endpoint_rows(baseline,'holdout'); hj=endpoint_rows(joint_result,'holdout')
    rows3=[]
    for floor,res,alpha in [(baseline_floor,baseline,base_alpha),(joint_floor,joint_result,joint_alpha)]:
        h=endpoint_rows(res,'holdout'); rr=row_for_alpha(h,0.0); tt=row_for_alpha(h,alpha)
        rows3.append({'schedule':f'{100*floor:g}% floor','alpha':alpha,'raw_bpb':float(rr['bpb']),'tsa_bpb':float(tt['bpb']),'tsa_gain':float(rr['bpb'])-float(tt['bpb'])})
    write_csv(out/'table3_joint_v2.csv',rows3)

    # Fixed-alpha interaction control at baseline alpha.
    hsel=endpoint_rows(joint_result,'holdout');
    rb=row_for_alpha(hb,0.0); tb=row_for_alpha(hb,base_alpha); rj=row_for_alpha(hsel,0.0); tj_fixed=row_for_alpha(hsel,base_alpha)
    fixed=[
      {'metric':'Raw floor effect (joint floor - 5%)','delta_bpb':float(rj['bpb'])-float(rb['bpb'])},
      {'metric':f'TSA alpha={base_alpha:.2f} floor effect (joint floor - 5%)','delta_bpb':float(tj_fixed['bpb'])-float(tb['bpb'])},
      {'metric':'TSA gain at 5%','delta_bpb':float(rb['bpb'])-float(tb['bpb'])},
      {'metric':f'TSA gain at {100*joint_floor:g}%','delta_bpb':float(rj['bpb'])-float(tj_fixed['bpb'])},
    ]
    write_csv(out/'fixed_alpha_interaction_control.csv',fixed)

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8.2,'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.2,
                         'axes.linewidth':0.8,'pdf.fonttype':42,'ps.fonttype':42})
    def title(ax,t,st):
        ax.set_title(t,fontsize=9.2,fontweight='bold',pad=13)
        ax.text(.5,1.015,st,transform=ax.transAxes,ha='center',va='bottom',fontsize=7.2,fontweight='bold')
    def save(fig,name):
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight'); fig.savefig(out/f'{name}.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    def plot_series(ax, series, zoom=False):
        for pts,label,ls,lw in series:
            x=[p[0] for p in pts]; y=[p[1] for p in pts]
            ax.plot(x,y,label=label,linestyle=ls,linewidth=lw)
            ax.scatter([x[-1]],[y[-1]],s=13,zorder=4)
        ax.set_xlabel('Optimizer step'); ax.set_ylabel('Validation BPB'); ax.grid(True,alpha=.18,linewidth=.55)
        if zoom:
            xlo=2480; ax.set_xlim(xlo,3010); ax.set_ylim(*auto_ylim([p[0] for p in series],xlo=xlo))
        else:
            ax.margins(x=.015,y=.055)
        ax.legend(frameon=False,loc='best',handlelength=2.3)

    # Experiment 1: baseline estimator ablation.
    e1=[(get_curve(baseline,0.0),'Raw','--',1.45),(get_curve(baseline,base_alpha),f'TSA ($\\alpha={base_alpha:.2f}$)','-',2.0),(get_curve(baseline,1.0),'LAWA ($\\alpha=1$)',':',1.65)]
    fig,ax=plt.subplots(figsize=(3.15,2.18)); plot_series(ax,e1,False); title(ax,'Partial averaging improves the output model','Muon+AdamW · 5% terminal floor'); save(fig,'fig1_averaging_full_v2')
    fig,ax=plt.subplots(figsize=(3.15,2.18)); plot_series(ax,e1,True); title(ax,'TSA retains the best terminal output','Final 500 optimizer steps · same training trajectory'); save(fig,'fig1_averaging_zoom_v2')

    # Experiment 2: raw clamp effect, using joint-selected floor.
    e2=[(get_curve(baseline,0.0),'Raw · 5% floor','--',1.7),(get_curve(joint_result,0.0),f'Raw · {100*joint_floor:g}% floor','-',1.9)]
    fig,ax=plt.subplots(figsize=(3.15,2.18)); plot_series(ax,e2,False); title(ax,'A more active tail worsens the raw endpoint','Muon+AdamW · no output averaging'); save(fig,'fig2_clamp_full_v2')
    fig,ax=plt.subplots(figsize=(3.15,2.18)); plot_series(ax,e2,True); title(ax,'The schedule penalty appears late in training','Terminal 500 optimizer steps · raw iterates'); save(fig,'fig2_clamp_zoom_v2')

    # Experiment 3: co-designed baseline vs joint recipe. The fixed-alpha control is exported separately.
    e3=[(get_curve(baseline,0.0),'5% · Raw','--',1.25),(get_curve(baseline,base_alpha),f'5% · TSA ($\\alpha={base_alpha:.2f}$)','-',1.65),
        (get_curve(joint_result,0.0),f'{100*joint_floor:g}% · Raw','--',1.45),(get_curve(joint_result,joint_alpha),f'{100*joint_floor:g}% · TSA ($\\alpha={joint_alpha:.2f}$)','-',2.05)]
    fig,ax=plt.subplots(figsize=(3.25,2.28)); plot_series(ax,e3,False); title(ax,'Schedule and output estimator should be co-designed',f'Joint selection chooses {100*joint_floor:g}% floor and $\\alpha={joint_alpha:.2f}$'); save(fig,'fig3_joint_full_v2')
    fig,ax=plt.subplots(figsize=(3.25,2.28)); plot_series(ax,e3,True); title(ax,'Averaging unlocks the more active terminal trajectory','Terminal 500 optimizer steps · joint recipe'); save(fig,'fig3_joint_zoom_v2')

    # Alpha selection provenance on alpha_select.
    ar=endpoint_rows(baseline,'alpha_select'); apoints=[]
    for a in baseline['alpha_grid']:
        rr=row_for_alpha(ar,float(a)); apoints.append((float(a),float(rr['bpb'])))
    fig,ax=plt.subplots(figsize=(3.15,2.12)); ax.plot([x for x,y in apoints],[y for x,y in apoints],marker='o',markersize=3,linewidth=1.7); ax.axvline(base_alpha,ls='--',lw=1,alpha=.7)
    title(ax,'Selecting the shrinkage strength','Baseline 5% schedule · designated selection split'); ax.set_xlabel('Shrinkage $\\alpha$'); ax.set_ylabel('Validation BPB'); ax.grid(True,alpha=.18,linewidth=.55); ax.margins(x=.03,y=.12); save(fig,'figS_alpha_selection_v2')

    # Joint selection response on floor_select, NOT holdout.
    fig,ax=plt.subplots(figsize=(3.25,2.2))
    for floor,r in sorted(results.items()):
        er=endpoint_rows(r,'floor_select'); pts=[(float(a),float(row_for_alpha(er,float(a))['bpb'])) for a in r['alpha_grid']]
        ax.plot([x for x,y in pts],[y for x,y in pts],marker='o',markersize=2.4,linewidth=1.25,label=f'{100*floor:g}%')
    ax.scatter([joint_alpha],[best['selection_bpb']],s=30,zorder=5)
    title(ax,'Jointly selecting shrinkage and terminal activity','Designated floor-selection split · lower is better'); ax.set_xlabel('Shrinkage $\\alpha$'); ax.set_ylabel('Validation BPB'); ax.grid(True,alpha=.18,linewidth=.55); ax.legend(title='LR floor',frameon=False,ncol=2,fontsize=6.6,title_fontsize=6.6); ax.margins(x=.02,y=.08); save(fig,'figS_joint_alpha_floor_selection_v2')

    summary=(f'Baseline alpha selected on alpha_select: {base_alpha:.3f}\n'
             f'Joint pair selected on floor_select: alpha={joint_alpha:.3f}, floor={100*joint_floor:.2f}%\n'
             f'Joint selection BPB: {best["selection_bpb"]:.9f}\n'
             f'Outputs: {out}\n')
    (out/'SUMMARY.txt').write_text(summary); print(summary)

if __name__=='__main__': main()
