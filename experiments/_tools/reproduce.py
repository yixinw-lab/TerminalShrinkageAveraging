#!/usr/bin/env python3
"""Rebuild paper-facing CSV tables and data-derived plots from saved records.

CPU only. Does not train, load checkpoints, submit jobs, or contact any service.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import t

# Also works through the section-specific runpy entrypoints.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from groupwise_records import role_records, floor_records

REPO = Path(__file__).resolve().parents[2]
LAYOUT = json.loads((Path(__file__).with_name('paper_layout.json')).read_text())
SECTIONS = {s['id']: s for s in LAYOUT['sections']}
FIGS = {int(k): v for k, v in LAYOUT['figure_names'].items()}
SEEDS = [11103, 12203, 13303, 14403, 15503]
DEV_SEEDS = [6103, 7203, 8303, 9403, 10503]
OPTIMIZERS = ['native', 'pure_adamw']
FLOORS = [0.05, 0.10, 0.15]
OPT_LABEL = {'native': 'Muon+AdamW', 'pure_adamw': 'Pure AdamW'}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def summary(values: Iterable[float], normal: bool = False) -> dict[str, Any]:
    a = np.asarray(list(values), dtype=float)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError('Missing or non-finite observations')
    n = len(a)
    se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    h = (1.96 if normal else float(t.ppf(.975, n - 1))) * se if n > 1 else 0.0
    return dict(n=n, mean=float(a.mean()), ci95_halfwidth=h,
                ci95_low=float(a.mean() - h), ci95_high=float(a.mean() + h),
                positive_count=int(np.sum(a > 0)))


class Output:
    def __init__(self, root: Path, section: str | None, plots: bool):
        self.root, self.section, self.plots = root, section, plots
        self.files: list[str] = []

    def selected(self, section: str) -> bool:
        return self.section is None or section == self.section

    def csv(self, section: str, filename: str, rows: list[dict]) -> None:
        if not self.selected(section):
            return
        if not rows:
            raise ValueError(f'Empty export: {filename}')
        p = self.root / 'results' / section / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with p.open('w', newline='') as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(rows)
        self.files.append(str(p.relative_to(self.root)))

    def plot(self, number: int, suffix: str, draw) -> None:
        owners = [s['id'] for s in SECTIONS.values() if number in s['figures']]
        if not self.plots or (self.section is not None and self.section not in owners):
            return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # One chart per output: multi-panel paper figures are exported as separate panels.
        fig, ax = plt.subplots(figsize=(7.2, 4.6), layout='constrained')
        draw(fig, ax)
        stem = f'{FIGS[number]}_replot{suffix}'
        directory = self.root / 'figures' / FIGS[number]
        directory.mkdir(parents=True, exist_ok=True)
        for ext in ['pdf', 'png']:
            p = directory / f'{stem}.{ext}'
            fig.savefig(p, dpi=180)
            self.files.append(str(p.relative_to(self.root)))
        plt.close(fig)


def error_curve(ax, x, y, halfwidth, label=None, marker='o', linestyle='-'):
    ax.errorbar(x, y, yerr=halfwidth, marker=marker, linestyle=linestyle,
                capsize=3, label=label)


def add_baseline(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(0, linewidth=.8, linestyle=':')


def replicated(out: Output):
    src = REPO / 'results/_sources/d12_replicated'
    paths = sorted(src.glob('confirmation/*/*/*/holdout_results.csv'))
    if len(paths) != 30:
        raise ValueError(f'Expected 30 confirmation runs, found {len(paths)}')
    ix: dict[tuple, float] = {}
    long = []
    for p in paths:
        for r in read_csv(p):
            key = (r['optimizer'], float(r['floor']), int(r['stream_seed']), r['recipe'])
            if key in ix:
                raise ValueError(f'Duplicate confirmation row: {key}')
            ix[key] = float(r['bpb'])
            long.append(r)
    def values(opt, floor, recipe):
        return np.array([ix[(opt, floor, s, recipe)] for s in SEEDS])
    def gain(opt, floor, recipe):
        return values(opt, floor, 'raw') - values(opt, floor, recipe)
    schedule, interactions, paired_schedule = [], [], []
    for opt in OPTIMIZERS:
        for floor in FLOORS:
            raw = values(opt, floor, 'raw') - values(opt, .05, 'raw')
            tsa = values(opt, floor, 'scalar_fixed_055') - values(opt, .05, 'scalar_fixed_055')
            for kind, vals in [('raw', raw), ('tsa_alpha_055', tsa), ('interaction', raw - tsa)]:
                row = dict(optimizer=opt, active_floor=floor, quantity=kind, **summary(vals))
                (interactions if kind == 'interaction' else schedule).append(row)
            for k, seed in enumerate(SEEDS):
                paired_schedule.append(dict(optimizer=opt, floor=floor, stream_seed=seed,
                    raw_effect=raw[k], tsa_effect=tsa[k], interaction=raw[k] - tsa[k]))
    s43 = 'sec_4_3_schedule_estimator_interaction'
    out.csv(s43, 'fig_01_fig_04_sec_4_3_schedule_effects.csv', schedule)
    out.csv(s43, 'table_01_sec_4_3_paired_observations.csv', paired_schedule)
    table1 = []
    for floor in [.1, .15]:
        d = {r['quantity']: r for r in schedule + interactions if r['optimizer']=='native' and r['active_floor']==floor}
        row = dict(active_floor=floor, n=5)
        for k in ['raw', 'tsa_alpha_055', 'interaction']:
            row[k + '_mean'] = d[k]['mean']; row[k + '_ci95_halfwidth'] = d[k]['ci95_halfwidth']
        table1.append(row)
    out.csv(s43, 'table_01_sec_4_3_schedule_estimator_interaction.csv', table1)
    out.csv('sec_4_2_raw_terminal_activity', 'fig_03_sec_4_2_raw_floor_effects.csv',
            [r for r in schedule if r['optimizer']=='native' and r['quantity']=='raw'])
    out.csv('sec_4_4_adamw_replication', 'fig_04_sec_4_4_fixed_alpha_interactions.csv',
            [r for r in interactions if r['active_floor'] > .05])
    def schedule_plot(fig, ax):
        for kind, label in [('raw','Raw final iterate'), ('tsa_alpha_055','TSA (alpha=0.55)')]:
            rows = [r for r in schedule if r['optimizer']=='native' and r['quantity']==kind]
            error_curve(ax, [100*r['active_floor'] for r in rows], [r['mean'] for r in rows],
                        [r['ci95_halfwidth'] for r in rows], label)
        add_baseline(ax); ax.legend(); ax.set(xlabel='Terminal learning-rate floor (%)',
            ylabel='BPB change relative to 5% floor', title='The schedule effect depends on the returned estimator')
    out.plot(1, '', schedule_plot)
    out.plot(4, '_panel_a_native', schedule_plot)
    def raw_plot(fig, ax):
        rows = [r for r in schedule if r['optimizer']=='native' and r['quantity']=='raw']
        error_curve(ax, [100*r['active_floor'] for r in rows], [r['mean'] for r in rows], [r['ci95_halfwidth'] for r in rows])
        add_baseline(ax); ax.set(xlabel='Terminal learning-rate floor (%)',ylabel='Raw BPB change relative to 5%',title='More terminal activity worsens the raw final iterate')
    out.plot(3, '', raw_plot)
    def interactions_plot(fig, ax):
        for opt in OPTIMIZERS:
            rows=[r for r in interactions if r['optimizer']==opt and r['active_floor']>.05]
            error_curve(ax,[100*r['active_floor'] for r in rows],[r['mean'] for r in rows], [r['ci95_halfwidth'] for r in rows],OPT_LABEL[opt])
        ax.legend(); ax.set(xlabel='Active terminal floor (%)',ylabel='Paired interaction in BPB',title='Fixed alpha=0.55: interaction across optimizers')
    out.plot(4, '_panel_b_optimizers', interactions_plot)
    grid, optima, fits = [], [], []
    for opt in OPTIMIZERS:
        for floor in FLOORS:
            a = np.arange(21)/20
            gs = np.array([gain(opt,floor,f'scalar:a{x:.3f}') for x in a])
            means = gs.mean(axis=1)
            for alpha, g in zip(a,gs):
                grid.append(dict(optimizer=opt,floor=floor,alpha=alpha,**summary(g)))
            best = int(np.argmax(means))
            optima.append(dict(optimizer=opt,floor=floor,best_tested_alpha=a[best],**summary(gs[best])))
            b,c = np.linalg.lstsq(np.column_stack([a,a*a]),means,rcond=None)[0]
            r2=1-float(np.sum((means-b*a-c*a*a)**2)/np.sum((means-means.mean())**2))
            fits.append(dict(optimizer=opt,floor=floor,best_tested_alpha=a[best],linear_coefficient=b,
                             quadratic_coefficient=c,fitted_vertex=-b/(2*c),r_squared=r2))
    out.csv('sec_4_1_replicated_shrinkage','fig_02_sec_4_1_shrinkage_response.csv',[r for r in grid if r['optimizer']=='native' and r['floor']==.05])
    out.csv('app_b_1_shrinkage_across_floors','fig_05_app_b_1_scalar_grid.csv',grid)
    out.csv('app_b_1_shrinkage_across_floors','table_03_app_b_1_scalar_optima.csv',optima)
    out.csv('app_b_9_quadratic_response','table_10_app_b_9_quadratic_fits.csv',fits)
    out.csv('app_b_9_quadratic_response','fig_13_app_b_9_curve_points.csv',grid)
    def one_curve(fig, ax):
        rows=[r for r in grid if r['optimizer']=='native' and r['floor']==.05]
        error_curve(ax,[r['alpha'] for r in rows],[r['mean'] for r in rows],[r['ci95_halfwidth'] for r in rows])
        add_baseline(ax);ax.set(xlabel='Shrinkage coefficient alpha',ylabel='Paired BPB improvement over raw',title='Partial shrinkage has a replicated interior optimum')
    out.plot(2,'',one_curve)
    for opt in OPTIMIZERS:
        def multi_curve(fig,ax,opt=opt):
            for floor in FLOORS:
                rows=[r for r in grid if r['optimizer']==opt and r['floor']==floor]
                error_curve(ax,[r['alpha'] for r in rows],[r['mean'] for r in rows],[r['ci95_halfwidth'] for r in rows],f'{100*floor:g}% floor')
            add_baseline(ax);ax.legend();ax.set(xlabel='Shrinkage coefficient alpha',ylabel='Paired BPB improvement over raw',title=OPT_LABEL[opt]+': shrinkage response across floors')
        out.plot(5,'_'+opt,multi_curve)
        for floor in FLOORS:
            def fitted_curve(fig,ax,opt=opt,floor=floor):
                rows=[r for r in grid if r['optimizer']==opt and r['floor']==floor]
                fit=next(r for r in fits if r['optimizer']==opt and r['floor']==floor)
                x=np.linspace(0,1,301)
                ax.plot(x,fit['linear_coefficient']*x+fit['quadratic_coefficient']*x*x,label='Quadratic fit')
                error_curve(ax,[r['alpha'] for r in rows],[r['mean'] for r in rows],[r['ci95_halfwidth'] for r in rows], 'Observed paired mean',linestyle='none')
                ax.axvline(fit['fitted_vertex'],linestyle='--',label='Fitted vertex')
                ax.legend();ax.set(xlabel='Shrinkage coefficient alpha',ylabel='Paired BPB improvement over raw',title=f'{OPT_LABEL[opt]} | {floor*100:g}% floor | R²={fit["r_squared"]:.5f}')
            out.plot(13,f'_{opt}_floor_{round(floor*100):02d}',fitted_curve)
    # Tables 4 and 5: optimizer-specific frozen scalar is intentional here.
    table4,table5=[],[]
    for opt in OPTIMIZERS:
        for floor in FLOORS:
            for dst,recipes in [(table4,{'scalar':'scalar_dev_selected','structured':'structured_dev_selected'}),
                                (table5,{'matrix':'group_only_dev_selected','complement':'other_only_dev_selected'})]:
                row=dict(optimizer=opt,floor=floor,n=5)
                vals={k:gain(opt,floor,v) for k,v in recipes.items()}
                for k,v in vals.items():
                    st=summary(v);row[k+'_gain']=st['mean'];row[k+'_ci95_halfwidth']=st['ci95_halfwidth']
                keys=list(vals)
                diff=vals[keys[1]]-vals[keys[0]] if dst is table4 else vals[keys[0]]-vals[keys[1]]
                st=summary(diff);row['difference']=st['mean'];row['difference_ci95_halfwidth']=st['ci95_halfwidth']
                dst.append(row)
    s='app_b_5_two_group_shrinkage'
    out.csv(s,'table_04_app_b_5_structured_vs_scalar.csv',table4)
    out.csv(s,'table_05_app_b_5_matrix_vs_complement.csv',table5)
    dev=defaultdict(list)
    paths=sorted(src.glob('development/*/*/*/calibration_results.csv'))
    if len(paths)!=10:raise ValueError('Expected 10 development runs')
    for p in paths:
        rows=read_csv(p);raw=next(float(r['bpb']) for r in rows if float(r['alpha_group'])==0 and float(r['alpha_other'])==0)
        for r in rows:
            if r['family']=='structured_grid':dev[(r['optimizer'],float(r['alpha_group']),float(r['alpha_other']))].append(raw-float(r['bpb']))
    devrows=[dict(optimizer=k[0],alpha_group=k[1],alpha_other=k[2],**summary(v)) for k,v in sorted(dev.items())]
    if any(r['n']!=5 for r in devrows):raise ValueError('Incomplete development surface')
    out.csv(s,'fig_08_app_b_5_development_surfaces.csv',devrows)
    for opt in OPTIMIZERS:
        def surface(fig,ax,opt=opt):
            rows=[r for r in devrows if r['optimizer']==opt]
            x=sorted(set(r['alpha_group'] for r in rows));y=sorted(set(r['alpha_other'] for r in rows))
            ix={(r['alpha_group'],r['alpha_other']):r['mean'] for r in rows}
            arr=np.array([[ix[(a,b)] for a in x] for b in y])
            im=ax.imshow(arr,origin='lower',extent=[-.0625,1.0625,-.0625,1.0625],aspect='auto')
            fig.colorbar(im,ax=ax,label='Mean paired BPB gain over raw')
            ax.plot([0,1],[0,1],linestyle='--',label='Scalar TSA')
            ax.plot(.625,.25,marker='x',markersize=10,linestyle='none',label='Frozen selection')
            ax.legend();ax.set(xlabel='Matrix-group alpha',ylabel='Complement alpha',title=OPT_LABEL[opt]+': development surface')
        out.plot(8,'_'+opt,surface)
    def localization(fig,ax):
        labels=[f'{OPT_LABEL[r["optimizer"]]} | {100*r["floor"]:g}%' for r in table5];y=np.arange(len(labels))
        for shift,k,label in [(-.12,'matrix','Matrix group only'),(.12,'complement','Complement only')]:
            ax.errorbar([r[k+'_gain'] for r in table5],y+shift,xerr=[r[k+'_ci95_halfwidth'] for r in table5],fmt='o',capsize=3,label=label)
        ax.set_yticks(y,labels);ax.invert_yaxis();ax.legend();add_baseline(ax,False)
        ax.set(xlabel='Paired BPB improvement over raw',title='Averaging gain is concentrated in the matrix group')
    out.plot(9,'',localization)


def appendix(out: Output):
    src=REPO/'results/_sources/d12_window_estimators'
    paths=sorted(src.glob('runs/*/endpoint_results.csv'))
    if len(paths)!=5:raise ValueError('Expected five appendix trajectories')
    groups=defaultdict(list);perseed=[];rindex={}
    for p in paths:
        rows=read_csv(p);raw=next(float(r['bpb']) for r in rows if r['recipe']=='raw')
        for r in rows:
            key=r['recipe'];g=raw-float(r['bpb']);groups[key].append((r,g));rindex[(r['stream_seed'],key)]=float(r['bpb'])
            perseed.append(dict(stream_seed=r['stream_seed'],recipe=key,bpb=float(r['bpb']),gain_vs_raw=g))
    allrows=[]
    for recipe,entries in sorted(groups.items()):
        r=entries[0][0]
        allrows.append(dict(recipe=recipe,method=r['method'],k=int(r['k']),spacing=int(r['spacing']),**summary(g for _,g in entries)))
    win=[r for r in allrows if r['recipe'].startswith('tsa:')]
    out.csv('app_b_2_checkpoint_windows','fig_06_app_b_2_checkpoint_windows.csv',win)
    s='app_b_3_b_4_alternative_estimators'
    out.csv(s,'fig_07_app_b_3_b_4_all_estimators.csv',allrows)
    out.csv(s,'app_b_3_b_4_per_seed.csv',perseed)
    keys=['tsa:k8:s32:a0.550','adaptive:hybrid:8:32:1.0:all','ewa:beta0.75:k8:s32','ema:0.95:16:16','lawa:k8:s32:a1.000','swastyle:32:16']
    chosen=[next(r for r in allrows if r['recipe']==k) for k in keys]
    out.csv(s,'fig_07_app_b_3_b_4_displayed_estimators.csv',chosen)
    vals=[rindex[(str(seed),'ewa:beta0.75:k8:s32')]-rindex[(str(seed),'tsa:k8:s32:a0.550')] for seed in [1103,2203,3303,4403,5503]]
    out.csv(s,'app_b_3_paired_tsa_over_ewa.csv',[dict(contrast='TSA gain over matched EWA beta=0.75',**summary(vals))])
    def windows(fig,ax):
        ks=[4,8,16];sp=[16,32,64];ix={(r['k'],r['spacing']):r['mean'] for r in win}
        arr=np.array([[ix[(k,d)] for d in sp] for k in ks])
        im=ax.imshow(arr,aspect='auto');fig.colorbar(im,ax=ax,label='Mean BPB gain over raw')
        ax.set_xticks(range(3),sp);ax.set_yticks(range(3),ks)
        ax.set(xlabel='Checkpoint spacing s',ylabel='Checkpoint count K',title='TSA sensitivity to checkpoint count and spacing')
    out.plot(6,'',windows)
    def estimators(fig,ax):
        labels=['TSA (K=8, s=32)','Adaptive TSA','Matched EWA (beta=.75)','Checkpoint EMA','LAWA','SWA-style']
        y=np.arange(len(chosen));ax.errorbar([r['mean'] for r in chosen],y,xerr=[r['ci95_halfwidth'] for r in chosen],fmt='o',capsize=3)
        ax.set_yticks(y,labels);ax.invert_yaxis();add_baseline(ax,False);ax.set(xlabel='Paired BPB improvement over raw',title='Alternative output estimators at depth 12')
    out.plot(7,'',estimators)


def preliminary(out: Output):
    src=REPO/'results/_sources/d12_preliminary'
    ix={}
    for p in src.glob('runs/*/result.json'):
        j=json.loads(p.read_text());ix[float(j['floor'])]={r['recipe']:r for r in j['recipe_evaluations'] if r['step']==3000 and r['eval_kind']=='curve'}
    if len(ix)!=5:raise ValueError('Expected five preliminary floor arms')
    raw=np.array(ix[.05]['raw']['bpb_values'])
    table8=[]
    for r in sorted(ix[.05].values(),key=lambda r:float(r['alpha'])):
        st=summary(raw-np.array(r['bpb_values']),normal=True)
        table8.append(dict(alpha=r['alpha'],validation_bpb=r['bpb'],gain=st['mean'],ci95_halfwidth=st['ci95_halfwidth'],n_eval_batches=st['n'],interval_method='1.96 times paired evaluation-batch SE; one trajectory'))
    table9=[]
    for floor,rs in sorted(ix.items()):
        r=rs['raw'];st=summary(np.array(r['bpb_values'])-raw,normal=True)
        table9.append(dict(floor=floor,validation_bpb=r['bpb'],effect=st['mean'],ci95_halfwidth=st['ci95_halfwidth'],n_eval_batches=st['n'],interval_method='1.96 times paired evaluation-batch SE; one trajectory'))
    s='app_b_8_preliminary_sweeps'
    out.csv(s,'table_08_app_b_8_scalar_sweep.csv',table8);out.csv(s,'table_09_app_b_8_floor_sweep.csv',table9)
    x=np.array([r['floor']-.05 for r in table9]);y=np.array([r['effect'] for r in table9])
    b,c=np.linalg.lstsq(np.column_stack([x,x*x]),y,rcond=None)[0]
    r2=1-float(np.sum((y-b*x-c*x*x)**2)/np.sum((y-y.mean())**2))
    out.csv(s,'fig_12_app_b_8_descriptive_quadratic_guide.csv',[dict(linear=b,quadratic=c,r_squared=r2,interpretation='Descriptive fit only; floor response is not asserted quadratic by the theory')])
    def draw(fig,ax):
        a=np.linspace(0,.125,151)
        ax.plot(100*(a+.05),b*a+c*a*a,linestyle='--',label=f'Descriptive quadratic (R²={r2:.3f})')
        error_curve(ax,[100*r['floor'] for r in table9],y,[r['ci95_halfwidth'] for r in table9],label='Observed sweep',linestyle='none')
        ax.legend();ax.set(xlabel='Terminal learning-rate floor (%)',ylabel='Raw BPB change relative to 5%',title='Preliminary single-trajectory floor response')
    out.plot(12,'',draw)


FAMILY_LABEL={
 'scalar':'Scalar TSA','complement_split':'Embedding / hidden / unembedding / rest',
 'hidden_rest':'Hidden matrices vs. rest','attn_mlp_rest':'Attention vs. MLP',
 'square_rect_rest':'Square vs. rectangular','functional4':'Four hidden-matrix roles',
 'input_write_rest':'Input vs. write matrices','module_layer_scan':'Module / layer scan',
 'layer_scan':'Single-layer scan','adaptive_binary':'Binary trajectory statistic',
 'adaptive_affine':'Affine trajectory statistic','depth_half_rest':'Early vs. late layers',
 'depth_thirds_rest':'Depth thirds'}

def groupwise(out: Output):
    role=role_records(REPO)
    dev=role['development'];conf=role['confirmation']
    s='app_b_6_parameter_role_shrinkage'
    out.csv(s,'table_06_app_b_6_frozen_role_rule.csv',role['table6'])
    out.csv(s,'fig_10_app_b_6_development_family_winners.csv',dev)
    out.csv(s,'fig_11_app_b_6_confirmation_summary.csv',conf)
    out.csv(s,'app_b_6_all_290_development_candidates.csv',role['candidates'])
    out.csv(s,'app_b_6_confirmation_run_level.csv',role['per_run'])
    def development(fig,ax):
        rows=[r for r in dev if r['family']!='scalar'];ys=np.arange(len(rows))
        means=np.array([float(r['mean_gain_vs_scalar']) for r in rows]);lo=np.array([float(r['lo95_gain_vs_scalar']) for r in rows]);hi=np.array([float(r['hi95_gain_vs_scalar']) for r in rows])
        fig.set_size_inches(9.5,6.0)
        ax.errorbar(means,ys,xerr=np.vstack([means-lo,hi-means]),fmt='o',capsize=3)
        ax.set_yticks(ys,[FAMILY_LABEL.get(r['family'],r['family']) for r in rows]);ax.invert_yaxis();add_baseline(ax,False)
        ax.set(xlabel='Development mean BPB gain over scalar TSA',title='Development search over parameter-group structures')
    out.plot(10,'',development)
    def confirmation(fig,ax):
        fam=[r['family'] for r in dev if r['family']!='scalar'];ys=np.arange(len(fam));fig.set_size_inches(9.5,6.0)
        for opt,shift,marker in [('native',-.14,'o'),('pure_adamw',.14,'s')]:
            rows=[next(r for r in conf if r['optimizer']==opt and r['family']==f and r['baseline_id']=='scalar|alpha=0.500') for f in fam]
            ax.errorbar([float(r['mean_gain_vs_baseline']) for r in rows],ys+shift,xerr=[float(r['halfwidth95']) for r in rows],fmt=marker,capsize=3,label=OPT_LABEL[opt])
        ax.set_yticks(ys,[FAMILY_LABEL.get(f,f) for f in fam]);ax.invert_yaxis();add_baseline(ax,False);ax.legend()
        ax.set(xlabel='Confirmation BPB gain over frozen scalar alpha=0.50',title='Frozen groupwise rules on fresh confirmation trajectories')
    out.plot(11,'',confirmation)
    floor=floor_records(REPO)
    s='app_b_7_groupwise_terminal_floors'
    out.csv(s,'table_07_app_b_7_groupwise_floor_intervention.csv',floor['table7'])
    out.csv(s,'table_07_app_b_7_paired_stream_effects.csv',floor['effects'])
    out.csv(s,'app_b_7_per_seed_all_arms.csv',floor['per_seed'])
    out.csv(s,'app_b_7_all_paired_contrasts.csv',floor['contrasts'])


def d22(out: Output):
    src = REPO/'results/_sources/d22_runs'
    rows = read_csv(src/'existing_controls_n3.csv')
    for p in sorted((src/'d22_alpha70/nanochat-causal-speedrun-v2/results').glob('n3_floor15_direct9_tsa0700_s*/tsa_core.json')):
        j = json.loads(p.read_text())
        seed = int(p.parent.name.rsplit('_s', 1)[1])
        rows.append(dict(seed=seed, configuration='floor15_direct9_tsa0700',
                         time_min=j['training_minutes'], val_bpb=j['val_bpb'], core=j['core_metric']))
    if len(rows) != 9:
        raise ValueError(f'Expected nine D22 source rows, found {len(rows)}')
    configs=['pr830_early_stop_9p4_raw','pr830_direct_9p0_raw','floor15_direct9_tsa0700']
    labels=['PR #830 early stop','PR #830 direct 9.0','15% floor + TSA']
    table=[];canonical_per_seed=[]
    for config,label in zip(configs,labels):
        cells=[r for r in rows if r['configuration']==config]
        if sorted(int(r['seed']) for r in cells)!=[42,43,44]:raise ValueError('Unexpected D22 seed set')
        row=dict(configuration=label,configuration_id=config,schedule_calibration=9.4 if 'early_stop' in config else 9.0,
                 terminal_floor_percent=15 if config==configs[-1] else 0,endpoint_step=10172,n=3)
        for metric in ['time_min','val_bpb','core']:
            st=summary(float(r[metric]) for r in cells)
            row[metric+'_mean']=st['mean'];row[metric+'_ci95_halfwidth']=st['ci95_halfwidth']
        row['core_basis']='recomputed_from_saved_records'
        table.append(row)
        for r in cells:
            canonical_per_seed.append(dict(seed=r['seed'],configuration=config,time_min=r['time_min'],val_bpb=r['val_bpb'],
                core=r['core'], core_status='saved_run_record'))
    s='sec_4_5_d22_scale_transfer'
    out.csv(s,'table_02_sec_4_5_d22_scale_transfer.csv',table)
    out.csv(s,'table_02_sec_4_5_time_bpb_and_control_core_per_seed.csv',canonical_per_seed)
    if out.selected(s):
        p=out.root/'results'/s/'table_02_sec_4_5_d22_scale_transfer.tex'
        lines=[r'% Means and 95% Student-t intervals are computed from the saved per-seed records.',r'\begin{table}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{3pt}',r'\caption{Matched-endpoint depth-22 comparison over three seeds. All configurations stop at step 10,172. Values are means $\pm$ 95\% intervals.}',r'\label{tab:d22-matched}',r'\begin{tabular}{lccccc}',r'\toprule',r'Configuration & Calibration & Floor & Time (min) & Val. BPB & CORE \\',r'\midrule']
        for r in table:
            label=r['configuration'].replace('#',r'\#').replace('%',r'\%');floor=r'$15\%$' if r['terminal_floor_percent'] else 'none'
            lines.append(f"{label} & {r['schedule_calibration']:.1f} & {floor} & ${r['time_min_mean']:.3f} \\pm {r['time_min_ci95_halfwidth']:.3f}$ & ${r['val_bpb_mean']:.6f} \\pm {r['val_bpb_ci95_halfwidth']:.6f}$ & ${r['core_mean']:.4f} \\pm {r['core_ci95_halfwidth']:.4f}$ " + r"\\")
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
        p.write_text('\n'.join(lines)+'\n');out.files.append(str(p.relative_to(out.root)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section',choices=sorted(SECTIONS),default=None)
    parser.add_argument('--output-root',type=Path,default=REPO,help='Root for generated results/ and figures/; source files are never overwritten')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    out=Output(args.output_root.resolve(),args.section,not args.no_plots)
    funcs={'replicated':replicated,'appendix':appendix,'preliminary':preliminary,'d22':d22,'role':groupwise,'floor':groupwise}
    if args.section:funcs[SECTIONS[args.section]['dataset']](out)
    else:
        for f in [replicated,appendix,preliminary,groupwise,d22]:f(out)
    print(f'Generated {len(out.files)} files under {out.root}')
    for path in out.files:print(path)

if __name__=='__main__':
    main()
