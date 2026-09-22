#!/usr/bin/env python3
"""Recompute B.6/B.7 from saved runs; never modify the saved inputs.

Original CPU summary functions are used to retain the original t-critical
rounding (B.6: 2.365; B.7: 2.364624 for eight streams). No GPU code is imported.
The B.6 development freeze is replayed in a temporary directory, not retuned on
confirmation data. Its complete JSON object must equal the saved frozen object.
"""
from __future__ import annotations
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import numpy as np


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def original_tool(repo, name):
    path = repo / 'experiments/_shared/d12_training/tools' / (name + '.py')
    spec = importlib.util.spec_from_file_location('_tsa_original_' + name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not load analysis tool: ' + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Checks:
    def __init__(self):
        self.rows = []

    def require(self, name, condition, detail=''):
        self.rows.append(dict(check=name, status='PASS' if condition else 'FAIL', detail=str(detail)))
        if not condition:
            raise ValueError(name + ': ' + str(detail))

    def matching_csv(self, label, calculated, saved, keys, fields, atol=1e-12):
        def index(rows):
            result = {tuple(str(r[k]) for k in keys): r for r in rows}
            self.require(label + ':unique_keys', len(result) == len(rows), len(rows))
            return result
        a, b = index(calculated), index(saved)
        self.require(label + ':same_keys', set(a) == set(b), len(a))
        error = 0.0
        for key in a:
            for field in fields:
                x, y = float(a[key][field]), float(b[key][field])
                self.require(label + ':finite', math.isfinite(x) and math.isfinite(y), field)
                error = max(error, abs(x - y))
        self.require(label + ':values', error <= atol, 'max absolute difference=' + str(error))


def check_source_manifest(repo, root, checks):
    harness = repo / 'experiments/_shared/d12_training'
    for line in (root / 'SOURCE_MANIFEST.sha256').read_text().splitlines():
        sha, relative = line.split(maxsplit=1)
        path = harness / relative.strip()
        checks.require(root.name + ':source:' + relative, path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha)


def role_records(repo, checks=None):
    repo = Path(repo)
    checks = checks if checks is not None else Checks()
    root = repo / 'results/_sources/app_b_6_parameter_roles/run_archive'
    module = original_tool(repo, 'analyze_tsa_group_search_d12')
    frozen = module._load_frozen(root)
    check_source_manifest(repo, root, checks)
    dev_files = sorted(root.glob('dev/**/candidate_results.csv'))
    checks.require('B6:development_runs', len(dev_files) == 8, len(dev_files))
    dev_manifest = read_csv(root / 'DEV_MANIFEST.csv')
    expected_dev = {(r['optimizer'], int(r['stream_seed'])) for r in dev_manifest}
    observed_dev = set()
    for path in dev_files:
        meta = json.loads(path.with_name('result.json').read_text())
        observed_dev.add((meta['optimizer_label'], meta['stream_seed']))
        rows = read_csv(path)
        checks.require('B6:290_candidates:' + str(meta['stream_seed']), len(rows) == len({r['candidate_id'] for r in rows}) == 290)
        checks.require('B6:development_protocol:' + str(meta['stream_seed']),
                       meta['seed'] == 1337 and meta['terminal_floor'] == .1 and meta['search_eval_batches'] == 32
                       and meta['ledger']['optimizer_steps'] == 3000 and meta['snapshot_steps'] == list(range(2776,3001,32)))
        raw = next(float(r['bpb']) for r in rows if r['candidate_id'] == 'scalar|alpha=0.000')
        checks.require('B6:development_gains:' + str(meta['stream_seed']), all(abs((raw-float(r['bpb']))-float(r['gain_vs_raw'])) < 1e-12 for r in rows))
    checks.require('B6:development_manifest', observed_dev == expected_dev and len(observed_dev) == 8)
    # Replay selection only on the original eight development files, in isolation.
    with tempfile.TemporaryDirectory(prefix='tsa-b6-freeze-check-') as temp:
        temp = Path(temp)
        for path in dev_files:
            destination = temp / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        with contextlib.redirect_stdout(io.StringIO()):
            module.freeze(temp)
        replayed = json.loads((temp/'FROZEN_GROUP_RULES.json').read_text())
        checks.require('B6:full_development_freeze_replay', replayed == frozen, frozen['freeze_sha256'])
        dev = read_csv(temp/'DEV_FAMILY_WINNERS.csv')
        candidates = read_csv(temp/'DEV_CANDIDATE_SUMMARY.csv')
    checks.matching_csv('B6:development_summary', candidates, read_csv(root/'DEV_CANDIDATE_SUMMARY.csv'),
                        ['candidate_id'], ['mean_gain_vs_raw','se_gain_vs_raw','mean_gain_vs_scalar','se_gain_vs_scalar','lo95_gain_vs_scalar','hi95_gain_vs_scalar'])
    conf_files = sorted(root.glob('confirm/**/confirmation_results.csv'))
    checks.require('B6:confirmation_runs', len(conf_files)==16, len(conf_files))
    expected_conf = {(r['optimizer'], int(r['stream_seed'])) for r in read_csv(root/'CONFIRM_MANIFEST.csv')}
    runs = {}
    for path in conf_files:
        meta=json.loads(path.with_name('result.json').read_text()); key=(meta['optimizer_label'],meta['stream_seed'])
        checks.require('B6:unique_confirmation:'+str(key), key not in runs)
        rows=read_csv(path); index={r['candidate_id']:r for r in rows}
        checks.require('B6:unique_candidate_rows:'+str(key), len(index)==len(rows))
        checks.require('B6:confirmation_protocol:'+str(key), meta['frozen_rules_sha256']==frozen['freeze_sha256']
                       and meta['terminal_floor']==.1 and meta['holdout_eval_batches']==96 and meta['seed']==1337
                       and meta['ledger']['optimizer_steps']==3000 and meta['snapshot_steps']==list(range(2776,3001,32)))
        runs[key]=(meta,index)
    checks.require('B6:confirmation_manifest',set(runs)==expected_conf)
    seeds=sorted(s for opt,s in runs if opt=='native')
    checks.require('B6:eight_paired_optimizer_streams',len(seeds)==8 and seeds==sorted(s for opt,s in runs if opt=='pure_adamw'))
    for seed in seeds:
        checks.require('B6:paired_training_permutation:'+str(seed), runs[('native',seed)][0]['permutation_fingerprint']==runs[('pure_adamw',seed)][0]['permutation_fingerprint'])
    conf=[]; per_run=[]
    for optimizer in ('native','pure_adamw'):
        baseline=frozen['frozen_scalar']['candidate_id'] if optimizer=='native' else 'scalar|alpha=0.500'
        for rule in frozen['selected_rules']:
            cid=rule['candidate_id']; diffs=[]
            for seed in seeds:
                row=runs[(optimizer,seed)][1]
                checks.require('B6:candidate_available:'+optimizer+':'+str(seed)+':'+cid, cid in row and baseline in row)
                if row[cid]['spec_json']:
                    checks.require('B6:frozen_spec:'+optimizer+':'+str(seed)+':'+cid,json.loads(row[cid]['spec_json'])==rule['spec'])
                gain=float(row[baseline]['bpb'])-float(row[cid]['bpb']);diffs.append(gain)
                per_run.append(dict(optimizer=optimizer,stream_seed=seed,baseline_id=baseline,candidate_id=cid,gain_vs_baseline=gain))
            stat=module._summary(diffs)
            conf.append(dict(optimizer=optimizer,baseline_id=baseline,candidate_id=cid,family=rule['family'],n=stat['n'],
                             mean_gain_vs_baseline=stat['mean'],se=stat['se'],halfwidth95=stat['halfwidth95'],lo95=stat['lo95'],hi95=stat['hi95']))
    checks.matching_csv('B6:confirmation_summary',conf,read_csv(root/'figures/confirmation_summary.csv'),
                        ['optimizer','baseline_id','candidate_id'],['n','mean_gain_vs_baseline','se','halfwidth95','lo95','hi95'])
    checks.matching_csv('B6:run_level',per_run,read_csv(root/'figures/confirmation_run_level.csv'),
                        ['optimizer','stream_seed','baseline_id','candidate_id'],['gain_vs_baseline'])
    coeff=frozen['overall_best']['spec']['params']
    checks.require('B6:table6_selected_role_rule',frozen['overall_best']['family']=='complement_split' and coeff['other']==coeff['scalar'])
    table6=[dict(parameter_group=label,alpha=coeff[key],newest_checkpoint_weight=1-coeff[key]+coeff[key]/8,
                 source_basis='Saved FROZEN_GROUP_RULES.json; development selection replay verified')
            for label,key in [('Embedding parameters','embedding'),('Hidden matrices','hidden'),('Unembedding matrix','unembed'),('Scalars and other parameters','scalar')]]
    return dict(development=dev,candidates=candidates,confirmation=conf,per_run=per_run,table6=table6,checks=checks.rows)


def floor_records(repo, checks=None):
    repo=Path(repo);checks=checks if checks is not None else Checks()
    root=repo/'results/_sources/app_b_7_groupwise_floor/run_archive'
    module=original_tool(repo,'analyze_groupwise_floor_d12')
    check_source_manifest(repo,root,checks)
    protocol,seeds,data=module._load(root) # includes pairing, duplicate and per-run hash checks
    payload=dict(protocol);claimed=payload.pop('protocol_sha256')
    checks.require('B7:canonical_protocol_hash',canonical_hash(payload)==claimed,claimed)
    checks.require('B7:32_runs_eight_paired_streams',len(data)==32 and len(seeds)==8)
    manifest={(int(r['stream_seed']),r['arm']) for r in read_csv(root/'RUN_MANIFEST.csv')}
    checks.require('B7:run_manifest',set(data)==manifest)
    per_seed=[]
    for seed in seeds:
        for arm in module.ARMS:
            record=data[(seed,arm)]
            checks.require('B7:training_steps:'+str((seed,arm)),record['ledger']['optimizer_steps']==3000)
            row=dict(stream_seed=seed,arm=arm,routing_mode=record['routing_mode'],matched_uniform_floor=record['matched_uniform_floor'])
            for estimator in module.ESTIMATORS:
                evaluation=record['evaluations'][estimator];values=evaluation['bpb_values']
                checks.require('B7:evaluation_batches:'+str((seed,arm,estimator)),len(values)==96 and all(math.isfinite(v) for v in values))
                checks.require('B7:evaluation_mean:'+str((seed,arm,estimator)),abs(float(np.mean(values))-evaluation['bpb'])<1e-12)
                row['bpb_'+estimator]=evaluation['bpb']
            row['gain_structured_vs_raw']=row['bpb_raw']-row['bpb_structured']
            row['gain_scalar055_vs_raw']=row['bpb_raw']-row['bpb_scalar055']
            per_seed.append(row)
    saved=read_csv(root/'figures/paired_contrasts.csv');contrasts=[]
    for old in saved:
        if old['reference_estimator']=='raw/interaction':
            row,_=module._interaction(seeds,data,name=old['contrast'],candidate_arm=old['candidate_arm'],reference_arm=old['reference_arm'],estimator=old['candidate_estimator'])
        else:
            row,_=module._contrast(seeds,data,name=old['contrast'],candidate_arm=old['candidate_arm'],reference_arm=old['reference_arm'],candidate_estimator=old['candidate_estimator'],reference_estimator=old['reference_estimator'])
        contrasts.append(row)
    checks.matching_csv('B7:paired_contrasts',contrasts,saved,['contrast'],['n','mean','se','halfwidth95','lo95','hi95'])
    checks.matching_csv('B7:per_seed',per_seed,read_csv(root/'figures/per_seed.csv'),['stream_seed','arm'],['bpb_raw','bpb_scalar055','bpb_structured','matched_uniform_floor'])
    effects=[]; table7=[]
    for seed in seeds:
        raw=module._bpb(data,seed,'mapped','raw')-module._bpb(data,seed,'matched_uniform','raw')
        tsa=module._bpb(data,seed,'mapped','structured')-module._bpb(data,seed,'matched_uniform','structured')
        effects.append(dict(stream_seed=seed,raw_floor_effect=raw,structured_floor_effect=tsa,interaction=raw-tsa))
    for label,key in [('Raw final iterate','raw_floor_effect'),('Frozen groupwise TSA','structured_floor_effect'),('Interaction','interaction')]:
        stat=module._stats([r[key] for r in effects])
        table7.append(dict(returned_estimator=label,n=stat['n'],effect=stat['mean'],ci95_halfwidth=stat['halfwidth95'],ci95_low=stat['lo95'],ci95_high=stat['hi95'],source_basis='Recomputed from eight paired streams in 32 saved run records; manuscript contrast direction'))
    return dict(table7=table7,per_seed=per_seed,effects=effects,contrasts=contrasts,checks=checks.rows)
