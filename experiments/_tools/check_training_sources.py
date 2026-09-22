#!/usr/bin/env python3
"""Import the local implementation modules and run three CPU self-tests; no real training.

Requires PyTorch, unlike the analysis-only reproduction command. This does not
load a model pack, import the external NanoChat model, or create a GPU job.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import csv
import importlib
import io
import json
from pathlib import Path
import platform
import sys
ROOT=Path(__file__).resolve().parents[2]
MODULES=['appendix_averaging_d12','filter_suite','groupwise_floor_d12','meta_experiment',
         'muon_aware_averaging','online_meta','paper_main_d12','skip_core','skip_predictors',
         'skip_suite','structured_tsa_d12','trajectory_groups','tsa_group_search_d12']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,default=ROOT/'.checks-local')
    args=parser.parse_args();args.report_dir.mkdir(parents=True,exist_ok=True)
    harness=ROOT/'experiments/_shared/d12_training';sys.path.insert(0,str(harness))
    checks=[];transcript=io.StringIO();versions={'python':platform.python_version()}
    def record(name,ok,detail=''):
        checks.append(dict(check=name,status='PASS' if ok else 'FAIL',detail=str(detail)))
    for path in sorted((harness/'nanochat_meta').glob('*.py')):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                module=node.module or ''
                targets=[]
                if node.level and module:targets=[module.split('.')[0]]
                elif node.level and not module:targets=[a.name for a in node.names]
                elif module.startswith('nanochat_meta.'):targets=[module.split('.')[1]]
                for target in targets:
                    record('local_dependency:'+path.name+':'+target,(harness/'nanochat_meta'/(target+'.py')).is_file())
    for name in MODULES:
        try:
            with contextlib.redirect_stdout(transcript):importlib.import_module('nanochat_meta.'+name)
            record('import:'+name,True)
        except Exception as exc:record('import:'+name,False,exc)
    for name in ['structured_tsa_d12','tsa_group_search_d12','groupwise_floor_d12']:
        try:
            with contextlib.redirect_stdout(transcript):importlib.import_module('nanochat_meta.'+name).selftest()
            record('cpu_selftest:'+name,True)
        except Exception as exc:record('cpu_selftest:'+name,False,exc)
    for name in ['torch','numpy']:
        try:versions[name]=importlib.import_module(name).__version__
        except ImportError:versions[name]='unavailable'
    with (args.report_dir/'training_source_checks.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['check','status','detail']);w.writeheader();w.writerows(checks)
    (args.report_dir/'training_source_environment.json').write_text(json.dumps(versions,indent=2)+'\n')
    passed=sum(r['status']=='PASS' for r in checks)
    text=f'''# D12 source checks

**Checks: {passed}/{len(checks)} passed.**

This checks the local import graph, imports 13 implementation modules, and runs three original CPU self-tests. It does not exercise the lazy upstream NanoChat builder or instantiate/train/evaluate the paper model. The matching upstream checkout, environment and packs are still separate setup requirements.

Test environment: Python {versions['python']}, NumPy {versions.get('numpy')}, PyTorch {versions.get('torch')}.

```text
{transcript.getvalue().strip()}
```

[Detailed checks](training_source_checks.csv). [Setup notes](../../experiments/_shared/d12_training/README.md).
'''
    if passed!=len(checks):text+='\n## Failures\n'+''.join('- '+r['check']+': '+r['detail']+'\n' for r in checks if r['status']=='FAIL')
    (args.report_dir/'TRAINING_SOURCE_CHECKS.md').write_text(text)
    print(f'Training-source checks: {passed}/{len(checks)}; no training performed')
    return 0 if passed==len(checks) else 1
if __name__=='__main__':raise SystemExit(main())
