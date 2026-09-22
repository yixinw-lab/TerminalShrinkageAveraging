#!/usr/bin/env python3
"""Verify B.6/B.7 saved records without running training or changing inputs."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from groupwise_records import Checks, role_records, floor_records
ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,default=ROOT/'.checks-local')
    args=parser.parse_args();args.report_dir.mkdir(parents=True,exist_ok=True)
    checks=Checks();error=None
    try:
        role_records(ROOT,checks);floor_records(ROOT,checks)
    except Exception as exc:
        error=str(exc)
        if not checks.rows or checks.rows[-1]['status']!='FAIL':
            checks.rows.append(dict(check='exception',status='FAIL',detail=error))
    with (args.report_dir/'groupwise_record_checks.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=['check','status','detail']);w.writeheader();w.writerows(checks.rows)
    passed=sum(r['status']=='PASS' for r in checks.rows)
    text=f'''# Groupwise run-level verification

**Checks: {passed}/{len(checks.rows)} passed.**

B.6: all 8 development runs and 16 confirmation runs are present. Every development run has 290 unique candidates. Selection is replayed only on the original development files in a temporary directory; the complete frozen JSON object and hash must match the saved freeze. Paired optimizer stream permutations, selected rule specifications, source-manifest hashes and confirmation metrics are checked.

B.7: all 32 runs, the protocol and 8-by-4 manifest are present. The canonical protocol hash and run-level references, paired training permutations, and 96 evaluation-batch observations per estimator are checked. Table 7 and all 14 stored paired contrasts are recalculated; numeric parity uses absolute tolerance 1e-12 for floating-point serialization differences. These are checks of saved files, not a retraining or independent rerun of model evaluation.

The original critical-value rounding is preserved: 2.365 for B.6 and 2.364624 for B.7 (eight streams). Source summary files are not modified. No checkpoint/pickle, model training, scheduler, network or cloud allocation is used.

[Detailed checks](groupwise_record_checks.csv).
'''
    if error:text+='\n## Failure\n\n'+error+'\n'
    (args.report_dir/'GROUPWISE_RECORD_CHECKS.md').write_text(text)
    print(f'Groupwise checks: {passed}/{len(checks.rows)}')
    if error:print(error,file=sys.stderr)
    return 1 if error else 0
if __name__=='__main__':raise SystemExit(main())
