#!/usr/bin/env python3
"""Run CPU-only repository, syntax, reconstruction, and manuscript-number checks.

This audit uses saved analysis inputs. It does not train models, load model
weights, submit jobs, or independently recompute CORE from checkpoints.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import json
import math
import statistics
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote

from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def equal(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return str(a) == str(b)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=ROOT/'.checks-local')
    parser.add_argument('--skip-rebuild', action='store_true')
    args = parser.parse_args()
    reports = args.report_dir
    reports.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, str]] = []

    def check(name: str, passed: bool, detail, basis: str = 'integrity') -> None:
        checks.append(dict(check=name, status='PASS' if passed else 'FAIL',
                           basis=basis, detail=str(detail)))

    dirs = sorted(p.name for p in ROOT.iterdir()
                  if p.is_dir() and not p.name.startswith('.'))
    check('top_level_content_folders', dirs == ['experiments', 'figures', 'results'], dirs)

    for path in ROOT.rglob('*.py'):
        try:
            ast.parse(path.read_text())
            check('python_syntax:' + str(path.relative_to(ROOT)), True, 'AST parse')
        except (SyntaxError, UnicodeError) as exc:
            check('python_syntax:' + str(path.relative_to(ROOT)), False, exc)

    bash = shutil.which('bash')
    if bash:
        for path in sorted(ROOT.rglob('*')):
            if path.suffix in {'.sh', '.sbatch'}:
                proc = subprocess.run([bash, '-n', str(path)], capture_output=True,
                                      text=True, timeout=10)
                check('shell_syntax:' + str(path.relative_to(ROOT)),
                      proc.returncode == 0, proc.stderr.strip() or 'bash -n')

    readmes = ([ROOT/'README.md', ROOT/'experiments/README.md', ROOT/'results/README.md',
                ROOT/'figures/README.md']
               + list(ROOT.glob('experiments/sec_*/README.md'))
               + list(ROOT.glob('experiments/app_*/README.md'))
               + list(ROOT.glob('results/sec_*/README.md'))
               + list(ROOT.glob('results/app_*/README.md'))
               + list(ROOT.glob('figures/*/README.md')))
    for path in readmes:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if re.match(r'^[a-z]+:', target) or target.startswith('#'):
                continue
            target = unquote(target.split('#')[0])
            if not target:
                continue
            dest = path.parent/target
            check('navigation:' + str(path.relative_to(ROOT)) + ':' + target,
                  dest.exists(), str(dest))

    expected = json.loads((ROOT/'experiments/_tools/manuscript_expectations.json').read_text())['checks']
    numeric: list[dict] = []
    for item in expected:
        rows = read_csv(ROOT/item['path'])
        matching = [row for row in rows
                    if all(equal(row.get(key), value) for key, value in item['match'].items())]
        digits = item['decimal_places']
        actual = float(matching[0][item['field']]) if len(matching) == 1 else float('nan')
        ok = (len(matching) == 1
              and f'{actual:.{digits}f}' == f'{item["expected"]:.{digits}f}')
        numeric.append(dict(
            paper_item=item['paper_item'], path=item['path'],
            row=json.dumps(item['match'], sort_keys=True), field=item['field'],
            actual=actual, expected=item['expected'], decimal_places=digits,
            status='PASS' if ok else 'FAIL', basis=item['basis']))
    matched = sum(row['status'] == 'PASS' for row in numeric)
    check('manuscript_numbers', matched == len(numeric),
          f'{matched}/{len(numeric)} matched at manuscript precision', 'numerical checks')

    d12 = ROOT/'results/_sources/d12_replicated'
    for role, count in [('confirmation', 30), ('development', 10)]:
        paths = list((d12/role).glob('*/*/*/result.json'))
        check(role + '_run_count', len(paths) == count, len(paths), 'record coverage')
        for path in paths:
            record = json.loads(path.read_text())
            ok = (record['train_steps'] == 3000 and record['seed'] == 1337
                  and record['checkpoint_k'] == 8 and record['checkpoint_spacing'] == 32)
            check('protocol:' + str(path.relative_to(d12)), ok,
                  'T=3000; initialization seed 1337; K=8; spacing=32',
                  'record fingerprint')

    d22 = ROOT/'results/_sources/d22_runs'
    control_rows = read_csv(d22/'existing_controls_n3.csv')
    control_keys = sorted((row['configuration'], int(row['seed'])) for row in control_rows)
    expected_control_keys = sorted((cfg, seed) for cfg in
                                   ['pr830_early_stop_9p4_raw', 'pr830_direct_9p0_raw']
                                   for seed in [42, 43, 44])
    check('d22_control_records', control_keys == expected_control_keys,
          f'{len(control_rows)} rows', 'record coverage')
    treatment_paths = sorted((d22/'d22_alpha70/nanochat-causal-speedrun-v2/results').glob(
        'n3_floor15_direct9_tsa0700_s*/tsa_core.json'))
    treatment_seeds = sorted(int(path.parent.name.rsplit('_s', 1)[1]) for path in treatment_paths)
    check('d22_treatment_records', treatment_seeds == [42, 43, 44],
          treatment_seeds, 'record coverage')

    per_seed = read_csv(ROOT/'results/sec_4_5_d22_scale_transfer/table_02_sec_4_5_time_bpb_and_control_core_per_seed.csv')
    expected_keys = expected_control_keys + [
        ('floor15_direct9_tsa0700', seed) for seed in [42, 43, 44]]
    check('d22_export_seed_coverage',
          sorted((row['configuration'], int(row['seed'])) for row in per_seed)
          == sorted(expected_keys),
          'One exported observation per configuration and seed', 'record coverage')

    source_rows = list(control_rows)
    for path in treatment_paths:
        seed = int(path.parent.name.rsplit('_s', 1)[1])
        record = json.loads(path.read_text())
        config = dict(line.split('=', 1) for line in
                      (path.parent/'run_config.txt').read_text().splitlines() if '=' in line)
        digest = dict(line.split('=', 1) for line in
                      (path.parent/'DIGEST.txt').read_text().splitlines() if '=' in line)
        tag = f'n3_floor15_direct9_tsa0700_s{seed}_r9_blend_0.70'
        check(f'd22_treatment_identity_seed{seed}',
              int(config['experiment_seed']) == seed
              and record['model_tag'] == config['tsa_model_tag'] == tag
              and record['recipe'] == 'blend:0.70'
              and equal(record['ratio'], 9.0)
              and int(record['step']) == int(config['final_step']) == 10172
              and equal(config['tsa_alpha'], 0.70)
              and equal(config['terminal_clamp'], 0.15),
              'Seed, model tag, recipe, ratio, step, alpha and floor agree',
              'record fingerprint')
        check(f'd22_treatment_digest_core_seed{seed}',
              equal(record['core_metric'], digest['tsa_core']),
              'JSON CORE agrees with the run digest', 'run-level records')
        source_rows.append(dict(seed=seed, configuration='floor15_direct9_tsa0700',
                                time_min=record['training_minutes'],
                                val_bpb=record['val_bpb'], core=record['core_metric']))

    for record in source_rows:
        config, seed = record['configuration'], int(record['seed'])
        matching = [row for row in per_seed
                    if row['configuration'] == config and int(row['seed']) == seed]
        for metric in ['time_min', 'val_bpb', 'core']:
            check(f'd22_export:{config}:seed{seed}:{metric}',
                  len(matching) == 1 and equal(matching[0][metric], record[metric]),
                  'Export agrees with its saved source observation', 'run-level records')
        check(f'd22_export_basis:{config}:seed{seed}',
              len(matching) == 1 and matching[0]['core_status'] == 'saved_run_record',
              'CORE is an observed per-seed score', 'run-level records')

    d22_summary = read_csv(ROOT/'results/sec_4_5_d22_scale_transfer/table_02_sec_4_5_d22_scale_transfer.csv')
    configs = sorted({row['configuration'] for row in source_rows})
    check('d22_summary_coverage',
          sorted(row['configuration_id'] for row in d22_summary) == configs,
          'One summary per configuration', 'record coverage')
    for config in configs:
        matching = [row for row in d22_summary if row['configuration_id'] == config]
        check(f'd22_summary_basis:{config}',
              len(matching) == 1 and matching[0]['core_basis'] == 'recomputed_from_saved_records',
              'Aggregate CORE is computed from the saved seed scores', 'run-level records')
        for metric in ['time_min', 'val_bpb', 'core']:
            values = [float(row[metric]) for row in source_rows if row['configuration'] == config]
            n = len(values)
            expected_mean = statistics.mean(values)
            expected_halfwidth = float(t.ppf(.975, n - 1)) * statistics.stdev(values) / math.sqrt(n)
            check(f'd22_summary:{config}:{metric}',
                  len(matching) == 1 and int(matching[0]['n']) == n
                  and equal(matching[0][metric + '_mean'], expected_mean)
                  and equal(matching[0][metric + '_ci95_halfwidth'], expected_halfwidth),
                  'Mean and 95% Student-t interval agree with the saved observations',
                  'independent aggregation')

    group_proc = subprocess.run(
        [sys.executable, str(ROOT/'experiments/_tools/verify_groupwise_records.py'),
         '--report-dir', str(reports)], capture_output=True, text=True, timeout=90)
    check('B6_B7_run_level_verification', group_proc.returncode == 0,
          group_proc.stdout.strip() or group_proc.stderr.strip(), 'run-level records')

    if not args.skip_rebuild:
        with tempfile.TemporaryDirectory(prefix='tsa-cpu-rebuild-') as temp_dir:
            proc = subprocess.run(
                [sys.executable, str(ROOT/'experiments/_tools/reproduce.py'), '--no-plots',
                 '--output-root', temp_dir], capture_output=True, text=True, timeout=90)
            check('CPU_reconstruction', proc.returncode == 0,
                  proc.stderr.strip() or 'completed', 'reconstruction')
            if proc.returncode == 0:
                for path in Path(temp_dir).rglob('*'):
                    if path.is_file():
                        current = ROOT/path.relative_to(temp_dir)
                        check('rebuild:' + str(path.relative_to(temp_dir)),
                              current.is_file() and path.read_bytes() == current.read_bytes(),
                              'exact byte parity for generated numerical exports',
                              'reconstruction')

    write_csv(reports/'numeric_checks.csv', numeric)
    write_csv(reports/'repository_checks.csv', checks)
    failed = [row for row in checks if row['status'] == 'FAIL']
    numeric_failed = [row for row in numeric if row['status'] == 'FAIL']
    basis_counts = Counter(row['basis'] for row in numeric)

    lines = [
        '# Repository audit', '',
        f'**Repository checks:** {len(checks)-len(failed)}/{len(checks)} passed.',
        f'**Manuscript numerical checks:** {len(numeric)-len(numeric_failed)}/{len(numeric)} matched at manuscript precision.',
        '',
        'Numerical exports are checked against the saved records and the included manuscript expectations. These checks do not independently evaluate model checkpoints.',
        '', '| Numerical-check basis | Fields |', '|---|---|',
    ] + [f'| {key} | {value} |' for key, value in sorted(basis_counts.items())]
    lines += [
        '',
        'See [numeric_checks.csv](numeric_checks.csv), [repository_checks.csv](repository_checks.csv), and [GROUPWISE_RECORD_CHECKS.md](GROUPWISE_RECORD_CHECKS.md).',
        '',
        'These checks are CPU-only. They do not rerun model training or evaluate model checkpoints.',
    ]
    if failed:
        lines += ['', '## Failures'] + [f'- {row["check"]}: {row["detail"]}' for row in failed]
    (reports/'AUDIT.md').write_text('\n'.join(lines) + '\n')

    print(f'Checks: {len(checks)-len(failed)}/{len(checks)}; '
          f'numeric fields: {len(numeric)-len(numeric_failed)}/{len(numeric)}; reports: {reports}')
    for row in numeric_failed:
        print('NUMBER MISMATCH:', row)
    for row in failed:
        print('FAIL:', row['check'], row['detail'])
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
