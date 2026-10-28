# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

suite, scale = sys.argv[1:3]
current = Path('target/release_debug/datafusion-bench')
variants = [('v1', current, False), ('combined', current, True)]
env = dict(os.environ, TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8', RUST_LOG='warn')
records = []
Path('scan-provenance.json').write_text(json.dumps({'binary_run': 40613767503, 'head': 'd88dd980c3b0153ce02faba6156ec584f3bd711a', 'scope': 'same-binary V1/V2 all queries; timeouts retained, not parity'}, indent=2))
metadata = {name: hashlib.sha256(binary.read_bytes()).hexdigest() for name, binary, _ in variants}
Path('scan-binaries.json').write_text(json.dumps(metadata, indent=2))
queries = range(43) if suite == 'clickbench' else range(1, 100)
guards = [0, 1, 4, 6, 7, 9, 10, 11, 13, 15, 16, 17, 22, 24, 25, 26, 28, 31, 32, 33, 36, 37, 38, 40, 41, 42] if suite == 'clickbench' else [1, 6, 10, 13, 15, 16, 18, 20, 21, 22, 28, 32, 35, 37, 38, 39, 40, 41, 42, 45, 47, 48, 49, 51, 55, 57, 58, 59, 62, 65, 66, 67, 76, 82, 86, 87, 91, 93, 94, 95, 97, 98, 99]
for stage, rounds, iterations, selected in [('all', 1, 5, queries), ('guards', 3, 9, guards)]:
    for rnd in range(rounds):
        for q in selected:
            shift = (q + rnd) % len(variants)
            for name, binary, v2 in variants[shift:] + variants[:shift]:
                label = f'scan-{stage}-r{rnd}-q{q}-{name}'
                query_iterations = 1 if suite == 'tpcds' and q == 72 else iterations
                cmd = [str(binary), suite, '--formats', 'vortex', '--queries', str(q), '--iterations', str(query_iterations), '--scan-v2', str(v2).lower(), '--hide-progress-bar', '--runner', 'scan-filehandle-all-8workers', '--display-format', 'gh-json', '--ingest-jsonl', label + '.jsonl']
                if suite == 'tpcds':
                    cmd += ['--opt', f'scale-factor={scale}']
                Path(label + '.command.json').write_text(json.dumps({'command': cmd, 'env': {k: env[k] for k in ['TOKIO_WORKER_THREADS', 'DATAFUSION_EXECUTION_TARGET_PARTITIONS', 'RUST_LOG']}}, indent=2))
                record = dict(stage=stage, round=rnd, query=q, variant=name, iterations=query_iterations, warmup_discard=2 if query_iterations > 2 else 0)
                try:
                    with open(label + '.txt', 'w') as log:
                        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900 if suite == 'tpcds' and q == 72 else 180)
                    record['status'] = 'ok' if result.returncode == 0 else 'failed'
                    record['returncode'] = result.returncode
                    if result.returncode == 0:
                        measurements = [json.loads(line) for line in Path(label + '.jsonl').read_text().splitlines()]
                        values = [m for m in measurements if m.get('kind') == 'query_measurement' and m['query_idx'] == q]
                        assert len(values) == 1 and len(values[0]['all_runtimes_ns']) == query_iterations
                        record['all_ns'] = values[0]['all_runtimes_ns']
                        record['hot_ms'] = statistics.median(record['all_ns'][2:] if query_iterations > 2 else record['all_ns']) / 1e6
                except subprocess.TimeoutExpired:
                    record['status'] = 'timeout'
                records.append(record)
                Path('scan-results.json').write_text(json.dumps(records, indent=2))
                print(json.dumps(record), flush=True)

if any(record['status'] != 'ok' for record in records):
    raise SystemExit('Some queries failed or timed out; see scan-results.json')

for name, binary, v2 in variants:
    label = f'scan-q32-metrics-{name}'
    if suite != 'tpcds':
        continue
    cmd = [str(binary), suite, '--formats', 'vortex', '--queries', '32', '--iterations', '1', '--scan-v2', str(v2).lower(), '--hide-progress-bar', '--runner', 'scan-current-head-diagnostics', '--show-metrics', '--opt', f'scale-factor={scale}']
    with open(label + '.txt', 'w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
    print(Path(label + '.txt').read_text(), flush=True)
