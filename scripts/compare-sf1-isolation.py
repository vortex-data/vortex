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
variants = [('v1', current, False), ('accepted', current, True),
            ('cached', Path('target/cached-baseline/datafusion-bench'), True),
            ('casts', Path('target/casts-baseline/datafusion-bench'), True),
            ('combined', Path('target/combined-baseline/datafusion-bench'), True)]
for _, binary, _ in variants:
    binary.chmod(0o755)
env = dict(os.environ, TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8', RUST_LOG='warn')
records = []
Path('scan-provenance.json').write_text(json.dumps({'runs': {'v1': 40444903733, 'accepted': 40444903733, 'cached': 40448514623, 'casts': 40446795204, 'combined': 40450616646}, 'scope': 'SF1 focused repeated regressions; not all-query coverage'}, indent=2))
metadata = {name: hashlib.sha256(binary.read_bytes()).hexdigest() for name, binary, _ in variants}
Path('scan-binaries.json').write_text(json.dumps(metadata, indent=2))
guards = [8, 11, 13, 16, 19, 20, 21, 22, 28, 35, 41, 49, 54, 55, 65, 70, 76, 92, 93]
for stage, rounds, iterations, selected in [('guards', 5, 9, guards)]:
    for rnd in range(rounds):
        for q in selected:
            shift = (q + rnd) % len(variants)
            for name, binary, v2 in variants[shift:] + variants[:shift]:
                label = f'scan-{stage}-r{rnd}-q{q}-{name}'
                cmd = [str(binary), suite, '--formats', 'vortex', '--queries', str(q), '--iterations', str(iterations), '--scan-v2', str(v2).lower(), '--hide-progress-bar', '--runner', 'scan-sf1-isolation-8workers', '--display-format', 'gh-json', '--ingest-jsonl', label + '.jsonl']
                if suite == 'tpcds':
                    cmd += ['--opt', f'scale-factor={scale}']
                Path(label + '.command.json').write_text(json.dumps({'command': cmd, 'env': {k: env[k] for k in ['TOKIO_WORKER_THREADS', 'DATAFUSION_EXECUTION_TARGET_PARTITIONS', 'RUST_LOG']}}, indent=2))
                record = dict(stage=stage, round=rnd, query=q, variant=name)
                try:
                    with open(label + '.log', 'w') as log:
                        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                    record['status'] = 'ok' if result.returncode == 0 else 'failed'
                    record['returncode'] = result.returncode
                    if result.returncode == 0:
                        measurements = [json.loads(line) for line in Path(label + '.jsonl').read_text().splitlines()]
                        values = [m for m in measurements if m.get('kind') == 'query_measurement' and m['query_idx'] == q]
                        assert len(values) == 1 and len(values[0]['all_runtimes_ns']) == iterations
                        record['all_ns'] = values[0]['all_runtimes_ns']
                        record['hot_ms'] = statistics.median(record['all_ns'][2:]) / 1e6
                except subprocess.TimeoutExpired:
                    record['status'] = 'timeout'
                records.append(record)
                Path('scan-results.json').write_text(json.dumps(records, indent=2))
                print(json.dumps(record), flush=True)

if any(record['status'] != 'ok' for record in records):
    raise SystemExit('Some queries failed or timed out; see scan-results.json')
