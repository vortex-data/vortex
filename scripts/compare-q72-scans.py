# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Measure Q72 separately; preserve timeouts as failures rather than dropping the query."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

scale = sys.argv[1]
binary = Path('target/release_debug/datafusion-bench')
binary.chmod(0o755)
Path('scan-binary-provenance.json').write_text(json.dumps({
    'run_id': 40475426727, 'commit': '208cd6358f3370cb5cc04a174b092699ef4bfe9f',
    'sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
    'note': 'Frozen combined planning, dense filters, early empty partitions and wide prefetch; same binary for V1 and V2.',
}, indent=2))
env = dict(os.environ, TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8', RUST_LOG='warn')
records = []
variants = [('v1', False), ('combined', True)]
if float(scale) == 30:
    variants.reverse()
for round_idx in range(3):
    for name, v2 in variants[round_idx % 2:] + variants[:round_idx % 2]:
        label = f'scan-q72-r{round_idx}-{name}'
        cmd = [str(binary), 'tpcds', '--formats', 'vortex', '--queries', '72',
               '--iterations', '1', '--scan-v2', str(v2).lower(), '--hide-progress-bar',
               '--runner', 'scan-q72-extended-8workers', '--display-format', 'gh-json',
               '--ingest-jsonl', label + '.jsonl', '--opt', f'scale-factor={scale}']
        Path(label + '.command.json').write_text(json.dumps({'command': cmd,
            'env': {k:env[k] for k in ['TOKIO_WORKER_THREADS','DATAFUSION_EXECUTION_TARGET_PARTITIONS','RUST_LOG']},
            'timeout_seconds': 900}, indent=2))
        record = dict(query=72, variant=name, scale=scale, round=round_idx)
        try:
            with open(label + '.log', 'w') as log:
                result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            record['status'] = 'ok' if result.returncode == 0 else 'failed'
            record['returncode'] = result.returncode
            if result.returncode == 0:
                values = [json.loads(line) for line in Path(label + '.jsonl').read_text().splitlines()]
                values = [v for v in values if v.get('kind') == 'query_measurement' and v['query_idx'] == 72]
                assert len(values) == 1 and len(values[0]['all_runtimes_ns']) == 1
                record['all_ns'] = values[0]['all_runtimes_ns']
                record['elapsed_ms'] = record['all_ns'][0]/1e6
        except subprocess.TimeoutExpired:
            record['status'] = 'timeout'
        records.append(record)
        Path('scan-q72-results.json').write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
    if any(r['status'] != 'ok' for r in records):
        break
if any(r['status'] != 'ok' for r in records):
    raise SystemExit('Q72 failed or timed out; see scan-q72-results.json')
