# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Measure Q72 separately; preserve timeouts as failures rather than dropping the query."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

scale = sys.argv[1]
binary = Path('target/release_debug/datafusion-bench')
binary.chmod(0o755)
Path('scan-binary-provenance.json').write_text(json.dumps({
    'run_id': 40432389813, 'commit': 'e09587be1c2b219bc6aabd984e47a905cbbd9bbf',
    'sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
    'note': 'Frozen coarse partition implementation; same binary for V1 and V2.',
}, indent=2))
env = dict(os.environ, TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8', RUST_LOG='warn')
records = []
variants = [('v1', False), ('coarse', True)]
if scale == '30':
    variants.reverse()
for name, v2 in variants:
    label = f'scan-q72-{name}'
    cmd = [str(binary), 'tpcds', '--formats', 'vortex', '--queries', '72',
           '--iterations', '5', '--scan-v2', str(v2).lower(), '--hide-progress-bar',
           '--runner', 'scan-q72-extended-8workers', '--display-format', 'gh-json',
           '--ingest-jsonl', label + '.jsonl', '--opt', f'scale-factor={scale}']
    Path(label + '.command.json').write_text(json.dumps({'command': cmd,
        'env': {k:env[k] for k in ['TOKIO_WORKER_THREADS','DATAFUSION_EXECUTION_TARGET_PARTITIONS','RUST_LOG']},
        'timeout_seconds': 1800}, indent=2))
    record = dict(query=72, variant=name, scale=scale)
    try:
        with open(label + '.log', 'w') as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        record['status'] = 'ok' if result.returncode == 0 else 'failed'
        record['returncode'] = result.returncode
        if result.returncode == 0:
            values = [json.loads(line) for line in Path(label + '.jsonl').read_text().splitlines()]
            values = [v for v in values if v.get('kind') == 'query_measurement' and v['query_idx'] == 72]
            assert len(values) == 1 and len(values[0]['all_runtimes_ns']) == 5
            record['all_ns'] = values[0]['all_runtimes_ns']
            record['hot_ms'] = statistics.median(record['all_ns'][2:])/1e6
    except subprocess.TimeoutExpired:
        record['status'] = 'timeout'
    records.append(record)
    Path('scan-q72-results.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
if any(r['status'] != 'ok' for r in records):
    raise SystemExit('Q72 failed or timed out; see scan-q72-results.json')
