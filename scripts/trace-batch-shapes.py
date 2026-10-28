# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Diagnostic only: capture ordered scan batches, not performance timings."""
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

scale = sys.argv[1]
binary = Path('target/release_debug/datafusion-bench').resolve()
env = dict(os.environ, TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8', DATAFUSION_EXECUTION_BATCH_SIZE='8192', RUST_LOG='warn', VORTEX_TRACE_SCAN_BATCHES='1')
keys = ['TOKIO_WORKER_THREADS', 'DATAFUSION_EXECUTION_TARGET_PARTITIONS', 'DATAFUSION_EXECUTION_BATCH_SIZE', 'RUST_LOG', 'VORTEX_TRACE_SCAN_BATCHES']
provenance = dict(run=os.environ.get('GITHUB_RUN_ID'), head=os.environ.get('GITHUB_SHA'), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), scale=scale, query=58, diagnostic_only=True, env={k:env[k] for k in keys})
Path('scan-provenance.json').write_text(json.dumps(provenance, indent=2))
pattern = re.compile(r'SCAN_BATCH v2=(true|false) stream=(\d+) index=(\d+) rows=(\d+) file=(".*") columns=(\[.*\])')
for v2 in [False, True]:
    label = 'scan-shapes-' + ('v2' if v2 else 'v1')
    cmd = [str(binary), 'tpcds', '--formats', 'vortex', '--queries', '58', '--iterations', '1', '--scan-v2', str(v2).lower(), '--hide-progress-bar', '--show-metrics', '--runner', 'scan-batch-shape-diagnostic', '--ingest-jsonl', label+'.jsonl', '--opt', f'scale-factor={scale}']
    Path(label+'.command.json').write_text(json.dumps(dict(command=cmd, env=provenance['env']), indent=2))
    with open(label+'.txt', 'w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    streams = {}
    for line in Path(label+'.txt').read_text().splitlines():
        if 'SCAN_BATCH ' not in line:
            continue
        match = pattern.search(line)
        assert match, line
        version, sid, index, rows, file, columns = match.groups()
        assert version == str(v2).lower()
        entry = streams.setdefault(sid, dict(file=json.loads(file), columns=json.loads(columns), rows=[]))
        assert entry['file'] == json.loads(file) and entry['columns'] == json.loads(columns)
        assert int(index) == len(entry['rows'])+1
        entry['rows'].append(int(rows))
    assert streams, 'No diagnostic events captured'
    for entry in streams.values():
        rows = entry['rows']
        slices = [n for size in rows for n in ([8192]*(size//8192)+([size%8192] if size%8192 else []))]
        entry.update(total_rows=sum(rows), batches=len(rows), last_batch=rows[-1], inferred_split_batches=len(slices), inferred_small_slices=sum(n<=4096 for n in slices), inferred_final_slice=slices[-1] if slices else 0)
    summary = dict(streams=streams, stream_count=len(streams), total_rows=sum(e['total_rows'] for e in streams.values()), small_final_slices=sum(0<e['inferred_final_slice']<=4096 for e in streams.values()), note='Split slices inferred for configured 8192-row DataSourceExec batches; actual events are before splitting. Timings with logging are diagnostic only.')
    Path(label+'-summary.json').write_text(json.dumps(summary, indent=2))
    print(label, {k:v for k,v in summary.items() if k!='streams'}, flush=True)
