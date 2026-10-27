# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
import hashlib
import json
import os
from pathlib import Path
import subprocess

binary = Path('target/release_debug/datafusion-bench').resolve()
env = dict(os.environ, TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8', RUST_LOG='warn,scan_split_diagnostic=debug')
provenance = {'head': os.environ['GITHUB_SHA'], 'run': os.environ['GITHUB_RUN_ID'], 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'diagnostic_only': True}
Path('scan-provenance.json').write_text(json.dumps(provenance, indent=2))
for query in ('24', '26'):
    for v2 in (False, True):
        label = f'scan-q{query}-v2-{str(v2).lower()}'
        command = [str(binary), 'clickbench', '--formats', 'vortex', '--queries', query, '--iterations', '1', '--scan-v2', str(v2).lower(), '--hide-progress-bar', '--display-format', 'gh-json', '--runner', 'filter-shapes-diagnostic', '--show-metrics', '--ingest-jsonl', label + '.jsonl']
        Path(label + '.command.json').write_text(json.dumps({'command': command, 'env': {k:env[k] for k in ('TOKIO_WORKER_THREADS', 'DATAFUSION_EXECUTION_TARGET_PARTITIONS', 'RUST_LOG')}, **provenance}, indent=2))
        with open(label + '.log', 'w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
        print(label + ' completed; diagnostic timing is not a performance measurement', flush=True)
