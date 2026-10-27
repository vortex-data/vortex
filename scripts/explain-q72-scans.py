# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

binary = Path('target/release_debug/datafusion-bench')
scale = sys.argv[1]
env = dict(os.environ, TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8', RUST_LOG='warn')
Path('scan-provenance.json').write_text(json.dumps({'binary_run': 40444903733, 'production_head': 'eb29280cffbea33f1a5c5fa6b22c71efb5ad0a0b', 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'mode': 'EXPLAIN only; no Q72 execution timings'}, indent=2))
for v2 in [False, True]:
    label = f'scan-q72-explain-v2-{str(v2).lower()}'
    cmd = [str(binary), 'tpcds', '--formats', 'vortex', '--queries', '72', '--scan-v2', str(v2).lower(), '--explain', '--hide-progress-bar', '--runner', 'scan-q72-explain', '--opt', f'scale-factor={scale}']
    Path(label + '.command.json').write_text(json.dumps({'command': cmd}, indent=2))
    with open(label + '.log', 'w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
