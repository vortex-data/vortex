# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

scale = sys.argv[1]
binary = Path('target/release_debug/datafusion-bench')
env = dict(os.environ, TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8', RUST_LOG='warn', VORTEX_SCAN_BYTE_VIEW_DIAGNOSTICS='1')
for v2 in [False, True]:
    label = f'scan-byte-view-v2-{str(v2).lower()}'
    cmd = [str(binary), 'tpcds', '--formats', 'vortex', '--queries', '58', '--iterations', '1', '--runner', 'scan-byte-view-diagnostic', '--scan-v2', str(v2).lower(), '--hide-progress-bar', '--display-format', 'gh-json', '--show-metrics', '--ingest-jsonl', label + '.jsonl', '--opt', f'scale-factor={scale}']
    Path(label + '.command.json').write_text(json.dumps({'command':cmd, 'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(), 'diagnostic_only': True}, indent=2))
    with open(label + '.log', 'w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
    print(label + ' completed; diagnostic timing is not a performance measurement', flush=True)
