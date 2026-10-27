# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
import json
import os
from pathlib import Path
import subprocess
import sys

scale = sys.argv[1]
env = dict(os.environ, TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8', RUST_LOG='warn')
def run(v2, label, iterations, diagnose=False):
    cmd = ['target/release_debug/datafusion-bench', 'tpcds', '--formats', 'vortex', '--queries', '58', '--iterations', str(iterations), '--scan-v2', str(v2).lower(), '--hide-progress-bar', '--display-format', 'gh-json', '--runner', 'scan-batch-diagnostics-8workers', '--opt', 'scale-factor=' + scale, '--ingest-jsonl', label + '.jsonl']
    run_env = dict(env)
    if diagnose:
        cmd.append('--show-metrics')
        run_env['RUST_LOG'] = 'warn,scan_batch_diagnostics=info'
    Path(label + '.command.json').write_text(json.dumps({'command':cmd,'env':{k:run_env[k] for k in ['TOKIO_WORKER_THREADS','DATAFUSION_EXECUTION_TARGET_PARTITIONS','RUST_LOG']}},indent=2))
    with open(label + '.log','w') as log:
        subprocess.run(cmd,env=run_env,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
for round in range(3):
    for v2 in ([False,True] if round%2==0 else [True,False]):
        run(v2,f'scan-timing-r{round}-v2-{str(v2).lower()}',9)
for v2 in [False,True]:
    run(v2,f'scan-batches-v2-{str(v2).lower()}',1,True)
