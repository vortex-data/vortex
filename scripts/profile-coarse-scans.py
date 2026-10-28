# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

scale = sys.argv[1]
assert float(scale) in [10, 30]
root = Path.cwd()
baseline = root / 'target/coarse-baseline/datafusion-bench'
aligned = root / 'target/release_debug/datafusion-bench'
base_hash = '9d17dbcae5ac168ee496c87135a977771e356dfe1dce8589363f9ea03e00e74a'
aligned_hash = '20086be38ecc47bdbcbd22613b007d54f884e97fbf78923d1c5f5c70856a9bf7'
assert hashlib.sha256(baseline.read_bytes()).hexdigest() == base_hash
assert hashlib.sha256(aligned.read_bytes()).hexdigest() == aligned_hash
variants = [
    ('v1', baseline, False, False),
    ('v2', baseline, True, False),
    ('v1-null', baseline, False, True),
    ('v2-null', baseline, True, True),
    ('aligned-null', aligned, True, True),
]
Path('scan-provenance.json').write_text(json.dumps(dict(
    baseline_run=40613767503, baseline_head='d88dd980c3b0153ce02faba6156ec584f3bd711a',
    aligned_run=40620380829, aligned_head='c22f6a62728a0cffe9697dec505e61764723a08a',
    runner_head=os.environ['GITHUB_SHA'], runner_run=os.environ['GITHUB_RUN_ID'],
    binary_sha256={name: aligned_hash if path == aligned else base_hash for name,path,_,_ in variants}
), indent=2))

def run(name, binary, v2, nulls, label, queries, iterations, extra=()):
    env = dict(os.environ, TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8',
               RUST_LOG='warn', DATAFUSION_OPTIMIZER_FILTER_NULL_JOIN_KEYS=str(nulls).lower())
    cmd = [str(binary), 'tpcds', '--formats', 'vortex', '--queries', queries,
           '--iterations', str(iterations), '--scan-v2', str(v2).lower(),
           '--hide-progress-bar', '--display-format', 'gh-json', '--runner', 'null-join-keys-8workers',
           '--ingest-jsonl', label+'.jsonl', '--opt', f'scale-factor={scale}', *extra]
    Path(label+'.command.json').write_text(json.dumps(dict(command=cmd, env={k:env[k] for k in (
        'TOKIO_WORKER_THREADS','DATAFUSION_EXECUTION_TARGET_PARTITIONS','RUST_LOG','DATAFUSION_OPTIMIZER_FILTER_NULL_JOIN_KEYS')}), indent=2))
    with Path(label+'.txt').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)

for rnd in range(3):
    for variant in variants[rnd:] + variants[:rnd]:
        run(*variant, f'scan-timing-r{rnd}-{variant[0]}', '20,28,32,58', 9)
for variant in variants:
    run(*variant, f'scan-metrics-{variant[0]}', '20,28,32,58', 1, ['--show-metrics'])
    run(*variant, f'scan-explain-{variant[0]}', '58', 1, ['--explain'])
