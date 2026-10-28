# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

scale = sys.argv[1]
suite = 'tpcds' if scale and float(scale) > 0 else 'clickbench'
root = Path.cwd()
binary = root / 'target/release_debug/datafusion-bench'
env = dict(os.environ)
env.update(TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8', RUST_LOG='warn')
profile_env = {key: env[key] for key in ('TOKIO_WORKER_THREADS', 'DATAFUSION_EXECUTION_TARGET_PARTITIONS', 'RUST_LOG')}

def command(v2, query, iterations, label):
    return [str(binary), suite, '--formats', 'vortex', '--queries', query,
            '--iterations', str(iterations), '--scan-v2', str(v2).lower(),
            '--hide-progress-bar', '--display-format', 'gh-json',
            '--runner', 'isolated-coarse-profiles-8workers',
            '--ingest-jsonl', label + '.jsonl'] + (['--opt', f'scale-factor={scale}'] if suite == 'tpcds' else [])

current_binary = binary
baseline = root / 'target/coarse-baseline/datafusion-bench'
variants = [('v1', current_binary, False), ('accepted', baseline, True), ('candidate', current_binary, True)]
Path('scan-provenance.json').write_text(json.dumps({'baseline_run': 40604183269, 'candidate_run': 40607756797, 'candidate_head': '4fae876c730d08e8efbc9dfc627d06731ac7e0f8', 'runner_head': os.environ['GITHUB_SHA'], 'candidate': 'Retain local file handle and size per immutable reader; preserve concurrency; atop 256K filtered work', 'binary_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path, _ in variants}}, indent=2))
assert hashlib.sha256(current_binary.read_bytes()).hexdigest() == '6a9e108cbe6cf868ce7cdc2075f80ccb385a4c8530e8e0270de603df875cc346'
assert hashlib.sha256(baseline.read_bytes()).hexdigest() == 'aac219f95c0c3265b34fb137f5839cd2ccd7ad3d84eb16d40c8f2a31f3e1c979'
for round in range(3):
    for label_name, binary, v2 in variants[round:] + variants[:round]:
        label = f'scan-timing-r{round}-{label_name}'
        cmd = command(v2, ('20,22,28,32,41,58,99' if suite == 'tpcds' else '13,25,37'), 21, label)
        Path(label + '.command.json').write_text(json.dumps({'command': cmd, 'env': profile_env}, indent=2))
        with open(label + '.txt', 'w') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
for label_name, binary, v2 in variants:
    label = f'scan-metrics-{label_name}'
    cmd = command(v2, ('20,22,28,32,41,58,99' if suite == 'tpcds' else '13,25,37'), 1, label) + ['--show-metrics']
    with open(label + '.txt', 'w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
