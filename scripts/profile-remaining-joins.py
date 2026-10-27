# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path.cwd()
binary = root / 'target/release_debug/datafusion-bench'
env = dict(os.environ)
env.update(TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8', RUST_LOG='warn')
profile_env = {key: env[key] for key in ('TOKIO_WORKER_THREADS', 'DATAFUSION_EXECUTION_TARGET_PARTITIONS', 'RUST_LOG')}

def command(v2, query, iterations, label):
    return [str(binary), 'clickbench', '--formats', 'vortex', '--queries', query,
            '--iterations', str(iterations), '--scan-v2', str(v2).lower(),
            '--hide-progress-bar', '--display-format', 'gh-json',
            '--runner', 'isolated-coarse-profiles-8workers',
            '--ingest-jsonl', label + '.jsonl']

current_binary = binary
candidate = root / 'target/planning-dense-baseline/datafusion-bench'
candidate.chmod(0o755)
variants = [('v1', candidate, False), ('accepted', current_binary, True), ('candidate', candidate, True)]
Path('scan-provenance.json').write_text(json.dumps({'binary_run': 40461998987, 'production_head': '65c7c2a0e5b2abde9762582e14bb843d9cc7f019', 'candidate_binary_run': 40475426727, 'candidate_source': '208cd6358f3370cb5cc04a174b092699ef4bfe9f', 'binary_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path, _ in variants}, 'scope': 'Latest dense infallible ClickBench remaining V1 gaps and unguarded queries; partial query set'}, indent=2))
for round_idx in range(6):
    for label_name, binary, v2 in variants[round_idx % 3:] + variants[:round_idx % 3]:
        label = f'scan-timing-r{round_idx}-{label_name}'
        cmd = command(v2, '0,12,13,14,22,26,28', 9, label)
        Path(label + '.command.json').write_text(json.dumps({'command': cmd, 'env': profile_env}, indent=2))
        with open(label + '.log', 'w') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200, check=True)
for label_name, binary, v2 in variants:
    label = f'scan-metrics-{label_name}'
    cmd = command(v2, '12,13,14,22', 1, label) + ['--show-metrics']
    with open(label + '.log', 'w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200, check=True)
binary = current_binary

