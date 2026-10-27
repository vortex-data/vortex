# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import urllib.request

scale = sys.argv[1]
root = Path.cwd()
binary = root / 'target/release_debug/datafusion-bench'
archive_name = 'samply-aarch64-unknown-linux-gnu.tar.xz'
release = 'https://github.com/mstange/samply/releases/download/samply-v0.13.1/'
archive = root / archive_name
archive.write_bytes(urllib.request.urlopen(release + archive_name).read())
expected = urllib.request.urlopen(release + archive_name + '.sha256').read().decode().split()[0]
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
with tarfile.open(archive) as bundle:
    bundle.extractall(root / 'samply-tool', filter='data')
executables = list((root / 'samply-tool').rglob('samply'))
assert len(executables) == 1
samply = executables[0].resolve()
samply.chmod(0o755)
assert shutil.which('llvm-symbolizer')

env = dict(os.environ)
env.update(TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8', RUST_LOG='warn')
profile_env = {key: env[key] for key in ('TOKIO_WORKER_THREADS', 'DATAFUSION_EXECUTION_TARGET_PARTITIONS', 'RUST_LOG')}

def command(v2, query, iterations, label):
    return [str(binary), 'tpcds', '--formats', 'vortex', '--queries', query,
            '--iterations', str(iterations), '--scan-v2', str(v2).lower(),
            '--hide-progress-bar', '--display-format', 'gh-json',
            '--runner', 'isolated-coarse-profiles-8workers',
            '--ingest-jsonl', label + '.jsonl', '--opt', f'scale-factor={scale}']

variants = [('v1', binary, False), ('accepted', binary, True)]

spec = importlib.util.spec_from_file_location('summary', root / '.agents/skills/samply/scripts/profile_summary.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
original_frame_label = summary.frame_label

def frame_label(thread, index, symbols=None):
    function = summary.value_at(thread.get('frameTable') or {}, 'func', index)
    resource_index = summary.value_at(thread.get('funcTable') or {}, 'resource', function) if isinstance(function, int) else None
    resource = summary.resource_name(thread, resource_index)
    return original_frame_label(thread, index, symbols if os.path.basename(resource) == binary.name else None)

summary.frame_label = frame_label
binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
for query, iterations in [('72', 1)]:
    for label_name, binary, v2 in variants:
        binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
        label = f'scan-q{query}-{label_name}'
        output = root / (label + '.profile.json.gz')
        with open(label + '.log', 'w') as log:
            benchmark = subprocess.Popen(command(v2, query, iterations, label), env=env,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            cmd = ['sudo', '-n', str(samply), 'record', '--save-only', '--rate', '1000',
                   '--pid', str(benchmark.pid), '--output', str(output)]
            metadata = {'command': command(v2, query, iterations, label), 'profile_command': cmd,
                        'env': profile_env, 'binary_sha256': binary_sha,
                        'binary_run': 40444903733,
                        'production_head': 'eb29280cffbea33f1a5c5fa6b22c71efb5ad0a0b',
                        'mode': 'bounded CPU profile; not a completed query timing'}
            profiler = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                try:
                    benchmark.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
            finally:
                metadata['query_finished_before_stop'] = benchmark.poll() is not None
                if benchmark.poll() is None:
                    os.killpg(benchmark.pid, signal.SIGTERM)
                    try:
                        benchmark.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(benchmark.pid, signal.SIGKILL)
                        benchmark.wait(timeout=20)
                metadata['query_returncode'] = benchmark.returncode
                metadata['capture_window_seconds'] = 60
                try:
                    metadata['profiler_returncode'] = profiler.wait(timeout=120)
                finally:
                    Path(label + '.command.json').write_text(json.dumps(metadata, indent=2))
                if profiler.returncode != 0:
                    raise RuntimeError(f'Profiler exited with {profiler.returncode}')
        profile = summary.load_profile(output)
        addresses = summary.collect_symbol_addresses(profile, binary.name)
        result = subprocess.run(['llvm-symbolizer', '--obj=' + str(binary), '--no-inlines', '--demangle'],
                                input='\n'.join(addresses) + '\n', text=True, capture_output=True, check=True)
        blocks = result.stdout.strip().split('\n\n')
        assert len(blocks) == len(addresses)
        symbols = {address: block.splitlines()[0] for address, block in zip(addresses, blocks)
                   if block.splitlines() and block.splitlines()[0] != '??'}
        Path(label + '.symbols.json').write_text(json.dumps(symbols))
        threads = sorted((t for t in profile['threads'] if t['name'] == 'tokio-rt-worker'),
                         key=summary.thread_cpu_ms, reverse=True)[:8]
        assert threads, 'No worker threads captured'
        own, inclusive, stacks = collections.Counter(), collections.Counter(), collections.Counter()
        cpu_ms = 0
        for thread in threads:
            counts = summary.summarize_thread(thread, symbols, 'cpu')
            cpu_ms += counts['cpu_ms']
            own.update(counts['self'])
            inclusive.update(counts['inclusive'])
            stacks.update(counts['stacks'])
        counters = {'cpu_ms': cpu_ms, 'self': own, 'inclusive': inclusive,
                    'all_threads': [{'name': t['name'], 'tid': t['tid'], 'cpu_ms': summary.thread_cpu_ms(t)} for t in profile['threads']],
                    'top_stacks': [{'weight': weight, 'frames': frames} for frames, weight in stacks.most_common(30)]}
        Path(label + '.counters.json').write_text(json.dumps(counters, indent=2))
        print(json.dumps({'profile': str(output), 'worker_cpu_ms': cpu_ms,
                          'symbols_resolved': len(symbols), 'symbol_addresses': len(addresses),
                          'top_self': own.most_common(8)}), flush=True)
