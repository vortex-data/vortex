# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

scale = int(float(sys.argv[1]))
cpus = sorted(os.sched_getaffinity(0))
assert len(cpus) == 8 and set(cpus).isdisjoint({0, 1}), cpus
candidate = Path('target/release_debug/datafusion-bench').resolve()
baseline = Path('target/coarse-baseline/datafusion-bench').resolve()
expected_baseline = 'a1dbc572cbec9b9eaa0d8f461cdc9f9e0d15d976de0b0e5b77ecc2380476a388'
assert hashlib.sha256(baseline.read_bytes()).hexdigest() == expected_baseline
variants = [('v1', baseline, False), ('accepted', baseline, True), ('candidate', candidate, True)]
env = dict(os.environ, TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8', RUST_LOG='warn')
record_env = {key: env[key] for key in ['TOKIO_WORKER_THREADS', 'DATAFUSION_EXECUTION_TARGET_PARTITIONS', 'RUST_LOG']}
Path('scan-provenance.json').write_text(json.dumps(dict(head=os.environ['GITHUB_SHA'],run=os.environ['GITHUB_RUN_ID'],scale=scale,cpus=cpus,baseline_run=40627772121,binaries={name:hashlib.sha256(binary.read_bytes()).hexdigest() for name,binary,_ in variants}),indent=2))
queries = [3,13,16,20,22,28,30,32,41,42,48,58,62,76,99]
records = []
for rnd in range(3):
    for q in queries:
        for name,binary,v2 in variants[rnd:] + variants[:rnd]:
            label = f'scan-r{rnd}-q{q}-{name}'
            cmd = [str(binary),'tpcds','--formats','vortex','--queries',str(q),'--iterations','9','--scan-v2',str(v2).lower(),'--runner','isolated-tail-lookahead-8cores','--hide-progress-bar','--display-format','gh-json','--ingest-jsonl',label+'.jsonl','--opt',f'scale-factor={scale}']
            Path(label+'.command.json').write_text(json.dumps(dict(command=cmd,env=record_env),indent=2))
            with open(label+'.txt','w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
            rows = [r for r in map(json.loads,Path(label+'.jsonl').read_text().splitlines()) if r.get('kind') == 'query_measurement']
            assert len(rows) == 1 and rows[0]['query_idx'] == q
            values = rows[0]['all_runtimes_ns']
            assert len(values) == 9
            record = dict(round=rnd,query=q,variant=name,ms=statistics.median(values[2:])/1e6,all_ns=values)
            records.append(record)
            Path('scan-results.json').write_text(json.dumps(records,indent=2))
            print(json.dumps(record),flush=True)
for name,binary,v2 in variants:
    label = f'scan-metrics-{name}'
    cmd = [str(binary),'tpcds','--formats','vortex','--queries','28,30,32,42,58,99','--iterations','1','--scan-v2',str(v2).lower(),'--runner','isolated-tail-lookahead-8cores','--hide-progress-bar','--show-metrics','--opt',f'scale-factor={scale}']
    Path(label+'.command.json').write_text(json.dumps(dict(command=cmd,env=record_env),indent=2))
    with open(label+'.txt','w') as log:
        subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
