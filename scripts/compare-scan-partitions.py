# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

scale = sys.argv[1]
env = dict(os.environ, TOKIO_WORKER_THREADS='8', DATAFUSION_EXECUTION_TARGET_PARTITIONS='8', RUST_LOG='warn')
variants = [('v1','target/release_debug/datafusion-bench',False), ('published','target/structural-baseline/datafusion-bench',True), ('coarse','target/release_debug/datafusion-bench',True), ('coarse-coordinated','target/coordinated-coarse/datafusion-bench',True)]
for binary in {v[1] for v in variants}:
    Path(binary).chmod(0o755)
hashes = {binary:hashlib.sha256(Path(binary).read_bytes()).hexdigest() for binary in {v[1] for v in variants}}
def run(variant, queries, iterations, label):
    name,binary,v2 = variant
    cmd = [binary,'tpcds','--formats','vortex','--queries',queries,'--iterations',str(iterations),'--scan-v2',str(v2).lower(),'--hide-progress-bar','--display-format','gh-json','--runner','paired-partition-8workers','--opt','scale-factor='+scale,'--ingest-jsonl',label+'.jsonl']
    meta = {'command':cmd,'binary_sha256':hashes[binary],'env':{k:env[k] for k in ['TOKIO_WORKER_THREADS','DATAFUSION_EXECUTION_TARGET_PARTITIONS','RUST_LOG']}}
    Path(label+'.command.json').write_text(json.dumps(meta,indent=2))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    with open(label+'.log','w') as log:
        subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    meta.update(elapsed_seconds=time.monotonic()-start,user_seconds=after.ru_utime-before.ru_utime,system_seconds=after.ru_stime-before.ru_stime,input_blocks=after.ru_inblock-before.ru_inblock,major_faults=after.ru_majflt-before.ru_majflt)
    Path(label+'.resources.json').write_text(json.dumps(meta,indent=2))
for round in range(3):
    order = variants[round:]+variants[:round]
    for variant in order:
        run(variant,'13,28,41,42,49,52,55,58,76,96',9,f'scan-r{round}-{variant[0]}')
for round in range(2):
    for variant in (variants if round==0 else variants[::-1]):
        run(variant,'41',300,f'scan-cpu-r{round}-{variant[0]}')
