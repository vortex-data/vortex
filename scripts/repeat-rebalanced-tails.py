# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Repeat frozen scan-tail candidates, retaining each command and raw timing."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

scale=sys.argv[1]
root=Path.cwd()
variants=[('v1',root/'target/release_debug/datafusion-bench',False),('accepted',root/'target/release_debug/datafusion-bench',True),('candidate',root/'target/rebalanced/datafusion-bench',True)]
expected={'v1':'9d17dbcae5ac168ee496c87135a977771e356dfe1dce8589363f9ea03e00e74a','accepted':'9d17dbcae5ac168ee496c87135a977771e356dfe1dce8589363f9ea03e00e74a','candidate':'a1dbc572cbec9b9eaa0d8f461cdc9f9e0d15d976de0b0e5b77ecc2380476a388'}
hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path,_ in variants}
assert hashes==expected,hashes
provenance=dict(run=os.environ['GITHUB_RUN_ID'],head=os.environ['GITHUB_SHA'],binary_runs={'v1':40613767503,'accepted':40613767503,'candidate':40627772121},binary_sha256=hashes,rounds=3,iterations=21,discard=2,queries=[28,32,41,58],scale=scale)
Path('scan-provenance.json').write_text(json.dumps(provenance,indent=2))
env=dict(os.environ,TOKIO_WORKER_THREADS='8',DATAFUSION_EXECUTION_TARGET_PARTITIONS='8',RUST_LOG='warn')
for rnd in range(3):
    for name,binary,v2 in variants[rnd:]+variants[:rnd]:
        label=f'scan-repeat-r{rnd}-{name}'
        cmd=[str(binary),'tpcds','--formats','vortex','--queries','28,32,41,58','--iterations','21','--scan-v2',str(v2).lower(),'--hide-progress-bar','--display-format','gh-json','--runner','rebalanced-tails-long-repeat','--ingest-jsonl',label+'.jsonl','--opt',f'scale-factor={scale}']
        Path(label+'.command.json').write_text(json.dumps(dict(command=cmd,env={k:env[k] for k in ['TOKIO_WORKER_THREADS','DATAFUSION_EXECUTION_TARGET_PARTITIONS','RUST_LOG']}),indent=2))
        with open(label+'.txt','w') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        print('Completed',label,flush=True)
