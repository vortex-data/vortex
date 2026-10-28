# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Compare frozen scan-batching candidates without rebuilding or mixing changes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path.cwd()
variants = [('v1', root/'target/release_debug/datafusion-bench', False), ('accepted', root/'target/release_debug/datafusion-bench', True), ('density', root/'target/density/datafusion-bench', True), ('aligned', root/'target/aligned/datafusion-bench', True)]
expected = {'v1':'9d17dbcae5ac168ee496c87135a977771e356dfe1dce8589363f9ea03e00e74a','accepted':'9d17dbcae5ac168ee496c87135a977771e356dfe1dce8589363f9ea03e00e74a','density':'939f157d66c86498c261ea4e612cc4f29d16561c2945ce36addef63325194df2','aligned':'20086be38ecc47bdbcbd22613b007d54f884e97fbf78923d1c5f5c70856a9bf7'}
hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path,_ in variants}
assert hashes == expected, hashes
provenance=dict(run=os.environ['GITHUB_RUN_ID'],head=os.environ['GITHUB_SHA'],binary_runs={'v1':40613767503,'accepted':40613767503,'density':40625109574,'aligned':40620380829},binary_sha256=hashes,rounds=3,iterations=21,discard=2,queries=[13,25,28])
Path('scan-provenance.json').write_text(json.dumps(provenance,indent=2))
env=dict(os.environ,TOKIO_WORKER_THREADS='8',DATAFUSION_EXECUTION_TARGET_PARTITIONS='8',RUST_LOG='warn')
for rnd in range(3):
    for name,binary,v2 in variants[rnd:]+variants[:rnd]:
        label=f'scan-repeat-r{rnd}-{name}'
        cmd=[str(binary),'clickbench','--formats','vortex','--queries','13,25,28','--iterations','21','--scan-v2',str(v2).lower(),'--hide-progress-bar','--display-format','gh-json','--runner','click-batching-long-repeat','--ingest-jsonl',label+'.jsonl']
        Path(label+'.command.json').write_text(json.dumps(dict(command=cmd,env={k:env[k] for k in ['TOKIO_WORKER_THREADS','DATAFUSION_EXECUTION_TARGET_PARTITIONS','RUST_LOG']}),indent=2))
        with open(label+'.txt','w') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        print('Completed',label,flush=True)
