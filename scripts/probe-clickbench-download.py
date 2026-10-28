# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import time

spec = importlib.util.spec_from_file_location('prefetch', 'scripts/prefetch-clickbench.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
started = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    records = list(pool.map(module.download, range(4)))
result = dict(head=os.environ['GITHUB_SHA'], run=os.environ['GITHUB_RUN_ID'],
              seconds=time.monotonic()-started, shards=records)
Path('scan-download-provenance.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(seconds=result['seconds'], total_bytes=sum(r['size'] for r in records))), flush=True)
