# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Runner-only resumable prefetch of the benchmark's unchanged Parquet inputs."""
import concurrent.futures
import hashlib
import http.client
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

BASE = 'https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_many'
DESTINATION = Path('vortex-bench/data/clickbench_partitioned/parquet')
CHUNK = 8 << 20
HEADERS = {'User-Agent': 'vortex-bench-prefetch/1.0', 'Accept-Encoding': 'identity'}


def retry(operation):
    for attempt in range(12):
        try:
            return operation()
        except (OSError, TimeoutError, http.client.IncompleteRead) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code in (403, 404, 412, 416):
                raise
            if attempt == 11:
                raise
            time.sleep(min(2 ** attempt, 20))


def metadata(url):
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD', headers=HEADERS), timeout=60) as response:
        size = int(response.headers['Content-Length'])
        etag = response.headers['ETag']
        if size < 8 or not etag or etag.startswith('W/'):
            raise ValueError('Missing strong object identity or invalid file size')
        return {'url': url, 'size': size, 'etag': etag}


def valid(path, size):
    if not path.exists() or path.stat().st_size != size:
        return False
    with path.open('rb') as stream:
        first = stream.read(4)
        stream.seek(-4, 2)
        return first == b'PAR1' and stream.read(4) == b'PAR1'


def download(index, destination=DESTINATION, base=BASE):
    url = f'{base}/hits_{index}.parquet'
    info = retry(lambda: metadata(url))
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / f'hits_{index}.parquet'
    part = final.with_suffix('.parquet.part')
    identity = final.with_suffix('.parquet.identity.json')
    if identity.exists() and json.loads(identity.read_text()) != info:
        raise ValueError(f'Source identity changed for shard {index}')
    if not identity.exists():
        if part.exists() or final.exists():
            raise ValueError(f'Unidentified preexisting data for shard {index}')
        identity.write_text(json.dumps(info))
    if final.exists():
        if not valid(final, info['size']):
            raise ValueError(f'Invalid completed shard {index}')
    else:
        offset = part.stat().st_size if part.exists() else 0
        if offset > info['size']:
            raise ValueError(f'Oversized partial shard {index}')
        while offset < info['size']:
            end = min(offset + CHUNK, info['size']) - 1

            def read_range():
                request = urllib.request.Request(url, headers={
                    'Range': f'bytes={offset}-{end}', 'If-Match': info['etag'],
                    **HEADERS,
                })
                with urllib.request.urlopen(request, timeout=60) as response:
                    if (response.status != 206
                            or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{info["size"]}'
                            or response.headers.get('ETag') != info['etag']):
                        raise ValueError(f'Range or identity mismatch for shard {index}')
                    data = response.read(end - offset + 2)
                    if len(data) != end - offset + 1:
                        raise OSError('Incomplete range response')
                    return data

            data = retry(read_range)
            with part.open('ab') as stream:
                stream.write(data)
            offset += len(data)
        if not valid(part, info['size']):
            raise ValueError(f'Invalid Parquet framing for shard {index}')
        part.replace(final)
    digest = hashlib.sha256()
    with final.open('rb') as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    print(f'Completed shard {index}: {info["size"]} bytes', flush=True)
    return dict(info, index=index, sha256=digest.hexdigest())


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, range(100)))
    Path('scan-download-provenance.json').write_text(json.dumps(results, indent=2) + '\n')
