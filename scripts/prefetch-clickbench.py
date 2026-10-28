# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Runner-only resumable prefetch of the benchmark's unchanged Parquet inputs."""
import concurrent.futures
import hashlib
import http.client
import json
import subprocess
import tempfile
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
            print(f'Retrying download after {type(error).__name__}: {error}', flush=True)
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
                with tempfile.TemporaryDirectory(prefix='clickbench-range-') as temporary:
                    headers_path = Path(temporary) / 'headers'
                    body_path = Path(temporary) / 'body'
                    command = ['curl', '--silent', '--show-error', '--fail',
                               '--connect-timeout', '5', '--max-time', '10',
                               '--range', f'{offset}-{end}',
                               '--header', f'If-Match: {info["etag"]}',
                               '--user-agent', HEADERS['User-Agent'],
                               '--dump-header', str(headers_path),
                               '--output', str(body_path), url]
                    started = time.monotonic()
                    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
                    if result.returncode:
                        raise OSError(f'curl range failed ({result.returncode}): {result.stderr.strip()}')
                    # Proxies can prepend a CONNECT response; validate the final response block.
                    blocks = headers_path.read_text().strip().split('\n\n')
                    lines = blocks[-1].splitlines()
                    headers = dict(line.split(':', 1) for line in lines[1:] if ':' in line)
                    headers = {k.lower(): v.strip() for k, v in headers.items()}
                    if (lines[0].split()[1] != '206'
                            or headers.get('content-range') != f'bytes {offset}-{end}/{info["size"]}'
                            or headers.get('etag') != info['etag']):
                        raise ValueError(f'Range or identity mismatch for shard {index}')
                    data = body_path.read_bytes()
                    if len(data) != end - offset + 1:
                        raise OSError('Incomplete range response')
                    print(json.dumps(dict(shard=index, offset=offset, bytes=len(data),
                                          seconds=time.monotonic()-started)), flush=True)
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
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        pending = [pool.submit(download, index) for index in range(100)]
        for future in concurrent.futures.as_completed(pending):
            results.append(future.result())
            Path('scan-download-provenance.json').write_text(json.dumps(results, indent=2) + '\n')
    assert {result['index'] for result in results} == set(range(100))
