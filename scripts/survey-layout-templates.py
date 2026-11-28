# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Survey retained develop benchmark objects and the published random-access fixtures.

The TSV is one row per input: path, file bytes, footer bytes, layout bytes,
prototype bytes, node occurrences, and distinct per-file templates.
"""

import argparse
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path



def inventory(client):
    bucket = "vortex-ci-benchmark-datasets"
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="develop/")
    latest = {}
    for page in pages:
        for entry in page.get("Contents", []):
            key = entry["Key"]
            # develop/<run id>/<preset>/<benchmark path>. Keep the latest generation
            # of each path within each preset so reruns don't weight a dataset twice.
            parts = key.split("/", 2)
            if len(parts) == 3 and parts[1].isdecimal() and key.endswith(".vortex"):
                run = int(parts[1])
                if parts[2] not in latest or latest[parts[2]][0] < run:
                    latest[parts[2]] = (run, key, entry["Size"], entry["ETag"])
    if not latest:
        raise RuntimeError("No retained develop benchmark objects found")
    print(f"Surveying {len(latest)} distinct develop benchmark paths", flush=True)
    for _, key, size, etag in sorted(latest.values()):
        yield client, bucket, key, size, etag


def fixtures():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    client = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
    bucket = "vortex-public-benchmark-fixtures"
    for name in ["taxi/taxi", *(f"random_access/{name}/{name}" for name in ("feature_vectors", "nested_lists", "nested_structs"))]:
        key = f"random-access/v1/{name}.vortex"
        head = client.head_object(Bucket=bucket, Key=key)
        yield client, bucket, key, head["ContentLength"], head["ETag"]


def measure(binary, item):
    client, bucket, key, size, etag = item
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "footer.vortex"
        with path.open("wb") as file:
            def fetch(start, end):
                response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end - 1}", IfMatch=etag)
                with response["Body"] as stream:
                    data = stream.read()
                if len(data) != end - start:
                    raise ValueError(f"Short range read for {key}")
                file.seek(start)
                file.write(data)
                file.flush()

            tail = max(0, size - 65536)
            fetch(tail, size)
            start = int(subprocess.check_output([binary, "--footer-start", path]))
            if start < tail:
                fetch(start, tail)
        row = subprocess.check_output([binary, path], text=True)
        return f"s3://{bucket}/{key}\t" + row.split("\t", 1)[1]


def summarize(lines):
    groups = defaultdict(lambda: [0] * 7)
    for line in lines:
        name, *fields = line.rstrip().split("\t")
        if name.startswith("s3://vortex-ci-benchmark-datasets/develop/"):
            group = name.split("/", 6)[6].split("/")[0]
        elif name.startswith("s3://"):
            group = "random-access fixtures"
        else:
            group = name.split("/")[0]
        values = [1, *map(int, fields)]
        for key in (group, "All files"):
            groups[key] = [a + b for a, b in zip(groups[key], values, strict=True)]
    print("| Dataset | Files | Layout bytes | Prototype bytes | Layout saved | Footer saved | File saved | Nodes | Templates¹ | Repeated nodes |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for group, (n, size, footer, old, new, nodes, templates) in sorted(groups.items()):
        saved = old - new
        print(f"| {group} | {n} | {old:,} | {new:,} | {saved/old:.1%} | {saved/footer:.1%} | "
              f"{saved/size:.3%} | {nodes:,} | {templates:,} | {1-templates/nodes:.1%} |")
    print("\n¹ Sum of file-local distinct templates. Percentages are weighted by bytes or occurrences.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.summary:
        with args.summary.open() as file:
            summarize(file)
        return
    if not args.binary or not args.output:
        parser.error("specify --binary and --output, or --summary")
    import boto3

    items = list(inventory(boto3.client("s3", region_name="us-east-1")))
    items.extend(fixtures())
    with ThreadPoolExecutor(max_workers=8) as pool, args.output.open("w") as output:
        for row in pool.map(lambda item: measure(args.binary.resolve(), item), items):
            output.write(row)


if __name__ == "__main__":
    main()
