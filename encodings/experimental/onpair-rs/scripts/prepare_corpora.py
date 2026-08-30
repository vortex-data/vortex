#!/usr/bin/env python3
"""Materialize deterministic, byte-capped corpora for the OnPair benchmarks.

The output format is deliberately tiny and dependency-free for the Rust reader:
``ONPAIR01``, followed by little-endian payload-byte and row counts, followed by
``u32 length, bytes`` records.  The payload (excluding framing) is exactly the
requested size; only the final row may be truncated.

Hugging Face and Parquet adapters require the ``datasets`` package. Paper
artifact downloads are parsed with the standard library and streamed rather
than retained in full.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import struct
import tarfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MIB = 1024 * 1024
MAGIC = b"ONPAIR01"

FINEWEB_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
STACK_V3_REVISION = "df4b205fbba4cc1c2fd1f205b10d66f730798bb9"
CLICKBENCH_PARQUET = (
    "https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet"
)
PAPER_ARTIFACT_REVISION = "ef3360530e9e963dedc3b59280b5bc2014ce7416"
MSMARCO_QUERIES_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz"
)
MSMARCO_URLS_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz"
)
AMAZON_BOOK_TITLES_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "meta_categories/meta_Books.jsonl.gz"
)
DBPEDIA_ABSTRACTS_URL = (
    "https://databus.dbpedia.org/dbpedia/text/short-abstracts/2022.12.01/"
    "short-abstracts_lang=en.ttl.bz2"
)
AMAZON_BOOK_REVIEWS_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "review_categories/Books.jsonl.gz"
)
NEWS_REVISION = "bc91c8c8dbea6a44069e0a955b6ed8dd54fb7fe3"
TWEETS_REVISION = "c8aaf1333e9d0cf5886f4eb9506ad917f1d8931c"
NASA_ACCESS_LOG_URL = "https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz"


def open_binary(source: str) -> Any:
    """Open a local file or URL as a binary context manager."""
    local = Path(source)
    if local.is_file():
        return local.open("rb")
    request = urllib.request.Request(source, headers={"User-Agent": "onpair-benchmark/1"})
    return urllib.request.urlopen(request)


def msmarco_queries(source: str) -> Iterable[str]:
    with open_binary(source) as archive:
        with tarfile.open(fileobj=archive, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".tsv"):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                for raw_line in extracted:
                    fields = raw_line.decode("utf-8").split("\t", 1)
                    if len(fields) == 2 and (query := fields[1].strip()):
                        yield query


def msmarco_urls(source: str) -> Iterable[str]:
    with open_binary(source) as archive:
        with gzip.open(archive, mode="rt", encoding="utf-8") as lines:
            for line in lines:
                fields = line.split("\t", 2)
                if len(fields) >= 2 and fields[1].startswith("http"):
                    yield fields[1]


def amazon_book_titles(source: str) -> Iterable[str]:
    with open_binary(source) as archive:
        with gzip.open(archive, mode="rt", encoding="utf-8") as lines:
            for line in lines:
                try:
                    title = json.loads(line).get("title", "").strip()
                except (json.JSONDecodeError, AttributeError):
                    continue
                if title:
                    yield title


def amazon_book_reviews(source: str) -> Iterable[str]:
    with open_binary(source) as archive:
        with gzip.open(archive, mode="rt", encoding="utf-8") as lines:
            for line in lines:
                try:
                    review = json.loads(line).get("text", "").strip()
                except (json.JSONDecodeError, AttributeError):
                    continue
                if review:
                    yield review


def huggingface_field(
    dataset: str, field: str, revision: str, cache_dir: str | None
) -> Iterable[str]:
    from datasets import load_dataset

    rows = load_dataset(
        dataset,
        split="train",
        streaming=True,
        revision=revision,
        cache_dir=cache_dir,
    )
    for row in rows:
        value = row[field]
        if value:
            yield value


def dbpedia_abstracts(source: str) -> Iterable[str]:
    count = 0
    with open_binary(source) as archive:
        with bz2.open(archive, mode="rt", encoding="utf-8") as lines:
            for line in lines:
                if line.startswith("#") or not line.strip():
                    continue
                start = line.find('"')
                end = line.rfind('"@en')
                if start == -1 or end <= start:
                    continue
                yield line[start + 1 : end]
                count += 1
                if count == 1_000_000:
                    break


def apache_access_logs(source: str) -> Iterable[bytes]:
    """Yield complete Common Log Format records without newline delimiters."""
    with open_binary(source) as archive:
        with gzip.open(archive, mode="rb") as lines:
            for line in lines:
                value = line.rstrip(b"\r\n")
                if value:
                    yield value


def fineweb(cache_dir: str | None) -> Iterable[str]:
    from datasets import load_dataset

    rows = load_dataset(
        "HuggingFaceFW/fineweb",
        "sample-10BT",
        split="train",
        streaming=True,
        revision=FINEWEB_REVISION,
        cache_dir=cache_dir,
    )
    for row in rows:
        yield row["text"]


def stack_v3(cache_dir: str | None) -> Iterable[str]:
    from datasets import load_dataset

    repos = load_dataset(
        "HuggingFaceCode/stack-v3-train",
        split="train",
        streaming=True,
        revision=STACK_V3_REVISION,
        cache_dir=cache_dir,
    )
    repos = repos.select_columns(["files"])
    for repo in repos:
        for file in repo["files"]:
            content = file.get("content")
            if content:
                yield content


def parquet_rows(
    source: str, column: str, cache_dir: str | None
) -> Iterable[str | bytes]:
    local = Path(source)
    if local.is_file():
        import pyarrow.parquet as parquet

        file = parquet.ParquetFile(local)
        for batch in file.iter_batches(columns=[column]):
            for value in batch.column(0).to_pylist():
                if value is not None:
                    yield value
        return
    from datasets import load_dataset

    rows = load_dataset(
        "parquet",
        data_files={"train": source},
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )
    rows = rows.select_columns([column])
    for row in rows:
        value = row[column]
        if value is not None:
            yield value


def source_rows(args: argparse.Namespace) -> tuple[Iterable[str | bytes], dict[str, Any]]:
    if args.dataset == "fineweb":
        return fineweb(args.cache_dir), {
            "dataset": "HuggingFaceFW/fineweb",
            "config": "sample-10BT",
            "revision": FINEWEB_REVISION,
            "field": "text",
        }
    if args.dataset == "stack-v3":
        return stack_v3(args.cache_dir), {
            "dataset": "HuggingFaceCode/stack-v3-train",
            "revision": STACK_V3_REVISION,
            "field": "files[].content",
        }
    if args.dataset == "clickbench":
        return parquet_rows(CLICKBENCH_PARQUET, args.column or "URL", args.cache_dir), {
            "dataset": "ClickHouse/ClickBench",
            "source": CLICKBENCH_PARQUET,
            "field": args.column or "URL",
        }
    paper_sources = {
        "msmarco-queries": (
            msmarco_queries,
            MSMARCO_QUERIES_URL,
            "queries.tar.gz/*.tsv: field 2",
        ),
        "msmarco-urls": (
            msmarco_urls,
            MSMARCO_URLS_URL,
            "msmarco-docs.tsv.gz: URL",
        ),
        "amazon-book-titles": (
            amazon_book_titles,
            AMAZON_BOOK_TITLES_URL,
            "meta_Books.jsonl.gz: title",
        ),
        "dbpedia-abstracts": (
            dbpedia_abstracts,
            DBPEDIA_ABSTRACTS_URL,
            "short-abstracts_lang=en.ttl.bz2: English literal",
        ),
    }
    if args.dataset in paper_sources:
        adapter, default_source, field = paper_sources[args.dataset]
        source = args.source or default_source
        return adapter(source), {
            "dataset": args.dataset,
            "source": source,
            "field": field,
            "paper_artifact_revision": PAPER_ARTIFACT_REVISION,
        }
    if args.dataset == "paper-book-reviews":
        source = args.source or AMAZON_BOOK_REVIEWS_URL
        return amazon_book_reviews(source), {
            "dataset": "Amazon Reviews 2023 / Books",
            "source": source,
            "field": "text",
            "paper_table_2_dataset": "Book Reviews",
        }
    if args.dataset == "paper-news-headlines":
        return huggingface_field(
            "DeveloperOats/Million_News_Headlines",
            "headline_text",
            NEWS_REVISION,
            args.cache_dir,
        ), {
            "dataset": "DeveloperOats/Million_News_Headlines",
            "revision": NEWS_REVISION,
            "field": "headline_text",
            "paper_table_2_dataset": "News Headlines",
        }
    if args.dataset == "paper-tweets":
        return huggingface_field(
            "bdanko/sentiment140", "text", TWEETS_REVISION, args.cache_dir
        ), {
            "dataset": "bdanko/sentiment140",
            "revision": TWEETS_REVISION,
            "field": "text",
            "paper_table_2_dataset": "Tweets",
        }
    if args.dataset == "tpch-l-comment":
        source = args.source or str(
            Path("vortex-bench/data/tpch/1.0/parquet/lineitem_0.parquet")
        )
        return parquet_rows(source, "l_comment", args.cache_dir), {
            "dataset": "TPC-H",
            "scale_factor": "1.0",
            "source": source,
            "field": "lineitem.l_comment",
            "generator": "tpchgen-rs@438e9c2dbc25b2fff82c0efc08b3f13b5707874f",
        }
    if args.dataset == "apache-access":
        source = args.source or NASA_ACCESS_LOG_URL
        return apache_access_logs(source), {
            "dataset": "LBNL NASA-HTTP",
            "source": source,
            "field": "complete access-log line",
            "trace": "NASA_access_log_Jul95.gz",
        }
    if args.dataset == "parquet":
        if not args.source or not args.column:
            raise SystemExit("parquet requires --source and --column")
        return parquet_rows(args.source, args.column, args.cache_dir), {
            "dataset": "local-parquet",
            "source": args.source,
            "field": args.column,
        }
    raise AssertionError(args.dataset)


def materialize(rows: Iterable[str | bytes], target: int) -> list[bytes]:
    result: list[bytes] = []
    total = 0
    for value in rows:
        data = value if isinstance(value, bytes) else value.encode("utf-8")
        if not data:
            continue
        remaining = target - total
        if len(data) >= remaining:
            result.append(data[:remaining])
            total += remaining
            break
        result.append(data)
        total += len(data)
    if total != target:
        raise RuntimeError(f"source ended at {total} bytes, before target {target}")
    return result


def write_corpus(path: Path, rows: list[bytes], payload_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as output:
        header = MAGIC + struct.pack("<QQ", payload_bytes, len(rows))
        output.write(header)
        digest.update(header)
        for row in rows:
            framed = struct.pack("<I", len(row)) + row
            output.write(framed)
            digest.update(framed)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=(
            "fineweb",
            "clickbench",
            "stack-v3",
            "msmarco-queries",
            "msmarco-urls",
            "amazon-book-titles",
            "dbpedia-abstracts",
            "paper-book-reviews",
            "paper-news-headlines",
            "paper-tweets",
            "tpch-l-comment",
            "apache-access",
            "parquet",
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/onpair-corpora"))
    parser.add_argument("--name", help="output stem (defaults to the dataset adapter name)")
    parser.add_argument("--size-mib", type=int, default=32)
    parser.add_argument(
        "--source", help="override source path or URL (required for generic parquet)"
    )
    parser.add_argument("--column", help="column for ClickBench/parquet (ClickBench: URL)")
    parser.add_argument(
        "--cache-dir",
        default="/tmp/onpair-huggingface-cache",
        help="Hugging Face datasets cache (default: /tmp/onpair-huggingface-cache)",
    )
    args = parser.parse_args()

    target = args.size_mib * MIB
    rows, provenance = source_rows(args)
    try:
        materialized = materialize(rows, target)
    finally:
        close = getattr(rows, "close", None)
        if close is not None:
            close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.name or args.dataset}-{args.size_mib}mib.onpair"
    sha256 = write_corpus(output, materialized, target)

    manifest = {
        **provenance,
        "format": "ONPAIR01",
        "payload_bytes": target,
        "rows": len(materialized),
        "sha256": sha256,
        "output": output.name,
        "final_row_may_be_truncated": True,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
