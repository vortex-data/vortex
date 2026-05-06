#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import NoReturn


@dataclass(frozen=True)
class DatasetDef:
    key: str
    suite: str
    column: str
    description: str


@dataclass(frozen=True)
class QueryInfo:
    pattern: str
    query_type: str
    selectivity: str
    fsst_difficulty: str
    match_fraction: float


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "vortex-bench" / "data" / "string-filter-bench"

DATASETS: dict[str, DatasetDef] = {
    "clickbench-url": DatasetDef(
        key="clickbench-url",
        suite="ClickBench",
        column="URL",
        description="ClickBench hits URL column",
    ),
    "clickbench-title": DatasetDef(
        key="clickbench-title",
        suite="ClickBench",
        column="Title",
        description="ClickBench hits Title column",
    ),
    "clickbench-referer": DatasetDef(
        key="clickbench-referer",
        suite="ClickBench",
        column="Referer",
        description="ClickBench hits Referer column",
    ),
    "clickbench-search-phrase": DatasetDef(
        key="clickbench-search-phrase",
        suite="ClickBench",
        column="SearchPhrase",
        description="ClickBench hits SearchPhrase column",
    ),
    "clickbench-params": DatasetDef(
        key="clickbench-params",
        suite="ClickBench",
        column="Params",
        description="ClickBench hits Params column",
    ),
    "json-lines": DatasetDef(
        key="json-lines",
        suite="Synthetic",
        column="json_line",
        description="Synthetic NDJSON row strings",
    ),
    "fineweb-url": DatasetDef(
        key="fineweb-url",
        suite="FineWeb",
        column="url",
        description="FineWeb URL column",
    ),
    "fineweb-text": DatasetDef(
        key="fineweb-text",
        suite="FineWeb",
        column="text",
        description="FineWeb text column",
    ),
    "gharchive-repo-name": DatasetDef(
        key="gharchive-repo-name",
        suite="GitHub Archive",
        column="repo.name",
        description="GitHub Archive repo.name field",
    ),
    "gharchive-actor-login": DatasetDef(
        key="gharchive-actor-login",
        suite="GitHub Archive",
        column="actor.login",
        description="GitHub Archive actor.login field",
    ),
    "gharchive-payload-ref": DatasetDef(
        key="gharchive-payload-ref",
        suite="GitHub Archive",
        column="payload.ref",
        description="GitHub Archive payload.ref field",
    ),
    "gharchive-actor-avatar-url": DatasetDef(
        key="gharchive-actor-avatar-url",
        suite="GitHub Archive",
        column="actor.avatar_url",
        description="GitHub Archive actor.avatar_url field",
    ),
    "polarsignals-labels-comm": DatasetDef(
        key="polarsignals-labels-comm",
        suite="PolarSignals",
        column="labels.comm",
        description="PolarSignals labels.comm field",
    ),
    "polarsignals-labels-thread-name": DatasetDef(
        key="polarsignals-labels-thread-name",
        suite="PolarSignals",
        column="labels.thread_name",
        description="PolarSignals labels.thread_name field",
    ),
    "polarsignals-mapping-file": DatasetDef(
        key="polarsignals-mapping-file",
        suite="PolarSignals",
        column="locations.mapping_file",
        description="PolarSignals locations.mapping_file field",
    ),
    "polarsignals-function-name": DatasetDef(
        key="polarsignals-function-name",
        suite="PolarSignals",
        column="locations.lines.function_name",
        description="PolarSignals locations.lines.function_name field",
    ),
    "polarsignals-function-filename": DatasetDef(
        key="polarsignals-function-filename",
        suite="PolarSignals",
        column="locations.lines.function_filename",
        description="PolarSignals locations.lines.function_filename field",
    ),
    "tpch-lineitem": DatasetDef(
        key="tpch-lineitem",
        suite="TPC-H",
        column="lineitem.l_comment",
        description="TPC-H lineitem l_comment column",
    ),
}

DEFAULT_DATASETS = [
    "clickbench-url",
    "clickbench-referer",
    "clickbench-title",
    "clickbench-search-phrase",
    "clickbench-params",
    "tpch-lineitem",
]

DATASET_GROUPS = {
    "paper-dev": DEFAULT_DATASETS,
    "clickbench": [
        "clickbench-url",
        "clickbench-title",
        "clickbench-referer",
        "clickbench-search-phrase",
        "clickbench-params",
    ],
    "all": list(DATASETS),
}

GENERATED_QUERY_TYPES = ["like_prefix", "like_substr", "regex_basic"]
RUNNER_QUERY_TYPES = ["like_prefix", "like_substr", "like_suffix", "regex_basic"]


def die(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def dataset_to_stem(dataset: str) -> str:
    return dataset.replace("-", "_")


def dataset_output_dir(dataset: str) -> Path:
    return DATA_ROOT / dataset_to_stem(dataset)


def queries_path(dataset: str) -> Path:
    stem = dataset_to_stem(dataset)
    return dataset_output_dir(dataset) / f"{stem}_queries.json"


def strings_path(dataset: str) -> Path:
    stem = dataset_to_stem(dataset)
    return dataset_output_dir(dataset) / f"{stem}_strings.txt"


def results_path(dataset: str) -> Path:
    stem = dataset_to_stem(dataset)
    return dataset_output_dir(dataset) / f"{stem}_results.json"


def manifest_path(dataset: str) -> Path:
    stem = dataset_to_stem(dataset)
    return dataset_output_dir(dataset) / f"{stem}_microbenchmarks.json"


def named_queries_path(dataset: str, name: str) -> Path:
    stem = dataset_to_stem(dataset)
    return dataset_output_dir(dataset) / f"{stem}_queries_{name}.json"


def cargo_prefix(release: bool) -> list[str]:
    cmd = ["cargo", "run"]
    if release:
        cmd.append("--release")
    cmd.extend(["-p", "string-filter-bench", "--"])
    return cmd


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("+", shlex.join(cmd))
    if dry_run:
        return

    def run_once(*, clear_rustc_wrapper: bool) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        if clear_rustc_wrapper:
            env["RUSTC_WRAPPER"] = ""
        return subprocess.run(
            cmd,
            check=False,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            env=env,
        )

    result = run_once(clear_rustc_wrapper=False)
    if result.returncode == 0:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        return

    if (
        cmd
        and cmd[0] == "cargo"
        and "sccache: error: Operation not permitted" in result.stderr
    ):
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        print(
            "retrying cargo command with RUSTC_WRAPPER= due to sccache sandbox failure",
            file=sys.stderr,
        )
        retry = run_once(clear_rustc_wrapper=True)
        if retry.stdout:
            print(retry.stdout, end="")
        if retry.stderr:
            print(retry.stderr, end="", file=sys.stderr)
        if retry.returncode == 0:
            return
        raise subprocess.CalledProcessError(retry.returncode, cmd)

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    raise subprocess.CalledProcessError(result.returncode, cmd)


def resolve_datasets(group: str, datasets: list[str]) -> list[str]:
    resolved = datasets or DATASET_GROUPS.get(group)
    if resolved is None:
        die(f"unknown dataset group: {group}")
    unknown = [dataset for dataset in resolved if dataset not in DATASETS]
    if unknown:
        die(f"unknown dataset(s): {', '.join(unknown)}")
    return list(resolved)


def load_queries(dataset: str) -> list[QueryInfo]:
    return load_queries_from_path(queries_path(dataset))


def load_queries_from_path(path: Path) -> list[QueryInfo]:
    if not path.exists():
        return []

    raw = json.loads(path.read_text())
    return [
        QueryInfo(
            pattern=entry["pattern"],
            query_type=entry["query_type"],
            selectivity=entry["selectivity"],
            fsst_difficulty=entry["fsst_difficulty"],
            match_fraction=float(entry["match_fraction"]),
        )
        for entry in raw
    ]


def write_queries_to_path(path: Path, queries: list[QueryInfo], *, dry_run: bool) -> None:
    payload = [
        {
            "pattern": query.pattern,
            "query_type": query.query_type,
            "selectivity": query.selectivity,
            "fsst_difficulty": query.fsst_difficulty,
            "match_fraction": query.match_fraction,
        }
        for query in queries
    ]
    print(f"query file: {path}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def predicate_text(query: QueryInfo) -> str:
    if query.query_type.startswith("like_"):
        return f"LIKE {query.pattern!r}"
    return f"REGEX {query.pattern!r}"


def microbenchmark_rows(dataset: str, queries: list[QueryInfo]) -> list[dict[str, Any]]:
    meta = DATASETS[dataset]
    rows = []
    for index, query in enumerate(queries, start=1):
        rows.append(
            {
                "id": index,
                "suite": meta.suite,
                "dataset": dataset,
                "column": meta.column,
                "query_type": query.query_type,
                "pattern": query.pattern,
                "predicate": predicate_text(query),
                "operation": f"apply {predicate_text(query)} to {meta.column}",
                "selectivity": query.selectivity,
                "match_fraction": query.match_fraction,
                "fsst_difficulty": query.fsst_difficulty,
                "description": (
                    f"Apply {predicate_text(query)} to column {meta.column} "
                    f"from dataset {dataset}"
                ),
            }
        )
    return rows


def write_manifest(dataset: str, queries: list[QueryInfo], *, dry_run: bool) -> None:
    write_manifest_to_path(manifest_path(dataset), dataset, queries, dry_run=dry_run)


def write_manifest_to_path(path: Path, dataset: str, queries: list[QueryInfo], *, dry_run: bool) -> None:
    meta = DATASETS[dataset]
    payload = {
        "benchmark_model": "one query pattern applied to one string column",
        "suite": meta.suite,
        "dataset": dataset,
        "column": meta.column,
        "description": meta.description,
        "query_file": str(queries_path(dataset)),
        "microbenchmark_count": len(queries),
        "microbenchmarks": microbenchmark_rows(dataset, queries),
    }
    print(f"manifest: {path}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def print_query_summary(dataset: str, queries: list[QueryInfo]) -> None:
    meta = DATASETS[dataset]
    print()
    print(f"Dataset: {dataset}")
    print(f"Suite:   {meta.suite}")
    print(f"Column:  {meta.column}")
    print("Microbenchmark model:")
    print(f"  apply one query pattern to one string column: {meta.column}")
    if not queries:
        print("Queries: none found yet")
        print(f"  expected query file: {queries_path(dataset)}")
        return

    counts = Counter(query.query_type for query in queries)
    print(f"Query count: {len(queries)}")
    print("By query type:")
    for query_type, count in sorted(counts.items()):
        print(f"  {query_type}: {count}")

    print("Explicit microbenchmarks:")
    for row in microbenchmark_rows(dataset, queries):
        print(
            "  "
            f"[{row['id']:02d}] {row['query_type']} | {row['predicate']} | "
            f"selectivity={row['selectivity']} | match_fraction={row['match_fraction']:.4f}"
        )


def validate_runner_query_types(query_types: list[str]) -> None:
    invalid = [query_type for query_type in query_types if query_type not in RUNNER_QUERY_TYPES]
    if invalid:
        die(f"unknown query type(s): {', '.join(invalid)}")


def warn_missing_requested_types(dataset: str, queries: list[QueryInfo], query_types: list[str]) -> None:
    if not query_types or not queries:
        return
    present = {query.query_type for query in queries}
    missing = [query_type for query_type in query_types if query_type not in present]
    if missing:
        print(
            f"warning: {dataset} has no mined queries for: {', '.join(missing)}",
            file=sys.stderr,
        )


def copy_split_result(dataset: str, query_type: str, *, dry_run: bool) -> None:
    src = results_path(dataset)
    dst = src.with_name(f"{src.stem}_{query_type}{src.suffix}")
    print(f"results copy: {dst}")
    if dry_run:
        return
    if not src.exists():
        die(f"expected results file was not written: {src}")
    shutil.copyfile(src, dst)


def print_dataset_groups() -> None:
    print("Dataset groups")
    for group_name, datasets in DATASET_GROUPS.items():
        print(f"  {group_name}:")
        for dataset in datasets:
            meta = DATASETS[dataset]
            print(f"    {dataset:<34} {meta.column}")


def print_query_type_summary() -> None:
    print()
    print("Query types")
    print("  mined today:")
    for query_type in GENERATED_QUERY_TYPES:
        print(f"    {query_type}")
    print("  runner accepts:")
    for query_type in RUNNER_QUERY_TYPES:
        print(f"    {query_type}")


def load_strings(dataset: str) -> list[str]:
    path = strings_path(dataset)
    if not path.exists():
        die(
            f"missing strings file for {dataset}: {path}\n"
            "Run prep or the mining script first."
        )
    return path.read_text().splitlines()


def selectivity_label_from_fraction(match_fraction: float) -> str:
    if match_fraction > 0.1:
        return "high"
    if match_fraction > 0.01:
        return "medium"
    return "low"


def query_type_from_like_pattern(pattern: str) -> str:
    starts = pattern.startswith("%")
    ends = pattern.endswith("%")
    core = pattern.strip("%")

    if "%" not in core and "_" not in core:
        if starts and ends:
            return "like_substr"
        if ends and not starts:
            return "like_prefix"
        if starts and not ends:
            return "like_suffix"
    die(f"unsupported LIKE pattern shape for explicit workload: {pattern}")


def sql_like_to_regex(pattern: str) -> re.Pattern[str]:
    parts = ["^"]
    for ch in pattern:
        if ch == "%":
            parts.append(".*")
        elif ch == "_":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    parts.append("$")
    return re.compile("".join(parts))


def estimate_match_fraction(dataset: str, pattern: str, query_type: str) -> float:
    strings = load_strings(dataset)
    if not strings:
        return 0.0

    if query_type.startswith("like_"):
        matcher = sql_like_to_regex(pattern)
        matches = sum(1 for value in strings if matcher.match(value) is not None)
    else:
        matcher = re.compile(pattern)
        matches = sum(1 for value in strings if matcher.search(value) is not None)
    return matches / len(strings)


def make_query_info(pattern: str, query_type: str, dataset: str, fsst_difficulty: str = "easy") -> QueryInfo:
    match_fraction = estimate_match_fraction(dataset, pattern, query_type)
    return QueryInfo(
        pattern=pattern,
        query_type=query_type,
        selectivity=selectivity_label_from_fraction(match_fraction),
        fsst_difficulty=fsst_difficulty,
        match_fraction=match_fraction,
    )


SQL_COUNT_LIKE_RE = re.compile(
    r"""
    ^\s*select\s+count\s*\(\s*\*\s*\)\s+
    from\s+(?P<table>"?[A-Za-z_][A-Za-z0-9_]*"?)\s+
    where\s+(?P<column>"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s+
    like\s+'(?P<pattern>(?:''|[^'])*)'\s*;?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class ParsedLikeSql:
    table: str
    column: str
    pattern: str
    query_type: str


def parse_count_like_sql(sql: str) -> ParsedLikeSql:
    match = SQL_COUNT_LIKE_RE.match(sql)
    if match is None:
        die(
            "only simple COUNT(*) ... WHERE <column> LIKE '<pattern>' SQL is supported "
            "for explicit column workloads"
        )

    table = match.group("table").strip('"')
    column = match.group("column").strip('"')
    pattern = match.group("pattern").replace("''", "'")
    return ParsedLikeSql(
        table=table,
        column=column,
        pattern=pattern,
        query_type=query_type_from_like_pattern(pattern),
    )


def datasets_matching_column(datasets: list[str], column: str) -> list[str]:
    return [dataset for dataset in datasets if DATASETS[dataset].column == column]
