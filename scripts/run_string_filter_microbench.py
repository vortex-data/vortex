#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Run string-filter microbenchmarks from previously mined query files."""

from __future__ import annotations

import argparse
from pathlib import Path

from string_filter_bench_lib import cargo_prefix
from string_filter_bench_lib import copy_split_result
from string_filter_bench_lib import dataset_output_dir
from string_filter_bench_lib import die
from string_filter_bench_lib import load_queries
from string_filter_bench_lib import load_queries_from_path
from string_filter_bench_lib import named_queries_path
from string_filter_bench_lib import print_dataset_groups
from string_filter_bench_lib import print_query_summary
from string_filter_bench_lib import print_query_type_summary
from string_filter_bench_lib import queries_path
from string_filter_bench_lib import resolve_datasets
from string_filter_bench_lib import results_path
from string_filter_bench_lib import run_cmd
from string_filter_bench_lib import validate_runner_query_types
from string_filter_bench_lib import warn_missing_requested_types


def handle_list(args: argparse.Namespace) -> None:
    print_dataset_groups()
    print_query_type_summary()
    print()
    print("Run workflow")
    print("  1. read one previously mined query file")
    print("  2. show each microbenchmark as apply(query, column)")
    print("  3. invoke the Rust runner on that dataset")
    print()
    print("Explicit workload files")
    print("  use --query-file for an exact file path")
    print("  or use --query-name to resolve <stem>_queries_<name>.json")


def handle_run(args: argparse.Namespace) -> None:
    datasets = resolve_datasets(args.group, args.dataset)
    validate_runner_query_types(args.query_type)
    prefix = cargo_prefix(args.release)

    print("Running string-filter microbenchmarks")
    print(f"group:       {args.group}")
    print(f"datasets:    {' '.join(datasets)}")
    print(f"profile:     {'release' if args.release else 'debug'}")
    print(f"warmup:      {args.warmup}")
    print(f"iterations:  {args.iterations}")

    for dataset in datasets:
        if args.query_file is not None:
            if len(datasets) != 1:
                die("--query-file only supports a single selected dataset")
            query_file = args.query_file
        elif args.query_name is not None:
            query_file = named_queries_path(dataset, args.query_name)
        else:
            query_file = queries_path(dataset)

        if not query_file.exists():
            die(
                f"missing query file for {dataset}: {query_file}\n"
                "Run the mining script first."
            )

        if query_file == queries_path(dataset):
            queries = load_queries(dataset)
        else:
            queries = load_queries_from_path(query_file)
        requested_types = list(args.query_type)

        print()
        print(f"== {dataset} ==")
        print(f"output dir: {dataset_output_dir(dataset)}")
        print(f"queries:    {query_file}")
        print(f"results:    {results_path(dataset)}")
        print_query_summary(dataset, queries)
        warn_missing_requested_types(dataset, queries, requested_types)

        if args.split_query_types:
            split_types = requested_types or ["like_prefix", "like_substr", "regex_basic"]
            for query_type in split_types:
                run_cmd(
                    prefix
                    + [
                        "run",
                        "--warmup",
                        str(args.warmup),
                        "--iterations",
                        str(args.iterations),
                        "--queries-file",
                        str(query_file),
                        "--query-type",
                        query_type,
                        dataset,
                    ],
                    args.dry_run,
                )
                copy_split_result(dataset, query_type, dry_run=args.dry_run)
            continue

        cmd = prefix + [
            "run",
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--queries-file",
            str(query_file),
        ]
        if len(requested_types) == 1:
            cmd.extend(["--query-type", requested_types[0]])
        elif len(requested_types) > 1:
            die("multiple --query-type values require --split-query-types")
        cmd.append(dataset)
        run_cmd(cmd, args.dry_run)


def add_dataset_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--group",
        default="paper-dev",
        choices=["all", "clickbench", "paper-dev"],
        help="predefined dataset group",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="explicit dataset to use; may be repeated",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run string-filter microbenchmarks from mined query files"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list",
        help="list datasets and runner query classes",
    )
    list_parser.set_defaults(func=handle_list)

    run_parser = subparsers.add_parser(
        "run",
        help="run the microbenchmarks",
    )
    add_dataset_selection(run_parser)
    run_parser.add_argument("--warmup", type=int, default=3)
    run_parser.add_argument("--iterations", type=int, default=10)
    run_parser.add_argument(
        "--query-file",
        type=Path,
        help="explicit query workload JSON file; only valid with one dataset",
    )
    run_parser.add_argument(
        "--query-name",
        help="resolve <stem>_queries_<name>.json for each dataset",
    )
    run_parser.add_argument(
        "--query-type",
        action="append",
        default=[],
        help="query type filter; repeat only with --split-query-types",
    )
    run_parser.add_argument("--split-query-types", action="store_true")
    run_parser.add_argument("--debug", dest="release", action="store_false")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(release=True, func=handle_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
