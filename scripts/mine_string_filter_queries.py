#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Prepare data and mine explicit query sets for string-filter microbenchmarks."""

from __future__ import annotations

import argparse

from string_filter_bench_lib import cargo_prefix
from string_filter_bench_lib import datasets_matching_column
from string_filter_bench_lib import dataset_output_dir
from string_filter_bench_lib import make_query_info
from string_filter_bench_lib import named_queries_path
from string_filter_bench_lib import parse_count_like_sql
from string_filter_bench_lib import load_queries
from string_filter_bench_lib import print_dataset_groups
from string_filter_bench_lib import print_query_summary
from string_filter_bench_lib import print_query_type_summary
from string_filter_bench_lib import queries_path
from string_filter_bench_lib import resolve_datasets
from string_filter_bench_lib import run_cmd
from string_filter_bench_lib import write_manifest
from string_filter_bench_lib import write_manifest_to_path
from string_filter_bench_lib import write_queries_to_path


def handle_list(args: argparse.Namespace) -> None:
    print_dataset_groups()
    print_query_type_summary()
    print()
    print("Mining workflow")
    print("  1. prep one string column")
    print("  2. mine queries from that column")
    print("  3. write an explicit microbenchmark manifest")
    print()
    print("Explicit SQL workflow")
    print('  1. parse SQL like: SELECT COUNT(*) FROM hits WHERE "URL" LIKE \'%google%\';')
    print("  2. map the SQL column to one prepared column dataset")
    print("  3. write a one-query workload file")


def handle_mine(args: argparse.Namespace) -> None:
    datasets = resolve_datasets(args.group, args.dataset)
    prefix = cargo_prefix(args.release)

    print("Mining string-filter query sets")
    print(f"group:       {args.group}")
    print(f"datasets:    {' '.join(datasets)}")
    print(f"profile:     {'release' if args.release else 'debug'}")
    print(f"max rows:    {args.max_rows}")
    print(f"sample size: {args.sample_size}")
    print(f"queries/cat: {args.queries_per_category}")

    for dataset in datasets:
        print()
        print(f"== {dataset} ==")
        print(f"output dir: {dataset_output_dir(dataset)}")
        print(f"queries:    {queries_path(dataset)}")

        if not args.skip_prep:
            run_cmd(prefix + ["prep", "--max-rows", str(args.max_rows), dataset], args.dry_run)

        run_cmd(
            prefix
            + [
                "mine",
                "--sample-size",
                str(args.sample_size),
                "--queries-per-category",
                str(args.queries_per_category),
                dataset,
            ],
            args.dry_run,
        )

        queries = load_queries(dataset)
        print_query_summary(dataset, queries)
        write_manifest(dataset, queries, dry_run=args.dry_run)


def handle_describe(args: argparse.Namespace) -> None:
    datasets = resolve_datasets(args.group, args.dataset)
    prefix = cargo_prefix(args.release)

    for dataset in datasets:
        if args.mine_if_missing and not queries_path(dataset).exists():
            if not dataset_output_dir(dataset).exists():
                run_cmd(prefix + ["prep", "--max-rows", str(args.max_rows), dataset], args.dry_run)
            run_cmd(
                prefix
                + [
                    "mine",
                    "--sample-size",
                    str(args.sample_size),
                    "--queries-per-category",
                    str(args.queries_per_category),
                    dataset,
                ],
                args.dry_run,
            )

        queries = load_queries(dataset)
        print_query_summary(dataset, queries)
        write_manifest(dataset, queries, dry_run=args.dry_run)


def handle_sql(args: argparse.Namespace) -> None:
    parsed = parse_count_like_sql(args.sql)
    datasets = resolve_datasets(args.group, args.dataset)
    matching_datasets = datasets_matching_column(datasets, parsed.column)
    prefix = cargo_prefix(args.release)

    if not matching_datasets:
        available = ", ".join(datasets)
        raise SystemExit(
            f'error: SQL column "{parsed.column}" does not match the selected dataset set: {available}'
        )

    print("Building explicit SQL query workload")
    print(f"sql:         {args.sql}")
    print(f"table:       {parsed.table}")
    print(f"column:      {parsed.column}")
    print(f"pattern:     {parsed.pattern}")
    print(f"query type:  {parsed.query_type}")
    print(f"datasets:    {' '.join(matching_datasets)}")

    for dataset in matching_datasets:
        query_file = named_queries_path(dataset, args.name)
        manifest_file = query_file.with_name(query_file.stem.replace("_queries_", "_microbenchmarks_") + query_file.suffix)

        print()
        print(f"== {dataset} ==")
        print(f"output dir: {dataset_output_dir(dataset)}")
        print(f"query file: {query_file}")
        print(f"manifest:   {manifest_file}")

        if not args.skip_prep:
            run_cmd(prefix + ["prep", "--max-rows", str(args.max_rows), dataset], args.dry_run)

        query = make_query_info(parsed.pattern, parsed.query_type, dataset)
        queries = [query]
        write_queries_to_path(query_file, queries, dry_run=args.dry_run)
        print_query_summary(dataset, queries)
        write_manifest_to_path(manifest_file, dataset, queries, dry_run=args.dry_run)


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
        description="Mine explicit query sets for string-filter microbenchmarks"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list",
        help="list datasets and query classes",
    )
    list_parser.set_defaults(func=handle_list)

    mine_parser = subparsers.add_parser(
        "mine",
        help="run prep plus query mining",
    )
    add_dataset_selection(mine_parser)
    mine_parser.add_argument("--max-rows", type=int, default=1_000_000)
    mine_parser.add_argument("--sample-size", type=int, default=50_000)
    mine_parser.add_argument("--queries-per-category", type=int, default=5)
    mine_parser.add_argument("--skip-prep", action="store_true")
    mine_parser.add_argument("--debug", dest="release", action="store_false")
    mine_parser.add_argument("--dry-run", action="store_true")
    mine_parser.set_defaults(release=True, func=handle_mine)

    describe_parser = subparsers.add_parser(
        "describe",
        help="show explicit queries and microbenchmark manifests",
    )
    add_dataset_selection(describe_parser)
    describe_parser.add_argument("--mine-if-missing", action="store_true")
    describe_parser.add_argument("--max-rows", type=int, default=1_000_000)
    describe_parser.add_argument("--sample-size", type=int, default=50_000)
    describe_parser.add_argument("--queries-per-category", type=int, default=5)
    describe_parser.add_argument("--debug", dest="release", action="store_false")
    describe_parser.add_argument("--dry-run", action="store_true")
    describe_parser.set_defaults(release=True, func=handle_describe)

    sql_parser = subparsers.add_parser(
        "sql",
        help="write a one-query workload from a simple COUNT(*) ... LIKE SQL statement",
    )
    add_dataset_selection(sql_parser)
    sql_parser.add_argument(
        "--sql",
        required=True,
        help='SQL such as: SELECT COUNT(*) FROM hits WHERE "URL" LIKE \'%google%\';',
    )
    sql_parser.add_argument(
        "--name",
        default="custom",
        help="suffix for the generated query file name",
    )
    sql_parser.add_argument("--max-rows", type=int, default=1_000_000)
    sql_parser.add_argument("--skip-prep", action="store_true")
    sql_parser.add_argument("--debug", dest="release", action="store_false")
    sql_parser.add_argument("--dry-run", action="store_true")
    sql_parser.set_defaults(release=True, func=handle_sql)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
