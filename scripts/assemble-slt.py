#!/usr/bin/env python3
"""Assemble per-engine .slt files from existing per-query .slt.no result files.

Each engine gets its own top-level SLT file that includes per-query SLT files.
Queries with identical results across engines share a single file; queries with
differing results get engine-specific files.

Usage:
    python scripts/assemble-slt.py tpch
    python scripts/assemble-slt.py clickbench
    python scripts/assemble-slt.py all
"""

import sys
from pathlib import Path

ENGINES = ["datafusion", "duckdb"]


def parse_slt_file(path: Path) -> tuple[str, str, list[str]]:
    """Parse a .slt.no file, returning (header_line, sql, result_lines).

    header_line: e.g. "query TTTTTTTTTT rowsort"
    sql: the SQL query text
    result_lines: lines after the ---- separator
    """
    text = path.read_text()
    # Check for include directive
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("include "):
        # Resolve include
        include_path = path.parent / lines[0].split(" ", 1)[1].strip()
        return parse_slt_file(include_path.resolve())

    parts = text.split("----\n", 1)
    if len(parts) != 2:
        raise ValueError(f"No ---- separator in {path}")

    header_and_sql = parts[0]
    results_text = parts[1].rstrip("\n")

    header_lines = header_and_sql.strip().splitlines()
    header_line = header_lines[0]  # e.g. "query TTTTTTTTTT rowsort"
    sql = "\n".join(header_lines[1:])
    result_lines = results_text.splitlines() if results_text else []

    return header_line, sql, result_lines


def format_query_slt(
    header: str, sql: str, result_lines: list[str], qi: int
) -> str:
    """Format a single query as SLT content with a bench_N label."""
    parts = header.split()
    col_types = parts[1] if len(parts) > 1 else "T"
    sort_mode = parts[2] if len(parts) > 2 else "rowsort"

    lines = []
    lines.append(f"query {col_types} {sort_mode} bench_{qi}")
    lines.append(sql)
    lines.append("----")
    lines.extend(result_lines)
    lines.append("")
    return "\n".join(lines) + "\n"


def assemble_tpch(bench_dir: Path):
    results_dir = bench_dir / "tpch" / "slt" / "results"
    queries_dir = bench_dir / "tpch"
    slt_dir = bench_dir / "tpch" / "slt"
    per_query_dir = slt_dir / "queries"
    per_query_dir.mkdir(parents=True, exist_ok=True)

    # Track which include file each engine should use per query
    engine_includes: dict[str, list[str]] = {e: [] for e in ENGINES}

    for qi in range(1, 23):
        qname = f"q{qi:02d}"

        # Read SQL from canonical query file
        sql_path = queries_dir / f"q{qi}.sql"
        sql = sql_path.read_text().rstrip("\n")

        # Parse engine-specific results
        engine_data: dict[str, tuple[str, list[str]]] = {}
        for engine in ENGINES:
            engine_path = results_dir / engine / f"{qname}.slt.no"
            if engine_path.exists():
                header, _sql, result_lines = parse_slt_file(engine_path)
                engine_data[engine] = (header, result_lines)
            else:
                print(
                    f"WARNING: no {engine} file for {qname}, skipping",
                    file=sys.stderr,
                )

        if not engine_data:
            continue

        # Check if all engines have identical content
        values = list(engine_data.values())
        all_same = len(engine_data) == len(ENGINES) and all(
            v == values[0] for v in values[1:]
        )

        if all_same:
            # Write shared file
            header, result_lines = values[0]
            content = format_query_slt(header, sql, result_lines, qi)
            out_path = per_query_dir / f"bench_{qi:02d}.slt"
            out_path.write_text(content)
            for engine in ENGINES:
                engine_includes[engine].append(f"queries/bench_{qi:02d}.slt")
        else:
            # Write engine-specific files
            for engine in ENGINES:
                if engine in engine_data:
                    header, result_lines = engine_data[engine]
                    content = format_query_slt(header, sql, result_lines, qi)
                    out_path = per_query_dir / f"bench_{qi:02d}_{engine}.slt"
                    out_path.write_text(content)
                    engine_includes[engine].append(
                        f"queries/bench_{qi:02d}_{engine}.slt"
                    )

    # Write top-level include files
    for engine in ENGINES:
        lines = []
        lines.append(
            f"# TPC-H benchmark queries — {engine} expected results."
        )
        lines.append("")
        for inc in engine_includes[engine]:
            lines.append(f"include {inc}")
        lines.append("")

        output_path = slt_dir / f"tpch_{engine}.slt"
        output_path.write_text("\n".join(lines))
        print(f"Wrote {output_path} ({len(engine_includes[engine])} queries)")


def assemble_clickbench(bench_dir: Path):
    results_dir = bench_dir / "clickbench" / "slt" / "results"
    slt_dir = bench_dir / "clickbench" / "slt"
    slt_dir.mkdir(parents=True, exist_ok=True)
    per_query_dir = slt_dir / "queries"
    per_query_dir.mkdir(parents=True, exist_ok=True)

    # Track which include file each engine should use per query
    engine_includes: dict[str, list[str]] = {e: [] for e in ENGINES}

    for qi in range(0, 43):
        qname = f"q{qi:02d}"

        # Parse engine-specific results (ClickBench has SQL embedded in .slt.no)
        engine_data: dict[str, tuple[str, str, list[str]]] = {}
        for engine in ENGINES:
            engine_path = results_dir / engine / f"{qname}.slt.no"
            if engine_path.exists():
                header, sql, result_lines = parse_slt_file(engine_path)
                engine_data[engine] = (header, sql, result_lines)

        if not engine_data:
            print(
                f"WARNING: no files for {qname}, skipping", file=sys.stderr
            )
            continue

        # Check if all engines have identical content
        values = list(engine_data.values())
        all_same = len(engine_data) == len(ENGINES) and all(
            v == values[0] for v in values[1:]
        )

        if all_same:
            # Write shared file
            header, sql, result_lines = values[0]
            content = format_query_slt(header, sql, result_lines, qi)
            out_path = per_query_dir / f"bench_{qi:02d}.slt"
            out_path.write_text(content)
            for engine in ENGINES:
                engine_includes[engine].append(f"queries/bench_{qi:02d}.slt")
        elif len(engine_data) == 1:
            # Only one engine has this query
            engine = list(engine_data.keys())[0]
            header, sql, result_lines = engine_data[engine]
            content = format_query_slt(header, sql, result_lines, qi)
            out_path = per_query_dir / f"bench_{qi:02d}_{engine}.slt"
            out_path.write_text(content)
            engine_includes[engine].append(
                f"queries/bench_{qi:02d}_{engine}.slt"
            )
        else:
            # Write engine-specific files
            for engine in ENGINES:
                if engine in engine_data:
                    header, sql, result_lines = engine_data[engine]
                    content = format_query_slt(header, sql, result_lines, qi)
                    out_path = per_query_dir / f"bench_{qi:02d}_{engine}.slt"
                    out_path.write_text(content)
                    engine_includes[engine].append(
                        f"queries/bench_{qi:02d}_{engine}.slt"
                    )

    # Write top-level include files
    for engine in ENGINES:
        lines = []
        lines.append(
            f"# ClickBench benchmark queries — {engine} expected results."
        )
        lines.append("")
        for inc in engine_includes[engine]:
            lines.append(f"include {inc}")
        lines.append("")

        output_path = slt_dir / f"clickbench_{engine}.slt"
        output_path.write_text("\n".join(lines))
        print(f"Wrote {output_path} ({len(engine_includes[engine])} queries)")


def main():
    if len(sys.argv) < 2:
        print("Usage: assemble-slt.py [tpch|clickbench|all]", file=sys.stderr)
        sys.exit(1)

    bench_dir = Path(__file__).resolve().parent.parent / "vortex-bench"

    target = sys.argv[1]
    if target in ("tpch", "all"):
        assemble_tpch(bench_dir)
    if target in ("clickbench", "all"):
        assemble_clickbench(bench_dir)


if __name__ == "__main__":
    main()
