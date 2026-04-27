# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""CLI for benchmark orchestration."""

import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .comparison import analyzer
from .comparison.reporter import pivot_comparison_table
from .config import (
    Benchmark,
    BenchmarkTarget,
    BuildConfig,
    Engine,
    Format,
    RunConfig,
    group_targets_by_backend,
    parse_formats_json,
    parse_targets_json,
    resolve_axis_targets,
)
from .runner.builder import BenchmarkBuilder
from .runner.executor import BenchmarkExecutor
from .storage.store import ResultStore

app = typer.Typer(
    name="vortex-bench",
    help="Benchmark orchestration tool for Vortex",
    no_args_is_help=True,
)
console = Console()


def parse_engines(value: str) -> list[Engine]:
    """Parse comma-separated engine names."""
    return [Engine(e.strip()) for e in value.split(",")]


def parse_formats(value: str) -> list[Format]:
    """Parse comma-separated format names."""
    return [Format(f.strip()) for f in value.split(",")]


def parse_options(value: list[str] | None) -> dict[str, str]:
    """Parse repeated --opt key=value options into a dictionary."""
    parsed: dict[str, str] = {}
    if not value:
        return parsed

    for opt in value:
        for segment in opt.split(","):
            key, raw_value, *rest = segment.split("=")
            if rest:
                raise ValueError("Options must be key-value pairs separated by =")
            parsed[key] = raw_value
    return parsed


def parse_queries(value: str | None) -> list[int] | None:
    """Parse comma-separated query numbers."""
    if not value:
        return None

    result = set()
    for part in value.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            result.update(range(int(start), int(end) + 1))
        else:
            result.add(int(part))
    return sorted(result)


def run_ref_auto_complete() -> list[str]:
    return list(map(lambda x: x.run_id, ResultStore().list_runs(limit=None)))


def targets_from_axes(engine: str, format: str) -> tuple[list[BenchmarkTarget], list[str]]:
    """Resolve legacy engine/format axes into explicit benchmark targets."""
    return resolve_axis_targets(parse_engines(engine), parse_formats(format))


def backends_for_engines(engines: list[Engine]) -> list[Engine]:
    """Resolve legacy engine selections to unique execution engines."""
    seed_formats = {
        Engine.DATAFUSION: Format.PARQUET,
        Engine.DUCKDB: Format.PARQUET,
        Engine.LANCE: Format.LANCE,
    }
    return list(
        group_targets_by_backend(BenchmarkTarget(engine=engine, format=seed_formats[engine]) for engine in engines)
    )


@contextmanager
def open_results_output(path: Path | None):
    """Open an optional compatibility JSONL output file."""
    if path is None:
        yield None
        return

    if path.parent != Path():
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        yield handle


def write_result_line(line: str, store_writer, compatibility_file) -> None:
    """Write a raw result line to the run store and optional compatibility output."""
    store_writer(line)
    if compatibility_file is None:
        return

    line = line.strip()
    if line.startswith("{"):
        compatibility_file.write(line + "\n")
        compatibility_file.flush()


@app.command("prepare-data")
def prepare_data(
    benchmark: Annotated[Benchmark, typer.Argument(help="Benchmark suite to prepare data for")],
    format: Annotated[
        str | None,
        typer.Option("--format", "-f", help="Formats to generate (comma-separated)"),
    ] = None,
    formats_json: Annotated[
        str | None,
        typer.Option("--formats-json", help="Exact data formats to generate as a JSON array"),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log underlying commands")] = False,
    options: Annotated[list[str] | None, typer.Option("--opt", help="Benchmark-specific options")] = None,
) -> None:
    """Generate benchmark data for explicitly requested formats."""
    if format and formats_json:
        console.print("[red]Specify only one of --format or --formats-json[/red]")
        raise typer.Exit(1)
    if not format and not formats_json:
        console.print("[red]Must specify one of --format or --formats-json[/red]")
        raise typer.Exit(1)

    try:
        formats = parse_formats_json(formats_json) if formats_json else parse_formats(format or "")
        bench_opts = parse_options(options)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    builder = BenchmarkBuilder(verbose=verbose)
    binary_path = builder.get_data_generator_path()
    if not binary_path.exists():
        console.print(f"[red]Binary not found: {binary_path}[/red]")
        console.print("Build benchmark binaries before running prepare-data")
        raise typer.Exit(1)

    cmd = [str(binary_path), benchmark.value, "--formats", ",".join(fmt.value for fmt in formats)]
    if verbose:
        cmd.append("--verbose")
    for key, value in bench_opts.items():
        cmd.extend(["--opt", f"{key}={value}"])

    if verbose:
        console.print(f"[dim]$ {' '.join(cmd)}[/dim]")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Data generation failed: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def run(
    benchmark: Annotated[Benchmark, typer.Argument(help="Benchmark suite to run")],
    engine: Annotated[
        str, typer.Option("--engine", "-e", help="Engines to benchmark (comma-separated)")
    ] = "datafusion,duckdb",
    format: Annotated[
        str, typer.Option("--format", "-f", help="Formats to benchmark (comma-separated)")
    ] = "parquet,vortex",
    queries: Annotated[str | None, typer.Option("--queries", "-q", help="Specific queries to run")] = None,
    exclude_queries: Annotated[str | None, typer.Option("--exclude-queries", help="Queries to skip")] = None,
    iterations: Annotated[int, typer.Option("--iterations", "-i", help="Iterations per query")] = 5,
    label: Annotated[str | None, typer.Option("--label", "-l", help="Label for this run")] = None,
    track_memory: Annotated[bool, typer.Option("--track-memory", help="Track memory usage")] = False,
    samply: Annotated[bool, typer.Option("--samply", help="Record a profile using samply")] = False,
    sample_rate: Annotated[int, typer.Option("--sample-rate", help="Sample rate to run samply with")] = None,
    tracing: Annotated[bool, typer.Option("--tracing", help="Record a trace for use with perfetto")] = False,
    build: Annotated[bool, typer.Option("--build/--no-build", help="Build binaries before running")] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log underlying commands")] = False,
    targets_json: Annotated[
        str | None,
        typer.Option("--targets-json", help="Exact benchmark targets as a JSON array"),
    ] = None,
    runner: Annotated[
        str | None,
        typer.Option("--runner", help="Benchmark runner ID (e.g., ec2_c6id.8xlarge)"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional path for compatibility JSONL output"),
    ] = None,
    options: Annotated[list[str] | None, typer.Option("--opt", help="Engine or benchmark specific options")] = None,
) -> None:
    """Run benchmarks with specified configuration."""
    query_list = parse_queries(queries)
    exclude_list = parse_queries(exclude_queries)
    strict_failures = targets_json is not None

    try:
        bench_opts = parse_options(options)
        if targets_json:
            targets = parse_targets_json(targets_json)
            warnings: list[str] = []
        else:
            targets, warnings = targets_from_axes(engine, format)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    for warning in warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")
    if not targets:
        console.print("[red]No valid benchmark targets selected[/red]")
        raise typer.Exit(1)

    config = RunConfig(
        benchmark=benchmark,
        targets=targets,
        queries=query_list,
        exclude_queries=exclude_list,
        iterations=iterations,
        label=label,
        options=bench_opts,
        track_memory=track_memory,
    )

    errors = config.validate()
    if errors:
        for error in errors:
            console.print(f"[red]{error}[/red]")
        raise typer.Exit(1)

    build_config = BuildConfig()
    builder = BenchmarkBuilder(config=build_config, verbose=verbose)
    store = ResultStore()

    if build:
        binary_paths = builder.build(config.backends)
    else:
        binary_paths = {backend: builder.get_binary_path(backend) for backend in config.backends}
        for backend, path in binary_paths.items():
            if not path.exists():
                console.print(f"[red]Binary not found: {path}[/red]")
                console.print("Run with --build to build binaries first")
                raise typer.Exit(1)

    console.print(f"\n[bold]Running {benchmark.value} benchmark[/bold]")
    console.print(f"  Targets: {', '.join(map(str, config.targets))}")
    console.print(f"  Iterations: {iterations}")
    if label:
        console.print(f"  Label: {label}")
    console.print()

    backend_groups = group_targets_by_backend(config.targets)
    soft_failures: list[str] = []

    try:
        with store.create_run(config, build_config) as ctx, open_results_output(output) as compatibility_file:
            for backend, backend_targets in backend_groups.items():
                executor = BenchmarkExecutor(binary_paths[backend], backend, verbose=verbose)
                backend_formats = [target.format for target in backend_targets]

                try:
                    results = executor.run(
                        benchmark=benchmark,
                        formats=backend_formats,
                        queries=query_list,
                        exclude_queries=exclude_list,
                        iterations=iterations,
                        options=bench_opts,
                        track_memory=track_memory,
                        samply=samply,
                        sample_rate=sample_rate,
                        tracing=tracing,
                        runner=runner,
                        on_result=lambda line, store_writer=ctx.write_raw_json, compatibility=compatibility_file: (
                            write_result_line(
                                line,
                                store_writer,
                                compatibility,
                            )
                        ),
                    )
                    console.print(f"[green]{backend.value}: {len(results)} results[/green]")
                except RuntimeError as exc:
                    ctx.metadata.partial = True
                    if strict_failures:
                        raise
                    console.print(f"[red]{backend.value} failed: {exc}[/red]")
                    soft_failures.append(str(exc))

            ctx.metadata.binaries = {backend.value: str(path) for backend, path in binary_paths.items()}
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if soft_failures:
        console.print(f"[yellow]Completed with {len(soft_failures)} backend failure(s)[/yellow]")

    metadata = ctx.metadata
    console.print(f"\n[green]Results saved to run: {metadata.run_id}[/green]")

    # Show comparison table if we have multiple engine:format combinations
    df = store.load_results(metadata.run_id)
    if not df.empty:
        try:
            pivot = analyzer.compare_within_run(df)
            table = pivot_comparison_table(pivot)
            console.print()
            console.print(table)
        except ValueError:
            # Not enough combinations to compare
            pass

    # If tracing was enabled, start a localhost server to serve the trace file (./trace.json) and open the
    # perfetto UI in the browser
    if tracing:
        import http.server
        import socketserver
        import threading
        import webbrowser

        # This is the only localhost port allowed by Perfetto's CSP.
        HOST = "127.0.0.1"
        PORT = 9001

        class TraceRequestHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/trace.json":
                    self.path = "trace.json"
                return super().do_GET()

            def do_POST(self):
                self.send_error(404, "File not found")

            def end_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                super().end_headers()

        def start_server():
            socketserver.TCPServer.allow_reuse_address = True
            with socketserver.TCPServer(("", PORT), TraceRequestHandler) as httpd:
                console.print(f"[green]Serving trace on http://{HOST}:{PORT}/trace.json[/green]")
                httpd.serve_forever()

        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        webbrowser.open_new_tab(f"http://ui.perfetto.dev/#!/?url=http://{HOST}:{PORT}/trace.json")
        server_thread.join()


@app.command()
def compare(
    runs: Annotated[
        str | None,
        typer.Option("--runs", "-r", help="Runs to compare (comma-separated, 2 or more)"),
    ] = None,
    run: Annotated[
        str | None,
        typer.Option("--run", help="Single run for within-run comparison", autocompletion=run_ref_auto_complete),
    ] = None,
    baseline: Annotated[
        str | None,
        typer.Option("--baseline", help="Baseline engine:format for within-run comparison"),
    ] = None,
    threshold: Annotated[float, typer.Option("--threshold", help="Significance threshold (default 10%)")] = 0.10,
    filter_engine: Annotated[
        str | None, typer.Option("--engine", help="Filter only for results that use a specific engine")
    ] = None,
    filter_format: Annotated[
        str | None, typer.Option("--format", help="Filter only for results that use a specific file format")
    ] = None,
) -> None:
    """Compare benchmark results."""
    store = ResultStore()

    if run:
        # Within-run comparison
        run_meta = store.get_run(run)
        if not run_meta:
            console.print(f"[red]Run not found: {run}[/red]")
            raise typer.Exit(1)

        df = store.load_results(run_meta.run_id)

        if df.empty:
            console.print("[red]No results found[/red]")
            raise typer.Exit(1)

        # Parse baseline if provided
        baseline_engine = None
        baseline_format = None
        if baseline:
            if ":" in baseline:
                baseline_engine, baseline_format = baseline.split(":", 1)
            else:
                console.print("[red]--baseline must be engine:format[/red]")
                raise typer.Exit(1)

        try:
            pivot = analyzer.compare_within_run(df, baseline_engine, baseline_format, filter_engine, filter_format)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

        table = pivot_comparison_table(pivot, threshold)
        console.print(table)
        return

    elif runs:
        # Compare multiple runs (2 or more)
        run_refs = [r.strip() for r in runs.split(",")]
        if len(run_refs) < 2:
            console.print("[red]--runs requires at least two run references[/red]")
            raise typer.Exit(1)

        # Load all runs
        run_data: list[tuple[str, pd.DataFrame]] = []
        for ref in run_refs:
            run_meta = store.get_run(ref)
            if not run_meta:
                console.print(f"[red]Run not found: {ref}[/red]")
                raise typer.Exit(1)
            label = run_meta.label or run_meta.run_id[:16]
            df = store.load_results(run_meta.run_id)
            if df.empty:
                console.print(f"[red]No results for run: {ref}[/red]")
                raise typer.Exit(1)
            run_data.append((label, df))

        # Use baseline option if provided, otherwise first run
        baseline_label = None
        if baseline:
            # Find matching label
            for label, _ in run_data:
                if baseline in label:
                    baseline_label = label
                    break
            if baseline_label is None:
                console.print(f"[red]Baseline not found: {baseline}[/red]")
                raise typer.Exit(1)

        try:
            pivot = analyzer.compare_runs(run_data, baseline_label, filter_engine, filter_format)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

        table = pivot_comparison_table(pivot, threshold, row_keys=["query", "engine", "format"])
        console.print(table)
        return

    else:
        console.print("[red]Must specify either --runs or --run[/red]")
        raise typer.Exit(1)


@app.command("list")
def list_runs(
    benchmark: Annotated[str | None, typer.Option("--benchmark", "-b", help="Filter by benchmark")] = None,
    since: Annotated[str | None, typer.Option("--since", help="Time filter (e.g., '7 days')")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum runs to show")] = 20,
) -> None:
    """List past benchmark runs."""
    store = ResultStore()

    # Parse time filter
    since_dt = None
    if since:
        # Simple parsing for common formats
        if "day" in since:
            days = int(since.split()[0])
            since_dt = datetime.now() - timedelta(days=days)
        elif "week" in since:
            weeks = int(since.split()[0])
            since_dt = datetime.now() - timedelta(weeks=weeks)
        elif "hour" in since:
            hours = int(since.split()[0])
            since_dt = datetime.now() - timedelta(hours=hours)

    runs = store.list_runs(benchmark=benchmark, since=since_dt, limit=limit)

    if not runs:
        console.print("[yellow]No runs found[/yellow]")
        return

    table = Table(title="Benchmark Runs")
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Label", style="green")
    table.add_column("Benchmark")
    table.add_column("Engines")
    table.add_column("Results", justify="right")
    table.add_column("Status")

    for run in runs:
        status = "[yellow]partial[/yellow]" if run.partial else "[green]complete[/green]"
        table.add_row(
            run.run_id,
            run.label or "-",
            run.benchmark,
            ", ".join(run.engines),
            str(run.result_count),
            status,
        )

    console.print(table)


@app.command()
def show(
    run_ref: Annotated[str, typer.Argument(help="Run ID, label, or 'latest'", autocompletion=run_ref_auto_complete)],
) -> None:
    """Show details of a specific run."""
    store = ResultStore()
    metadata = store.get_run(run_ref)

    if not metadata:
        console.print(f"[red]Run not found: {run_ref}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Run: {metadata.run_id}[/bold]")
    if metadata.label:
        console.print(f"  Label: [green]{metadata.label}[/green]")
    console.print(f"  Timestamp: {metadata.timestamp}")
    console.print(f"  Benchmark: {metadata.benchmark}")
    console.print(f"  Engines: {', '.join(metadata.engines)}")
    console.print(f"  Formats: {', '.join(metadata.formats)}")
    if metadata.targets:
        console.print(
            "  Targets: " + ", ".join(f"{target['engine']}:{target['format']}" for target in metadata.targets)
        )
    console.print(f"  Iterations: {metadata.iterations}")
    console.print(f"  Git commit: {metadata.git_commit[:8]}")
    console.print(f"  Git branch: {metadata.git_branch}")
    if metadata.git_dirty:
        console.print("  [yellow]Working directory was dirty[/yellow]")
    console.print(f"  Profile: {metadata.profile}")
    console.print(f"  RUSTFLAGS: {metadata.rustflags}")

    if metadata.dataset_config:
        console.print(f"  Options: {metadata.dataset_config}")

    status = "[yellow]partial[/yellow]" if metadata.partial else "[green]complete[/green]"
    console.print(f"  Status: {status}")

    if metadata.completed_at:
        duration = metadata.completed_at - metadata.timestamp
        console.print(f"  Duration: {duration}")

    # Load and summarize results
    results_df = store.load_results(metadata.run_id)
    if not results_df.empty:
        console.print(f"\n  Results: {len(results_df)} measurements")


@app.command()
def build(
    engine: Annotated[
        str | None,
        typer.Option("--engine", "-e", help="Engines to build (comma-separated)"),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log underlying commands")] = False,
) -> None:
    """Build benchmark binaries."""
    builder = BenchmarkBuilder(verbose=verbose)

    if engine:
        engines = parse_engines(engine)
    else:
        engines = list(Engine)
    backends = backends_for_engines(engines)

    console.print(f"[bold]Building: {', '.join(e.value for e in engines)}[/bold]")
    console.print(f"  Profile: {builder.config.profile}")
    console.print(f"  RUSTFLAGS: {builder.config.rustflags}")
    console.print()

    try:
        paths = builder.build(backends)
        console.print("\n[green]Build complete:[/green]")
        for backend, path in paths.items():
            console.print(f"  {backend.value}: {path}")
    except RuntimeError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def clean(
    older_than: Annotated[
        str | None,
        typer.Option("--older-than", help="Delete runs older than (e.g., '30 days')"),
    ] = None,
    keep_labeled: Annotated[bool, typer.Option("--keep-labeled", help="Don't delete labeled runs")] = True,
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n", help="Show what would be deleted")] = False,
) -> None:
    """Clean old benchmark results."""
    store = ResultStore()

    # Parse time filter
    cutoff = None
    if older_than:
        if "day" in older_than:
            days = int(older_than.split()[0])
            cutoff = datetime.now() - timedelta(days=days)
        elif "week" in older_than:
            weeks = int(older_than.split()[0])
            cutoff = datetime.now() - timedelta(weeks=weeks)

    if not cutoff:
        console.print("[red]Must specify --older-than[/red]")
        raise typer.Exit(1)

    runs = store.list_runs(limit=1000)
    to_delete = []

    for run in runs:
        if run.timestamp < cutoff:
            if keep_labeled and run.label:
                continue
            to_delete.append(run)

    if not to_delete:
        console.print("[green]No runs to delete[/green]")
        return

    console.print(f"[yellow]Found {len(to_delete)} runs to delete:[/yellow]")
    for run in to_delete:
        console.print(f"  {run.run_id}")

    if dry_run:
        console.print("\n[yellow]Dry run - no changes made[/yellow]")
        return

    if not typer.confirm("Delete these runs?"):
        console.print("Cancelled")
        return

    for run in to_delete:
        store.delete_run(run.run_id)
        console.print(f"[red]Deleted: {run.run_id}[/red]")

    console.print(f"\n[green]Deleted {len(to_delete)} runs[/green]")


if __name__ == "__main__":
    app()
