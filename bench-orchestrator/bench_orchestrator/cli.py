# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""CLI for benchmark orchestration."""

from datetime import datetime, timedelta
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .comparison.analyzer import BenchmarkAnalyzer, TargetRef
from .comparison.reporter import BenchmarkReporter
from .config import (
    ENGINE_FORMATS,
    Benchmark,
    BuildConfig,
    Engine,
    Format,
    RunConfig,
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


def parse_queries(value: str | None) -> list[int] | None:
    """Parse comma-separated query numbers."""
    if not value:
        return None
    return [int(q.strip()) for q in value.split(",")]


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
    scale_factor: Annotated[
        str | None, typer.Option("--scale-factor", "-s", help="Scale factor for TPC benchmarks")
    ] = None,
    track_memory: Annotated[bool, typer.Option("--track-memory", help="Track memory usage")] = False,
    build: Annotated[bool, typer.Option("--build/--no-build", help="Build binaries before running")] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log underlying commands")] = False,
) -> None:
    """Run benchmarks with specified configuration."""
    engines = parse_engines(engine)
    formats = parse_formats(format)
    query_list = parse_queries(queries)
    exclude_list = parse_queries(exclude_queries)

    # Build options dict
    options: dict[str, str] = {}
    if scale_factor:
        options["scale_factor"] = scale_factor

    config = RunConfig(
        benchmark=benchmark,
        engines=engines,
        formats=formats,
        queries=query_list,
        exclude_queries=exclude_list,
        iterations=iterations,
        label=label,
        options=options,
        track_memory=track_memory,
    )

    # Validate configuration
    warnings = config.validate()
    for warning in warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")

    build_config = BuildConfig()
    builder = BenchmarkBuilder(config=build_config, verbose=verbose)
    store = ResultStore()

    # Build binaries if needed
    if build:
        binary_paths = builder.build(engines)
    else:
        binary_paths = {e: builder.get_binary_path(e) for e in engines}
        # Check binaries exist
        for eng, path in binary_paths.items():
            if not path.exists():
                console.print(f"[red]Binary not found: {path}[/red]")
                console.print("Run with --build to build binaries first")
                raise typer.Exit(1)

    console.print(f"\n[bold]Running {benchmark.value} benchmark[/bold]")
    console.print(f"  Engines: {', '.join(e.value for e in engines)}")
    console.print(f"  Formats: {', '.join(f.value for f in formats)}")
    console.print(f"  Iterations: {iterations}")
    if label:
        console.print(f"  Label: {label}")
    console.print()

    # Run benchmarks and store results
    with store.create_run(config, build_config) as ctx:
        for eng in engines:
            # Filter formats to those supported by this engine
            supported_formats = ENGINE_FORMATS.get(eng, [])
            engine_formats = [f for f in formats if f in supported_formats]

            if not engine_formats:
                console.print(f"[yellow]Skipping {eng.value}: no supported formats[/yellow]")
                continue

            executor = BenchmarkExecutor(binary_paths[eng], eng, verbose=verbose)

            try:
                results = executor.run(
                    benchmark=benchmark,
                    formats=engine_formats,
                    queries=query_list,
                    exclude_queries=exclude_list,
                    iterations=iterations,
                    options=options,
                    track_memory=track_memory,
                    on_result=ctx.write_raw_json,
                )
                console.print(f"[green]{eng.value}: {len(results)} results[/green]")
            except RuntimeError as e:
                console.print(f"[red]{eng.value} failed: {e}[/red]")

        # Update metadata with binary paths
        ctx.metadata.binaries = {e.value: str(p) for e, p in binary_paths.items()}

    console.print(f"\n[green]Results saved to run: {ctx.metadata.run_id}[/green]")


@app.command()
def compare(
    base: Annotated[
        str | None,
        typer.Option("--base", "-b", help="Base reference (engine:format@run)"),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option("--target", "-t", help="Target reference (engine:format@run)"),
    ] = None,
    runs: Annotated[
        str | None,
        typer.Option("--runs", "-r", help="Two runs to compare (comma-separated)"),
    ] = None,
    threshold: Annotated[float, typer.Option("--threshold", help="Significance threshold (default 10%)")] = 0.10,
) -> None:
    """Compare benchmark results."""
    store = ResultStore()

    if runs:
        # Compare two full runs
        run_refs = [r.strip() for r in runs.split(",")]
        if len(run_refs) != 2:
            console.print("[red]--runs requires exactly two run references[/red]")
            raise typer.Exit(1)

        base_run = store.get_run(run_refs[0])
        target_run = store.get_run(run_refs[1])

        if not base_run:
            console.print(f"[red]Run not found: {run_refs[0]}[/red]")
            raise typer.Exit(1)
        if not target_run:
            console.print(f"[red]Run not found: {run_refs[1]}[/red]")
            raise typer.Exit(1)

        base_df = store.load_results(base_run.run_id)
        target_df = store.load_results(target_run.run_id)

        base_label = base_run.label or base_run.run_id[:20]
        target_label = target_run.label or target_run.run_id[:20]

    elif base and target:
        # Compare specific configurations
        base_ref = TargetRef.parse(base)
        target_ref = TargetRef.parse(target)

        base_run = store.get_run(base_ref.run)
        target_run = store.get_run(target_ref.run)

        if not base_run:
            console.print(f"[red]Run not found: {base_ref.run}[/red]")
            raise typer.Exit(1)
        if not target_run:
            console.print(f"[red]Run not found: {target_ref.run}[/red]")
            raise typer.Exit(1)

        base_df = store.load_results(base_run.run_id)
        target_df = store.load_results(target_run.run_id)

        # Apply filters
        base_analyzer = BenchmarkAnalyzer(base_df)
        target_analyzer = BenchmarkAnalyzer(target_df)

        base_df = base_analyzer.filter_by_ref(base_ref)
        target_df = target_analyzer.filter_by_ref(target_ref)

        base_label = base
        target_label = target

    else:
        console.print("[red]Must specify either --runs or --base/--target[/red]")
        raise typer.Exit(1)

    if base_df.empty:
        console.print("[red]No results found for base[/red]")
        raise typer.Exit(1)
    if target_df.empty:
        console.print("[red]No results found for target[/red]")
        raise typer.Exit(1)

    # Perform comparison
    analyzer = BenchmarkAnalyzer(base_df)
    comparison = analyzer.compare(base_df, target_df)
    stats = analyzer.summary_stats(comparison)

    reporter = BenchmarkReporter(comparison, stats, threshold)

    table = reporter.to_rich_table(
        title="Benchmark Comparison",
        base_label=base_label,
        target_label=target_label,
    )
    console.print(table)
    reporter.print_summary()


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
    run_ref: Annotated[str, typer.Argument(help="Run ID, label, or 'latest'")],
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

    console.print(f"[bold]Building: {', '.join(e.value for e in engines)}[/bold]")
    console.print(f"  Profile: {builder.config.profile}")
    console.print(f"  RUSTFLAGS: {builder.config.rustflags}")
    console.print()

    try:
        paths = builder.build(engines)
        console.print("\n[green]Build complete:[/green]")
        for eng, path in paths.items():
            console.print(f"  {eng.value}: {path}")
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
