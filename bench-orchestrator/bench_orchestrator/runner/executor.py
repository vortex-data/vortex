# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Benchmark binary execution."""

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import final

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import Benchmark, ExecutionBackend, Format

console = Console()


@final
class BenchmarkExecutor:
    """Executes benchmark binaries and captures output."""

    def __init__(self, binary_path: Path, backend: ExecutionBackend, verbose: bool = False):
        self.binary_path = binary_path
        self.backend = backend
        self.verbose = verbose

    def build_command(
        self,
        benchmark: Benchmark,
        formats: list[Format],
        queries: list[int] | None = None,
        exclude_queries: list[int] | None = None,
        iterations: int = 5,
        options: dict[str, str] | None = None,
        track_memory: bool = False,
        samply: bool = False,
        sample_rate: int | None = None,
        tracing: bool = False,
    ) -> list[str]:
        """Build the command used to execute a benchmark binary."""
        cmd = [
            str(self.binary_path),
            benchmark.value,
            "--display-format",
            "gh-json",
            "--iterations",
            str(iterations),
            "--hide-progress-bar",
        ]

        if self.backend in {ExecutionBackend.DATAFUSION, ExecutionBackend.DUCKDB}:
            cmd.extend(["--formats", ",".join(fmt.value for fmt in formats)])
        if self.backend == ExecutionBackend.DUCKDB:
            cmd.append("--delete-duckdb-database")

        if queries:
            cmd.extend(["--queries", ",".join(map(str, queries))])
        if exclude_queries:
            cmd.extend(["--exclude-queries", ",".join(map(str, exclude_queries))])
        if track_memory:
            cmd.append("--track-memory")
        if tracing:
            cmd.append("--tracing")
        if options:
            for key, value in options.items():
                cmd.extend(["--opt", f"{key}={value}"])

        if samply:
            cmd = ["--"] + cmd
            cmd_prefix = ["samply", "record"]
            if sample_rate:
                cmd = cmd_prefix + ["--rate", str(sample_rate)] + cmd
            else:
                cmd = cmd_prefix + cmd

        if samply and self.backend == ExecutionBackend.DUCKDB:
            # Re-use the same DuckDB instance across runs to keep samply output readable.
            cmd.append("--reuse")

        return cmd

    def run(
        self,
        benchmark: Benchmark,
        formats: list[Format],
        queries: list[int] | None = None,
        exclude_queries: list[int] | None = None,
        iterations: int = 5,
        options: dict[str, str] | None = None,
        track_memory: bool = False,
        samply: bool = False,
        sample_rate: int | None = None,
        tracing: bool = False,
        on_result: Callable[[str], None] | None = None,
    ) -> list[str]:
        """
        Run benchmark and return results as JSON lines.

        Args:
            benchmark: The benchmark suite to run
            formats: Data formats to benchmark
            queries: Specific queries to run (None for all)
            exclude_queries: Queries to skip
            iterations: Number of runs per query
            options: Additional options (e.g., scale_factor)
            track_memory: Enable memory tracking
            on_result: Callback for each result line (for streaming)

        Returns:
            List of JSON lines from the benchmark output
        """
        cmd = self.build_command(
            benchmark=benchmark,
            formats=formats,
            queries=queries,
            exclude_queries=exclude_queries,
            iterations=iterations,
            options=options,
            track_memory=track_memory,
            samply=samply,
            sample_rate=sample_rate,
            tracing=tracing,
        )

        if self.verbose:
            console.print(f"[dim]$ {' '.join(cmd)}[/dim]")

        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            _task = progress.add_task(f"Running {self.backend.value} {benchmark.value}...", total=None)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                if line:
                    results.append(line)
                    if on_result:
                        on_result(line)

            ret_code = process.wait()

            if ret_code != 0:
                stderr = process.stderr.read() if process.stderr else ""
                console.print(f"[red]Benchmark failed with code {process.returncode}[/red]")
                if stderr:
                    console.print(f"[red]{stderr}[/red]")
                raise RuntimeError(f"Benchmark {self.backend.value} {benchmark.value} failed: {stderr}")

        return results
