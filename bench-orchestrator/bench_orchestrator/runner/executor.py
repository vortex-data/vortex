# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Benchmark binary execution."""

import selectors
import subprocess
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import final

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import Benchmark, Engine, Format

console = Console()


@final
class BenchmarkExecutor:
    """Executes benchmark binaries and captures output."""

    def __init__(self, binary_path: Path, backend: Engine, verbose: bool = False):
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
        runner: str | None = None,
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

        if self.backend in {Engine.DATAFUSION, Engine.DUCKDB}:
            cmd.extend(["--formats", ",".join(fmt.value for fmt in formats)])
        if self.backend == Engine.DUCKDB:
            cmd.append("--delete-duckdb-database")

        if queries:
            cmd.extend(["--queries", ",".join(map(str, queries))])
        if exclude_queries:
            cmd.extend(["--exclude-queries", ",".join(map(str, exclude_queries))])
        if track_memory:
            cmd.append("--track-memory")
        if tracing:
            cmd.append("--tracing")
        if runner:
            cmd.extend(["--runner", runner])
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

        if samply and self.backend == Engine.DUCKDB:
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
        runner: str | None = None,
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
            runner=runner,
        )

        if self.verbose:
            console.print(f"[dim]$ {' '.join(cmd)}[/dim]")

        results: list[str] = []
        diagnostic_lines: deque[str] = deque(maxlen=200)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            _task = progress.add_task(f"Running {self.backend.value} {benchmark.value}...", total=None)

            # Merge stderr into stdout so verbose benchmark logs cannot fill a separate pipe and
            # block the child process before it emits JSON results.
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)

            try:
                while selector.get_map():
                    for key, _mask in selector.select(timeout=0.1):
                        line = key.fileobj.readline()
                        if line == "":
                            selector.unregister(key.fileobj)
                            continue

                        line = line.rstrip()
                        if not line:
                            continue

                        if line.startswith("{"):
                            results.append(line)
                            if on_result:
                                on_result(line)
                        else:
                            diagnostic_lines.append(line)
                            console.print(line, markup=False)
            finally:
                selector.close()

            ret_code = process.wait()

            if ret_code != 0:
                console.print(f"[red]Benchmark failed with code {process.returncode}[/red]")
                diagnostics = "\n".join(diagnostic_lines)
                if diagnostics:
                    console.print(f"[red]{diagnostics}[/red]")
                raise RuntimeError(f"Benchmark {self.backend.value} {benchmark.value} failed: {diagnostics}")

        return results
