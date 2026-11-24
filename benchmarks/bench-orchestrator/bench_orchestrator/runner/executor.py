# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Benchmark binary execution."""

import selectors
import subprocess
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import final

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import Benchmark, Engine, Format, Integration, ScanEngine, default_scan_engine

console = Console()

# Arguments each backend's binary takes to run an integration. A DuckDB datasource target needs an
# entry here once DuckDB has a `DataSource` table function.
INTEGRATION_ARGS: dict[tuple[Engine, Integration], list[str]] = {
    (Engine.DATAFUSION, Integration.FILE): [],
    (Engine.DUCKDB, Integration.FILE): [],
    (Engine.DATAFUSION, Integration.DATASOURCE): ["--scan-api", "true"],
}


@final
class BenchmarkExecutor:
    """Executes benchmark binaries and captures output."""

    def __init__(self, binary_path: Path, backend: Engine, verbose: bool = False) -> None:
        self.binary_path = binary_path
        self.backend = backend
        self.verbose = verbose

    def build_command(
        self,
        benchmark: Benchmark,
        formats: list[Format],
        query: int | None = None,
        iterations: int = 5,
        options: dict[str, str] | None = None,
        track_memory: bool = False,
        samply: bool = False,
        sample_rate: int | None = None,
        tracing: bool = False,
        runner: str | None = None,
        ingest_output: Path | None = None,
        integration: Integration = Integration.FILE,
        scan: ScanEngine | None = None,
    ) -> list[str]:
        """Build the command used to execute a benchmark binary.

        `scan` selects the Vortex scan engine; without one the binary runs the suite's default.
        """
        self._check_integration(integration)
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
            scan = scan if scan is not None else default_scan_engine(benchmark)
            cmd.extend(["--scan-v1", str(scan is ScanEngine.V1).lower()])
            cmd.extend(INTEGRATION_ARGS.get((self.backend, integration), []))
        if self.backend == Engine.DUCKDB:
            cmd.append("--delete-duckdb-database")

        if query is not None:
            cmd.extend(["--queries", str(query)])
        if track_memory:
            cmd.append("--track-memory")
        if tracing:
            cmd.append("--tracing")
        if runner:
            cmd.extend(["--runner", runner])
        if ingest_output is not None:
            cmd.extend(["--ingest-jsonl", str(ingest_output)])
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

    def _check_integration(self, integration: Integration) -> None:
        if self.backend is not Engine.LANCE and (self.backend, integration) not in INTEGRATION_ARGS:
            raise ValueError(f"{self.backend.value} cannot run the {integration.value} integration")

    def write_data(self, benchmark: Benchmark, options: dict[str, str] | None = None) -> None:
        """Write the data this backend's queries read, without running a query.

        Only `lance-bench` writes its own data: it converts the base Parquet files to Lance on its
        first run. The run loop drops the OS caches before each query, so without this step the
        first Lance query would read files that the conversion just left in the page cache.
        """
        if self.backend is not Engine.LANCE:
            return
        cmd = [str(self.binary_path), benchmark.value, "--data-only"]
        for key, value in (options or {}).items():
            cmd.extend(["--opt", f"{key}={value}"])

        if self.verbose:
            console.print(f"[dim]$ {' '.join(cmd)}[/dim]")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to write {self.backend.value} data for {benchmark.value}: {result.stderr.strip()}"
            )

    def list_queries(
        self,
        benchmark: Benchmark,
        queries: list[int] | None = None,
        exclude_queries: list[int] | None = None,
    ) -> list[int]:
        """Return query indices this benchmark selects"""
        cmd = [str(self.binary_path), benchmark.value, "--print-queries"]
        if queries:
            cmd.extend(["--queries", ",".join(map(str, queries))])
        if exclude_queries:
            cmd.extend(["--exclude-queries", ",".join(map(str, exclude_queries))])

        if self.verbose:
            console.print(f"[dim]$ {' '.join(cmd)}[/dim]")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to list queries for {self.backend.value} {benchmark.value}: {result.stderr.strip()}"
            )

        return [int(line) for line in result.stdout.split() if line.strip()]

    def run(
        self,
        benchmark: Benchmark,
        formats: list[Format],
        query: int | None = None,
        iterations: int = 5,
        options: dict[str, str] | None = None,
        track_memory: bool = False,
        samply: bool = False,
        sample_rate: int | None = None,
        tracing: bool = False,
        runner: str | None = None,
        ingest_output: Path | None = None,
        on_result: Callable[[str], None] | None = None,
        integration: Integration = Integration.FILE,
        scan: ScanEngine | None = None,
    ) -> list[str]:
        """
        Run benchmark and return results as JSON lines.

        Args:
            benchmark: The benchmark suite to run
            formats: Data formats to benchmark
            query: Specific query to run (None for all)
            iterations: Number of runs per query
            options: Additional options (e.g., scale_factor)
            track_memory: Enable memory tracking
            integration: How the engine reaches Vortex files
            scan: The Vortex scan engine, or None for the suite's default
            on_result: Callback for each result line (for streaming)

        Returns:
            List of JSON lines from the benchmark output
        """
        cmd = self.build_command(
            benchmark=benchmark,
            formats=formats,
            query=query,
            iterations=iterations,
            options=options,
            track_memory=track_memory,
            samply=samply,
            sample_rate=sample_rate,
            tracing=tracing,
            runner=runner,
            ingest_output=ingest_output,
            integration=integration,
            scan=scan,
        )

        if self.verbose:
            console.print(f"[dim]$ {' '.join(cmd)}[/dim]")

        results: list[str] = []
        diagnostic_lines: deque[str] = deque(maxlen=200)

        # Outside a terminal (CI logs) the transient spinner only leaves an empty line behind.
        spinner = (
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            )
            if console.is_terminal
            else nullcontext()
        )
        with spinner as progress:
            if progress is not None:
                progress.add_task(f"Running {self.backend.value} {benchmark.value}...", total=None)

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
                            # soft_wrap keeps long log lines intact instead of folding them at the
                            # console width.
                            console.print(line, markup=False, highlight=False, soft_wrap=True)
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
