# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Benchmark binary execution."""

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal, final

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import Benchmark, Engine, Format

console = Console()


@final
class BenchmarkExecutor:
    """Executes benchmark binaries and captures output."""

    def __init__(self, binary_path: Path, engine: Engine, verbose: bool = False):
        self.binary_path = binary_path
        self.engine = engine
        self.verbose = verbose

    def run(
        self,
        benchmark: Benchmark,
        formats: list[Format],
        queries: list[int] | None = None,
        exclude_queries: list[int] | None = None,
        iterations: int = 5,
        options: dict[str, str] | None = None,
        track_memory: bool = False,
        show_session_metrics: bool = False,
        samply: bool = False,
        sample_rate: int | None = None,
        tracing: bool = False,
        allocator_mode: Literal["pooled", "default"] = "pooled",
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
            show_session_metrics: Print session metrics at the end of the benchmark process
            allocator_mode: Host allocator mode ("pooled" or "default")
            on_result: Callback for each result line (for streaming)

        Returns:
            List of JSON lines from the benchmark output
        """
        cmd = [
            str(self.binary_path),
            benchmark.value,
            "--display-format",
            "gh-json",
            "--iterations",
            str(iterations),
            "--formats",
            ",".join(f.value for f in formats),
            "--hide-progress-bar",
        ]

        if queries:
            cmd.extend(["--queries", ",".join(map(str, queries))])
        if exclude_queries:
            cmd.extend(["--exclude-queries", ",".join(map(str, exclude_queries))])
        if track_memory:
            cmd.append("--track-memory")
        if show_session_metrics:
            cmd.append("--show-session-metrics")
        if tracing:
            cmd.append("--tracing")
        if options:
            for k, v in options.items():
                cmd.extend(["--opt", f"{k}={v}"])

        if samply:
            cmd = ["--"] + cmd
            cmd_prefix = ["samply", "record"]
            if sample_rate:
                cmd = cmd_prefix + ["--rate", str(sample_rate)] + cmd
            else:
                cmd = cmd_prefix + cmd

        if samply and self.engine == Engine.DUCKDB:
            # Re-use the same DuckDB instance across runs make samply output readable
            cmd += ["--reuse"]

        if self.verbose:
            console.print(f"[dim]$ {' '.join(cmd)}[/dim]")

        env = os.environ.copy()
        env["VORTEX_HOST_ALLOCATOR"] = allocator_mode

        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            _task = progress.add_task(f"Running {self.engine.value} {benchmark.value}...", total=None)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                text=True,
                env=env,
            )

            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                if line:
                    if line.startswith("{"):
                        try:
                            payload = json.loads(line)
                            if isinstance(payload.get("target"), dict):
                                payload["target"]["allocator"] = allocator_mode
                            line = json.dumps(payload)
                        except json.JSONDecodeError:
                            pass
                    results.append(line)
                    if on_result:
                        on_result(line)

            ret_code = process.wait()

            if ret_code != 0:
                stderr = process.stderr.read() if process.stderr else ""
                console.print(f"[red]Benchmark failed with code {process.returncode}[/red]")
                if stderr:
                    console.print(f"[red]{stderr}[/red]")
                raise RuntimeError(f"Benchmark {self.engine.value} {benchmark.value} failed: {stderr}")

        return results
