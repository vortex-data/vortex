# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Run the `data-gen` binary, optionally for several datasets at once."""

import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.markup import escape

console = Console()


@dataclass(frozen=True)
class DataGenJob:
    """One `data-gen` invocation, with optional work to run before and after it."""

    name: str
    cmd: list[str]
    before: Callable[[], None] | None = field(default=None, compare=False)
    after: Callable[[], None] | None = field(default=None, compare=False)


def data_gen_command(
    binary_path: Path,
    benchmark: str,
    formats: list[str],
    options: dict[str, str],
    verbose: bool = False,
) -> list[str]:
    """Build the `data-gen` command line for one benchmark."""
    cmd = [str(binary_path), benchmark, "--formats", ",".join(formats)]
    if verbose:
        cmd.append("--verbose")
    for key, value in options.items():
        cmd.extend(["--opt", f"{key}={value}"])
    return cmd


def _run_job(job: DataGenJob, cwd: Path | None, prefix: bool) -> None:
    if job.before:
        job.before()
    if prefix:
        # Interleaved output from concurrent jobs is unreadable without a prefix.
        with subprocess.Popen(
            job.cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                console.print(f"[dim]\\[{job.name}][/dim] {escape(line.rstrip())}", highlight=False, soft_wrap=True)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, job.cmd)
    else:
        subprocess.run(job.cmd, check=True, cwd=cwd)
    if job.after:
        job.after()


def run_data_gen_jobs(
    jobs: list[DataGenJob],
    parallel: bool = False,
    cwd: Path | None = None,
    verbose: bool = False,
) -> list[str]:
    """Run every job and return the names of the ones that failed.

    Sequential runs stop at the first failure. Parallel runs let every job finish so one
    failing dataset does not hide the others' results.
    """
    for job in jobs:
        if verbose:
            console.print(f"[dim]$ {' '.join(job.cmd)}[/dim]")

    if not parallel or len(jobs) <= 1:
        for job in jobs:
            console.print(f"[blue]Generating {job.name}...[/blue]")
            try:
                _run_job(job, cwd, prefix=False)
            except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
                console.print(f"[red]Data generation failed for {job.name}: {exc}[/red]")
                return [job.name]
        return []

    console.print(f"[blue]Generating {', '.join(job.name for job in jobs)} in parallel...[/blue]")
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {job.name: pool.submit(_run_job, job, cwd, True) for job in jobs}
        for name, future in futures.items():
            try:
                future.result()
            except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
                console.print(f"[red]Data generation failed for {name}: {exc}[/red]")
                failed.append(name)
    return failed
