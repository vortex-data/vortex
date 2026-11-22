# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Result storage and retrieval."""

import json
import os
import re
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import TextIO

import pandas as pd

from ..config import BuildConfig, RunConfig, get_results_dir
from .schema import EnvTriple, QueryResult, RunMetadata, RunSummary


def _get_git_info() -> tuple[str, str, bool]:
    """Get git commit, branch, and dirty status."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        dirty = (
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            != ""
        )

        return commit, branch, dirty
    except subprocess.CalledProcessError:
        return "unknown", "unknown", False


def _get_env_triple() -> EnvTriple:
    """Get the current environment triple."""
    import platform

    return EnvTriple(
        architecture=platform.machine(),
        operating_system=platform.system().lower(),
        environment="unknown",
    )


class RunContext:
    """Context manager for writing results to a run."""

    def __init__(self, run_dir: Path, metadata: RunMetadata, *, _reserved: bool = False) -> None:
        self.run_dir = run_dir
        self.metadata = metadata
        self._reserved = _reserved
        self._results_file: TextIO | None = None
        self._result_count = 0

    def __enter__(self) -> "RunContext":
        if not self._reserved:
            self.run_dir.mkdir(parents=True, exist_ok=False)
        self._results_file = open(self.run_dir / "results.jsonl", "x")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._results_file:
            self._results_file.close()

        # Mark as partial if there was an exception
        if exc_type is not None:
            self.metadata.partial = True

        self.metadata.completed_at = datetime.now()

        # Write metadata
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

    def write_result(self, result: QueryResult) -> None:
        """Write a single result to the results file."""
        if self._results_file:
            self._results_file.write(json.dumps(result.to_dict()) + "\n")
            self._results_file.flush()
            self._result_count += 1

    def write_raw_json(self, json_line: str) -> None:
        """Write a raw JSON line directly (from benchmark binary output).

        Non-JSON lines (e.g. DuckDB ASCII table output) are silently skipped.
        """
        line = json_line.strip()
        if self._results_file and line.startswith("{"):
            self._results_file.write(line + "\n")
            self._results_file.flush()
            self._result_count += 1

    @property
    def result_count(self) -> int:
        return self._result_count


class ResultStore:
    """Manages benchmark result storage and retrieval."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or get_results_dir()

    def _generate_run_id(self, config: RunConfig, timestamp: datetime | None = None) -> str:
        """Generate a flat base run ID with timestamp prefix."""
        ts = (timestamp or datetime.now()).strftime("%Y-%m-%dT%H-%M-%S")
        benchmark = config.benchmark.value
        if config.label:
            safe_label = re.sub(r"[/\\]+", "_", config.label).strip("_.")
            if safe_label:
                return f"{ts}_{benchmark}_{safe_label}"
        return f"{ts}_{benchmark}"

    def _reserve_run_dir(self, base_run_id: str) -> tuple[str, Path]:
        """Exclusively reserve a top-level run directory, appending a unique suffix on collision."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        run_id = base_run_id
        while True:
            run_dir = self.base_dir / run_id
            try:
                run_dir.mkdir(parents=False, exist_ok=False)
                return run_id, run_dir
            except FileExistsError:
                run_id = f"{base_run_id}_{uuid.uuid4().hex[:8]}"

    def _publish_latest(self, run_id: str) -> None:
        """Atomically point the latest symlink at run_id."""
        latest_link = self.base_dir / "latest"
        tmp_link = self.base_dir / f".latest.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            tmp_link.symlink_to(run_id)
            os.replace(tmp_link, latest_link)
        except BaseException:
            tmp_link.unlink(missing_ok=True)
            raise

    @contextmanager
    def create_run(self, config: RunConfig, build_config: BuildConfig) -> Iterator[RunContext]:
        """Create a new run and return a context for writing results."""
        now = datetime.now()
        base_run_id = self._generate_run_id(config, timestamp=now)
        run_id, run_dir = self._reserve_run_dir(base_run_id)

        git_commit, git_branch, git_dirty = _get_git_info()
        env_triple = _get_env_triple()

        metadata = RunMetadata(
            run_id=run_id,
            timestamp=now,
            label=config.label,
            benchmark=config.benchmark.value,
            dataset_config=config.options,
            engines=[e.value for e in config.engines],
            formats=[f.value for f in config.formats],
            targets=[target.to_dict() for target in config.targets],
            queries=config.queries or [],
            iterations=config.iterations,
            git_commit=git_commit,
            git_branch=git_branch,
            git_dirty=git_dirty,
            env_triple=env_triple,
            rustflags=build_config.rustflags,
            profile=build_config.profile,
        )

        ctx = RunContext(run_dir, metadata, _reserved=True)
        with ctx:
            yield ctx

        if not ctx.metadata.partial:
            self._publish_latest(run_id)

    def _iter_run_dirs(self) -> Iterator[Path]:
        """Yield run directories containing metadata.json, skipping symlinks and hidden paths."""
        if not self.base_dir.exists():
            return
        for root, dirs, files in os.walk(self.base_dir, followlinks=False):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if d != "latest" and not d.startswith(".") and not (root_path / d).is_symlink()]
            if root_path != self.base_dir and "metadata.json" in files:
                yield root_path

    def _resolve_latest_run_id(self, latest_link: Path) -> str:
        """Return the full run ID targeted by the latest symlink."""
        target_path = Path(os.readlink(latest_link))
        if not target_path.is_absolute():
            return target_path.as_posix()
        resolved = target_path.resolve()
        try:
            return resolved.relative_to(self.base_dir.resolve()).as_posix()
        except ValueError:
            return resolved.name

    def list_runs(
        self,
        benchmark: str | None = None,
        since: datetime | None = None,
        limit: int | None = 50,
    ) -> list[RunSummary]:
        """List runs matching criteria, sorted by timestamp descending."""
        if not self.base_dir.exists():
            return []

        summaries = []
        for run_dir in self._iter_run_dirs():
            metadata_path = run_dir / "metadata.json"

            try:
                with open(metadata_path) as f:
                    data = json.load(f)
                metadata = RunMetadata.from_dict(data)

                # Apply filters
                if benchmark and metadata.benchmark != benchmark:
                    continue
                if since and metadata.timestamp < since:
                    continue

                # Count results
                results_path = run_dir / "results.jsonl"
                result_count = 0
                if results_path.exists():
                    with open(results_path) as f:
                        result_count = sum(1 for _ in f)

                summaries.append(RunSummary.from_metadata(metadata, result_count))
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by timestamp descending
        summaries.sort(key=lambda s: s.timestamp, reverse=True)
        if limit:
            return summaries[:limit]
        else:
            return summaries

    def _read_run_metadata(self, run_dir: Path) -> RunMetadata | None:
        """Read RunMetadata from run_dir/metadata.json, or return None if missing or invalid."""
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            return None
        try:
            with open(metadata_path) as f:
                return RunMetadata.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def get_run(self, run_id_or_label: str) -> RunMetadata | None:
        """Get a run by ID, label, or 'latest'."""
        if run_id_or_label == "latest":
            latest_link = self.base_dir / "latest"
            if latest_link.is_symlink():
                target_id = self._resolve_latest_run_id(latest_link)
                target_meta = self._read_run_metadata(self.base_dir / target_id)
                if target_meta is not None and not target_meta.partial:
                    return target_meta
            # Fall back to most recent non-partial run (handles missing, broken, or partial latest link).
            complete_runs = [r for r in self.list_runs(limit=None) if not r.partial]
            if not complete_runs:
                return None
            return self._read_run_metadata(self.base_dir / complete_runs[0].run_id)

        # Try direct match
        run_dir = self.base_dir / run_id_or_label
        if run_dir.exists():
            metadata = self._read_run_metadata(run_dir)
            if metadata is not None:
                return metadata

        # Try to find by label
        for run_dir in self._iter_run_dirs():
            metadata = self._read_run_metadata(run_dir)
            if metadata is not None and metadata.label == run_id_or_label:
                return metadata

        return None

    def load_results(self, run_id: str) -> pd.DataFrame:
        """Load results as a DataFrame for analysis."""
        run_dir = self.base_dir / run_id
        results_path = run_dir / "results.jsonl"

        if not results_path.exists():
            return pd.DataFrame()

        return pd.read_json(results_path, lines=True)

    def resolve_reference(self, ref: str) -> str | None:
        """Resolve 'latest', labels, or partial IDs to full run_id."""
        metadata = self.get_run(ref)
        return metadata.run_id if metadata else None

    def _has_descendant_runs(self, run_dir: Path) -> bool:
        """Return whether run_dir contains any nested run directories with metadata.json."""
        for root, dirs, files in os.walk(run_dir, followlinks=False):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if not (root_path / d).is_symlink()]
            if root_path != run_dir and "metadata.json" in files:
                return True
        return False

    def delete_run(self, run_id: str) -> bool:
        """Delete a run by ID without removing any descendant runs nested beneath it."""
        import shutil

        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            return False

        if not self._has_descendant_runs(run_dir):
            shutil.rmtree(run_dir)
            return True

        # Preserve older nested descendant runs by deleting only this run's own entries.
        for child in list(run_dir.iterdir()):
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir() and not (child / "metadata.json").exists() and not self._has_descendant_runs(child):
                shutil.rmtree(child)
        return True
