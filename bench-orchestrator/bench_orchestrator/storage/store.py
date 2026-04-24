# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Result storage and retrieval."""

import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

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

    def __init__(self, run_dir: Path, metadata: RunMetadata):
        self.run_dir = run_dir
        self.metadata = metadata
        self._results_file: Any | None = None
        self._result_count = 0

    def __enter__(self) -> "RunContext":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._results_file = open(self.run_dir / "results.jsonl", "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
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

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or get_results_dir()

    def _generate_run_id(self, config: RunConfig) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        benchmark = config.benchmark.value
        if config.label:
            return f"{timestamp}_{benchmark}_{config.label}"
        return f"{timestamp}_{benchmark}"

    @contextmanager
    def create_run(self, config: RunConfig, build_config: BuildConfig) -> Iterator[RunContext]:
        """Create a new run and return a context for writing results."""
        run_id = self._generate_run_id(config)
        run_dir = self.base_dir / run_id

        git_commit, git_branch, git_dirty = _get_git_info()
        env_triple = _get_env_triple()

        metadata = RunMetadata(
            run_id=run_id,
            timestamp=datetime.now(),
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

        ctx = RunContext(run_dir, metadata)
        with ctx:
            yield ctx

        # Update the latest symlink
        latest_link = self.base_dir / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(run_id)

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
        for run_dir in self.base_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name == "latest":
                continue

            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                continue

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

    def get_run(self, run_id_or_label: str) -> RunMetadata | None:
        """Get a run by ID, label, or 'latest'."""
        if run_id_or_label == "latest":
            latest_link = self.base_dir / "latest"
            if latest_link.is_symlink():
                run_id_or_label = latest_link.resolve().name
            else:
                # Find most recent run
                runs = self.list_runs(limit=1)
                if not runs:
                    return None
                run_id_or_label = runs[0].run_id

        # Try direct match
        run_dir = self.base_dir / run_id_or_label
        if run_dir.exists():
            metadata_path = run_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    return RunMetadata.from_dict(json.load(f))

        # Try to find by label
        for run_dir in self.base_dir.iterdir():
            if not run_dir.is_dir():
                continue
            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            try:
                with open(metadata_path) as f:
                    data = json.load(f)
                if data.get("label") == run_id_or_label:
                    return RunMetadata.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                continue

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

    def delete_run(self, run_id: str) -> bool:
        """Delete a run by ID."""
        import shutil

        run_dir = self.base_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
            return True
        return False
