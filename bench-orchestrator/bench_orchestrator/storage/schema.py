# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Data models for benchmark results storage."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EnvTriple:
    """Environment triple describing the target platform."""

    architecture: str
    operating_system: str
    environment: str

    def to_dict(self) -> dict[str, str]:
        return {
            "architecture": self.architecture,
            "operating_system": self.operating_system,
            "environment": self.environment,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "EnvTriple":
        return cls(
            architecture=data["architecture"],
            operating_system=data["operating_system"],
            environment=data["environment"],
        )


@dataclass
class QueryResult:
    """A single query benchmark result."""

    name: str
    storage: str
    dataset: dict[str, Any]
    unit: str
    value: int  # median in nanoseconds
    all_runtimes: list[int]
    target: dict[str, str]  # {"engine": "...", "format": "..."}
    commit_id: str
    env_triple: EnvTriple

    @property
    def engine(self) -> str:
        return self.target.get("engine", "")

    @property
    def format(self) -> str:
        return self.target.get("format", "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "storage": self.storage,
            "dataset": self.dataset,
            "unit": self.unit,
            "value": self.value,
            "all_runtimes": self.all_runtimes,
            "target": self.target,
            "commit_id": self.commit_id,
            "env_triple": self.env_triple.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryResult":
        return cls(
            name=data["name"],
            storage=data.get("storage", ""),
            dataset=data.get("dataset", {}),
            unit=data["unit"],
            value=data["value"],
            all_runtimes=data.get("all_runtimes", []),
            target=data["target"],
            commit_id=data.get("commit_id", ""),
            env_triple=EnvTriple.from_dict(data["env_triple"])
            if "env_triple" in data
            else EnvTriple("unknown", "unknown", "unknown"),
        )


@dataclass
class RunMetadata:
    """Metadata for a benchmark run."""

    run_id: str
    timestamp: datetime
    benchmark: str
    engines: list[str]
    formats: list[str]
    targets: list[dict[str, str]]
    iterations: int
    git_commit: str
    git_branch: str
    git_dirty: bool
    env_triple: EnvTriple
    rustflags: str
    profile: str
    label: str | None = None
    dataset_config: dict[str, Any] = field(default_factory=dict)
    queries: list[int] = field(default_factory=list)
    binaries: dict[str, str] = field(default_factory=dict)
    partial: bool = False
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "label": self.label,
            "benchmark": self.benchmark,
            "dataset_config": self.dataset_config,
            "engines": self.engines,
            "formats": self.formats,
            "targets": self.targets,
            "queries": self.queries,
            "iterations": self.iterations,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "env_triple": self.env_triple.to_dict(),
            "rustflags": self.rustflags,
            "profile": self.profile,
            "binaries": self.binaries,
            "partial": self.partial,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunMetadata":
        return cls(
            run_id=data["run_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            label=data.get("label"),
            benchmark=data["benchmark"],
            dataset_config=data.get("dataset_config", {}),
            engines=data["engines"],
            formats=data["formats"],
            targets=data.get("targets", []),
            queries=data.get("queries", []),
            iterations=data["iterations"],
            git_commit=data["git_commit"],
            git_branch=data["git_branch"],
            git_dirty=data["git_dirty"],
            env_triple=EnvTriple.from_dict(data["env_triple"]),
            rustflags=data["rustflags"],
            profile=data["profile"],
            binaries=data.get("binaries", {}),
            partial=data.get("partial", False),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        )


@dataclass
class RunSummary:
    """Summary of a run for listing."""

    run_id: str
    timestamp: datetime
    label: str | None
    benchmark: str
    engines: list[str]
    formats: list[str]
    git_commit: str
    partial: bool
    result_count: int = 0

    @classmethod
    def from_metadata(cls, metadata: RunMetadata, result_count: int = 0) -> "RunSummary":
        return cls(
            run_id=metadata.run_id,
            timestamp=metadata.timestamp,
            label=metadata.label,
            benchmark=metadata.benchmark,
            engines=metadata.engines,
            formats=metadata.formats,
            git_commit=metadata.git_commit,
            partial=metadata.partial,
            result_count=result_count,
        )
