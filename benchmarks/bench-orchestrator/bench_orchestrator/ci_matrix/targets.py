# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Ordered target sets used by CI benchmark declarations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from ..config import BenchmarkTarget, Engine, Format


def _dedupe(targets: Iterable[BenchmarkTarget]) -> tuple[BenchmarkTarget, ...]:
    """Normalize and de-duplicate targets, preserving first-seen order."""
    return tuple(dict.fromkeys(target.normalized() for target in targets))


@dataclass(frozen=True)
class TargetSet:
    """An ordered set of engine/format targets with small set algebra."""

    targets: tuple[BenchmarkTarget, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", _dedupe(self.targets))

    def __or__(self, other: TargetSet) -> TargetSet:
        """Return the ordered union of two target sets."""
        return TargetSet(self.targets + other.targets)

    def only(self, *engines: Engine) -> TargetSet:
        """Restrict the target set to the given engines."""
        keep = set(engines)
        return TargetSet(tuple(target for target in self.targets if target.engine in keep))

    def __iter__(self) -> Iterator[BenchmarkTarget]:
        return iter(self.targets)

    def __len__(self) -> int:
        return len(self.targets)


def targets(engine: Engine, *formats: Format) -> TargetSet:
    """Build targets for one engine across several formats."""
    return TargetSet(tuple(BenchmarkTarget(engine=engine, format=fmt) for fmt in formats))


def df(*formats: Format) -> TargetSet:
    """Build DataFusion targets across several formats."""
    return targets(Engine.DATAFUSION, *formats)


def duck(*formats: Format) -> TargetSet:
    """Build DuckDB targets across several formats."""
    return targets(Engine.DUCKDB, *formats)
