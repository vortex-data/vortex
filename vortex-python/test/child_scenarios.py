# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""The child-process scenarios driven by `test_fork.py`.

Run as a script, one scenario per interpreter:

    python child_scenarios.py <scenario> <start-method> <vortex-file>

It prints `OK` on success and raises otherwise.

This is a real module rather than text handed to `python -c` because `spawn` and `forkserver`
re-import `__main__` in the child, and can only pickle callables that live at module level. Every
child target below is therefore top-level and takes its arguments explicitly instead of closing
over parent state.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import TypeVar, cast

import pyarrow as pa

import vortex as vx
import vortex.expr as ve
from vortex.expr import Expr
from vortex.file import VortexFile

ROWS = 10_000
CHILD_TIMEOUT_SECONDS = 60

FORK = "fork"
SPAWN = "spawn"
FORKSERVER = "forkserver"
ALL_START_METHODS = (FORK, SPAWN, FORKSERVER)

T = TypeVar("T")


def in_child(ctx: BaseContext, target: Callable[..., T], *args: object) -> T:
    """Run `target` in one child process and return its result, or raise on hang or error."""
    with ctx.Pool(1) as pool:
        result = pool.apply_async(target, args)
        try:
            return result.get(timeout=CHILD_TIMEOUT_SECONDS)
        except multiprocessing.TimeoutError:
            pool.terminate()
            raise AssertionError(f"child produced no result within {CHILD_TIMEOUT_SECONDS}s") from None


# Child targets. Top-level so that every start method can pickle them by reference.


def count_rows(path: str) -> int:
    return len(vx.open(path).scan().read_all())


def count_file_rows(file: VortexFile) -> int:
    return len(file.scan().read_all())


def count_filtered(args: tuple[str, Expr]) -> int:
    path, expr = args
    return len(vx.open(path).scan(expr=expr).read_all())


def filtered_names(path: str, expr: Expr) -> list[str]:
    table = vx.open(path).scan(["name"], expr=expr).read_all().to_arrow_table()
    return cast("list[str]", table.column("name").to_pylist())


def write_then_count(values: vx.Array, path: str) -> int:
    vx.io.write(values, path)
    return len(vx.open(path))


def worker_threads() -> int:
    return vx.worker_threads()


# Scenarios, each run in the parent of a fresh interpreter.


def read_after_parent_read(ctx: BaseContext, path: str) -> None:
    """The parent has already built a runtime, so the child inherits a populated (and useless) one
    under `fork` and must build its own before it can read anything."""
    assert count_rows(path) == ROWS
    assert in_child(ctx, count_rows, path) == ROWS


def read_after_saturated_blocking_pool(ctx: BaseContext, path: str) -> None:
    """Reaching the blocking-pool limit in the parent must not prevent a fresh pool from starting in
    the child. The old process-global `blocking` executor could never recover from this state."""
    os.environ["BLOCKING_MAX_THREADS"] = "1"
    assert count_rows(path) == ROWS
    assert in_child(ctx, count_rows, path) == ROWS


def write_after_parent_write(ctx: BaseContext, path: str) -> None:
    """Local writes use the runtime-owned blocking pool too; otherwise async-fs would retain the
    same process-global fork hazard after the read path was fixed."""
    values = vx.array(pa.array([{"value": i} for i in range(10)]))
    vx.io.write(values, f"{path}.parent-write")
    assert in_child(ctx, write_then_count, values, f"{path}.child-write") == 10


def child_has_live_workers(ctx: BaseContext, path: str) -> None:
    """The child must end up with real worker threads, not the parent's phantom handles."""
    assert count_rows(path) == ROWS
    assert in_child(ctx, worker_threads) >= 1


def expr_filter_in_child(ctx: BaseContext, path: str) -> None:
    """An Expr pickled into the child and evaluated there."""
    assert count_rows(path) == ROWS
    expr = (ve.column("age") >= ROWS - 2) & ve.like(ve.column("name"), "person-%")
    assert in_child(ctx, filtered_names, path, expr) == [f"person-{ROWS - 2}", f"person-{ROWS - 1}"]


def child_inherits_worker_count(ctx: BaseContext, path: str) -> None:
    """The worker count configured in the parent must carry over to the forked child's fresh pool.

    Fork-only: `spawn` and `forkserver` children start from a new interpreter, so they legitimately
    know nothing about the parent's configuration.
    """
    vx.set_worker_threads(3)
    assert count_rows(path) == ROWS
    assert in_child(ctx, worker_threads) == 3


def pool_map_with_pickled_expr(ctx: BaseContext, path: str) -> None:
    """A Pool pickles both the callable and its arguments: the shape `datasets` uses for `num_proc`,
    and the reason Expr needs `__reduce__`."""
    assert count_rows(path) == ROWS
    expr = ve.column("age") < 100
    with ctx.Pool(2) as pool:
        result = pool.map_async(count_filtered, [(path, expr), (path, expr)])
        try:
            counts = result.get(timeout=CHILD_TIMEOUT_SECONDS)
        except multiprocessing.TimeoutError:
            pool.terminate()
            raise AssertionError(f"Pool did not finish within {CHILD_TIMEOUT_SECONDS}s") from None
    assert counts == [100, 100], counts


def pickled_file_reads_in_child(ctx: BaseContext, path: str) -> None:
    """A pickled VortexFile is reopened by path in the child."""
    vxf = vx.open(path)
    assert len(vxf.scan().read_all()) == ROWS
    restored = cast(VortexFile, cast(object, pickle.loads(pickle.dumps(vxf))))
    assert restored.path == path
    assert in_child(ctx, count_file_rows, restored) == ROWS


@dataclass(frozen=True)
class Scenario:
    run: Callable[[BaseContext, str], None]
    start_methods: Sequence[str] = ALL_START_METHODS


SCENARIOS: dict[str, Scenario] = {
    "read_after_parent_read": Scenario(read_after_parent_read),
    "read_after_saturated_blocking_pool": Scenario(read_after_saturated_blocking_pool),
    "write_after_parent_write": Scenario(write_after_parent_write),
    "child_has_live_workers": Scenario(child_has_live_workers),
    "expr_filter_in_child": Scenario(expr_filter_in_child),
    "child_inherits_worker_count": Scenario(child_inherits_worker_count, start_methods=(FORK,)),
    "pool_map_with_pickled_expr": Scenario(pool_map_with_pickled_expr),
    "pickled_file_reads_in_child": Scenario(pickled_file_reads_in_child),
}


def main(argv: Sequence[str]) -> None:
    name, start_method, path = argv
    scenario = SCENARIOS[name]
    if start_method not in scenario.start_methods:
        raise AssertionError(f"scenario {name!r} does not apply to the {start_method!r} start method")
    scenario.run(multiprocessing.get_context(start_method), path)
    print("OK")


if __name__ == "__main__":
    main(sys.argv[1:])
