# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""The Vortex runtime must be rebuilt in forked children, or every read there hangs.

Each scenario runs in a **fresh interpreter** that imports nothing but Vortex, and only then forks.
That keeps the tests measuring Vortex's own fork behaviour: several third-party libraries (Ray,
DuckDB, CUDA drivers) install their own `atfork` handlers or hold locks that make `fork` unsafe
process-wide, so forking straight from the pytest process would be testing them as much as us.

A regression shows up as a timeout, since the failure mode being guarded against is a hang.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pyarrow as pa
import pytest

import vortex as vx

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="fork(2) is not available on Windows")

TIMEOUT_SECONDS = 120
ROWS = 10_000

PREAMBLE = textwrap.dedent(
    """
    import multiprocessing, sys
    import vortex as vx
    import vortex.expr as ve

    PATH = sys.argv[1]
    CTX = multiprocessing.get_context("fork")

    def in_fork(target, *args):
        "Run `target` in a forked child and return its result, or raise on hang/error."
        parent_conn, child_conn = CTX.Pipe(duplex=False)

        def entry():
            try:
                child_conn.send(("ok", target(*args)))
            except BaseException as err:
                child_conn.send(("err", f"{type(err).__name__}: {err}"))
            finally:
                child_conn.close()

        process = CTX.Process(target=entry)
        process.start()
        try:
            if not parent_conn.poll(60):
                process.terminate()
                raise AssertionError("forked child produced no result within 60s")
            status, payload = parent_conn.recv()
        finally:
            process.join(30)
        if status == "err":
            raise AssertionError(f"forked child raised {payload}")
        return payload
    """
)

# Each scenario must print nothing but a trailing "OK" on success.
SCENARIOS: dict[str, str] = {
    # The parent has already built a runtime, so the child inherits a populated (and useless) one
    # and must build its own before it can read anything.
    "read_after_parent_read": """
        assert len(vx.open(PATH).scan().read_all()) == ROWS

        def child():
            return len(vx.open(PATH).scan().read_all())

        assert in_fork(child) == ROWS
        print("OK")
    """,
    # Reaching the blocking-pool limit in the parent must not prevent a fresh pool from starting in
    # the child. The old process-global `blocking` executor could never recover from this state.
    "read_after_saturated_blocking_pool": """
        import os
        os.environ["BLOCKING_MAX_THREADS"] = "1"
        assert len(vx.open(PATH).scan().read_all()) == ROWS

        def child():
            return len(vx.open(PATH).scan().read_all())

        assert in_fork(child) == ROWS
        print("OK")
    """,
    # Local writes use the runtime-owned blocking pool too; otherwise async-fs would retain the
    # same process-global fork hazard after the read path was fixed.
    "write_after_parent_write": """
        import pyarrow as pa
        values = vx.array(pa.array([{"value": i} for i in range(10)]))
        vx.io.write(values, f"{PATH}.parent-write")

        def child():
            child_path = f"{PATH}.child-write"
            vx.io.write(values, child_path)
            return len(vx.open(child_path))

        assert in_fork(child) == 10
        print("OK")
    """,
    # The child must end up with real worker threads, not the parent's phantom handles.
    "child_has_live_workers": """
        assert len(vx.open(PATH).scan().read_all()) == ROWS
        assert in_fork(vx.worker_threads) >= 1
        print("OK")
    """,
    # An Expr filter evaluated in the child.
    "expr_filter_in_child": """
        assert len(vx.open(PATH).scan().read_all()) == ROWS
        expr = (ve.column("age") >= ROWS - 2) & ve.like(ve.column("name"), "person-%")

        def child(expr):
            table = vx.open(PATH).scan(["name"], expr=expr).read_all().to_arrow_table()
            return table.column("name").to_pylist()

        assert in_fork(child, expr) == [f"person-{ROWS - 2}", f"person-{ROWS - 1}"]
        print("OK")
    """,
    # The worker count configured in the parent must carry over to the child's fresh pool.
    "child_inherits_worker_count": """
        vx.set_worker_threads(3)
        assert len(vx.open(PATH).scan().read_all()) == ROWS
        assert in_fork(vx.worker_threads) == 3
        print("OK")
    """,
    # A fork-backed Pool pickles both the callable and its arguments: the shape `datasets` uses for
    # `num_proc`, and the reason Expr needs `__reduce__`.
    "pool_map_with_pickled_expr": """
        assert len(vx.open(PATH).scan().read_all()) == ROWS
        expr = ve.column("age") < 100
        with CTX.Pool(2) as pool:
            result = pool.map_async(count_filtered, [(PATH, expr), (PATH, expr)])
            try:
                counts = result.get(timeout=60)
            except multiprocessing.TimeoutError:
                pool.terminate()
                raise AssertionError("fork-backed Pool did not finish within 60s")
        assert counts == [100, 100], counts
        print("OK")
    """,
    # A pickled VortexFile is reopened by path in the child.
    "pickled_file_reads_in_child": """
        import pickle
        vxf = vx.open(PATH)
        assert len(vxf.scan().read_all()) == ROWS
        restored = pickle.loads(pickle.dumps(vxf))
        assert restored.path == PATH
        assert in_fork(read_file, restored) == ROWS
        print("OK")
    """,
}

# Top-level so `Pool.map` can pickle it by reference.
HELPERS = textwrap.dedent(
    """
    def count_filtered(args):
        path, expr = args
        return len(vx.open(path).scan(expr=expr).read_all())

    def read_file(file):
        return len(file.scan().read_all())
    """
)


@pytest.fixture(scope="module")
def people(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("fork") / "people.vortex"
    vx.io.write(vx.array(pa.array([{"name": f"person-{i}", "age": i} for i in range(ROWS)])), str(path))
    return path


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_fork_scenario(name: str, people: Path) -> None:
    script = "\n".join(
        [
            PREAMBLE,
            f"ROWS = {ROWS}",
            HELPERS,
            "if __name__ == '__main__':",
            textwrap.indent(textwrap.dedent(SCENARIOS[name]), "    "),
        ]
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, str(people)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"scenario {name!r} did not finish within {TIMEOUT_SECONDS}s (likely a fork deadlock)")

    if proc.returncode != 0 or not proc.stdout.strip().endswith("OK"):
        pytest.fail(
            f"scenario {name!r} failed (exit {proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
