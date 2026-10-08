# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# /// script
# dependencies = ["pyarrow", "vortex-data"]
# ///
# Read every .vortex file in a directory with the installed vortex-data wheel and dump each one as
# Arrow IPC (<name>.arrow), or record the read error in <name>.error.
#
# If a `queries.json` sidecar is present (written by `vortex-compat sweep`), each file's queries are
# run too and dumped as <name>.q<id>.arrow or <name>.q<id>.error.
#
# Files are read by a child process in batches. When the child aborts (a non-unwinding panic in
# the reader), the file it was on is recorded as aborted and a new child resumes from the next
# file, so one bad file costs one process rather than ending the run.
import json
import operator
import os
import pathlib
import subprocess
import sys

OPS = {
    "eq": operator.eq,
    "not_eq": operator.ne,
    "gt": operator.gt,
    "gt_eq": operator.ge,
    "lt": operator.lt,
    "lt_eq": operator.le,
}

EXCEPTION_PREFIXES = ("pyarrow.lib.", "ValueError", "RuntimeError", "Exception", "TypeError", "AttributeError")


def write_ipc(path: pathlib.Path, table: object) -> None:
    import pyarrow.ipc as ipc

    with ipc.new_file(str(path), table.schema) as w:
        w.write_table(table)


def read_one(src: pathlib.Path, dst: pathlib.Path) -> int:
    import vortex as vx

    table = vx.open(str(src)).to_arrow().read_all()
    write_ipc(dst / f"{src.name}.arrow", table)
    return table.num_rows


def literal_dtype(spec: dict) -> object:
    """A vortex DType for a query literal, using constructors every binding has."""
    from vortex._lib import dtype as vd

    nullable = spec["nullable"]
    kind = spec["kind"]
    if kind == "int":
        return vd.int_(spec["width"], nullable=nullable)
    if kind == "uint":
        return vd.uint(spec["width"], nullable=nullable)
    if kind == "float":
        return vd.float_(spec["width"], nullable=nullable)
    if kind == "bool":
        return vd.bool_(nullable=nullable)
    if kind == "utf8":
        return vd.utf8(nullable=nullable)
    raise ValueError(f"unknown literal kind {kind}")


def run_query(src: pathlib.Path, dst: pathlib.Path, query: dict) -> int:
    import pyarrow as pa

    import vortex as vx
    import vortex.expr as ve

    f = vx.open(str(src))
    projection = query.get("projection")
    if query.get("indices") is not None:
        indices = vx.array(pa.array(query["indices"], type=pa.uint64()))
        table = f.scan(projection, indices=indices).read_all().to_arrow_table()
    elif query.get("limit") is not None:
        # `scan(limit=)` exists on every wheel; `to_arrow(limit=)` does not.
        table = f.scan(projection, limit=query["limit"]).read_all().to_arrow_table()
    elif query.get("filter") is None:
        table = f.to_arrow(projection).read_all()
    else:
        flt = query["filter"]
        expr = OPS[flt["op"]](ve.column(flt["column"]), ve.literal(literal_dtype(flt["dtype"]), flt["value"]))
        table = f.to_arrow(projection, expr=expr).read_all()
    write_ipc(dst / f"{src.name}.q{query['id']}.arrow", table)
    return table.num_rows


def load_queries(src_dir: pathlib.Path) -> dict:
    path = src_dir / "queries.json"
    return json.loads(path.read_text()) if path.exists() else {}


def child(dst: pathlib.Path, files: list[pathlib.Path]) -> None:
    """Read each file in turn, then run its queries, reporting one line per step on stdout."""
    queries = load_queries(files[0].parent) if files else {}
    for f in files:
        print(f"BEGIN {f.name}", flush=True)
        try:
            rows = read_one(f, dst)
            print(f"OK {f.name} {rows}", flush=True)
        except Exception as e:  # noqa: BLE001 - every reader error is a result here
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            (dst / f"{f.name}.error").write_text(msg)
            print(f"FAIL {f.name} {msg}", flush=True)
        for query in queries.get(f.name, []):
            qid = query["id"]
            print(f"QBEGIN {f.name}#q{qid}", flush=True)
            try:
                rows = run_query(f, dst, query)
                print(f"QOK {f.name}#q{qid} {rows}", flush=True)
            except Exception as e:  # noqa: BLE001
                msg = str(e).splitlines()[0] if str(e) else type(e).__name__
                (dst / f"{f.name}.q{qid}.error").write_text(msg)
                print(f"QFAIL {f.name}#q{qid} {msg}", flush=True)


def panic_message(stderr: str) -> str:
    lines = [line for line in stderr.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if "panicked at" in line and i + 1 < len(lines):
            return lines[i + 1].strip()
    return lines[-1].strip() if lines else "no stderr"


def main() -> None:
    if sys.argv[1] == "--child":
        child(pathlib.Path(sys.argv[2]), [pathlib.Path(p) for p in sys.argv[3:]])
        return

    src, dst = map(pathlib.Path, sys.argv[1:3])
    dst.mkdir(parents=True, exist_ok=True)
    import vortex as vx

    print(f"old-reader: vortex-data {getattr(vx, '__version__', '?')}", file=sys.stderr)
    pending = sorted(src.glob("*.vortex"))
    queries = load_queries(src)
    env = {"RUST_BACKTRACE": "0", **os.environ}
    while pending:
        proc = subprocess.run(
            [sys.executable, "-I", __file__, "--child", str(dst), *map(str, pending)],
            capture_output=True,
            text=True,
            env=env,
        )
        done: set[str] = set()
        current: str | None = None
        current_file: str | None = None
        for line in proc.stdout.splitlines():
            tag, _, rest = line.partition(" ")
            if tag == "BEGIN":
                current = current_file = rest
            elif tag == "QBEGIN":
                current = rest
            elif tag in ("OK", "FAIL", "QOK", "QFAIL"):
                name, _, detail = rest.partition(" ")
                if tag in ("OK", "FAIL"):
                    done.add(name)
                current = None
                if tag in ("OK", "QOK"):
                    print(f"  ok   {name} ({detail} rows)", file=sys.stderr)
                else:
                    print(f"  FAIL {name}: {detail}", file=sys.stderr)
        if proc.returncode == 0 and current is None:
            break
        # The child died. Blame the step it was on, mark the rest of that file's queries as not
        # run, then resume with the next file.
        victim = current or current_file
        if victim is None:
            if len(done) < len(pending):
                victim = pending[len(done)].name
            else:
                break
        msg = f"reader aborted (exit {proc.returncode}): {panic_message(proc.stderr)}"
        file_name, _, qid = victim.partition("#q")
        if qid:
            (dst / f"{file_name}.q{qid}.error").write_text(msg)
            later = "not run: the reader aborted on an earlier query of this file"
        else:
            (dst / f"{file_name}.error").write_text(msg)
            later = "not run: the reader aborted reading this file"
        for query in queries.get(file_name, []):
            arrow = dst / f"{file_name}.q{query['id']}.arrow"
            error = dst / f"{file_name}.q{query['id']}.error"
            if not arrow.exists() and not error.exists():
                error.write_text(later)
        print(f"  FAIL {victim}: {msg}", file=sys.stderr)
        done.add(file_name)
        pending = [f for f in pending if f.name not in done]


main()
