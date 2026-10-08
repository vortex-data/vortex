# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# /// script
# dependencies = ["pyarrow", "vortex-data"]
# ///
# Read every .vortex file in a directory with the installed vortex-data wheel and dump each one as
# Arrow IPC (<name>.arrow), or record the read error in <name>.error.
#
# Files are read by a child process in batches. When the child aborts (a non-unwinding panic in
# the reader), the file it was on is recorded as aborted and a new child resumes from the next
# file, so one bad file costs one process rather than ending the run.
import os
import pathlib
import subprocess
import sys

EXCEPTION_PREFIXES = ("pyarrow.lib.", "ValueError", "RuntimeError", "Exception", "TypeError", "AttributeError")


def read_one(src: pathlib.Path, dst: pathlib.Path) -> int:
    import pyarrow.ipc as ipc

    import vortex as vx

    table = vx.open(str(src)).to_arrow().read_all()
    with ipc.new_file(str(dst / f"{src.name}.arrow"), table.schema) as w:
        w.write_table(table)
    return table.num_rows


def child(dst: pathlib.Path, files: list[pathlib.Path]) -> None:
    """Read each file in turn, reporting one line per file on stdout."""
    for f in files:
        print(f"BEGIN {f.name}", flush=True)
        try:
            rows = read_one(f, dst)
            print(f"OK {f.name} {rows}", flush=True)
        except Exception as e:  # noqa: BLE001 - every reader error is a result here
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            (dst / f"{f.name}.error").write_text(msg)
            print(f"FAIL {f.name} {msg}", flush=True)


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
        for line in proc.stdout.splitlines():
            tag, _, rest = line.partition(" ")
            if tag == "BEGIN":
                current = rest
            elif tag in ("OK", "FAIL"):
                name, _, detail = rest.partition(" ")
                done.add(name)
                current = None
                status = "ok  " if tag == "OK" else "FAIL"
                suffix = f"({detail} rows)" if tag == "OK" else detail
                print(
                    f"  {status} {name}: {suffix}" if tag == "FAIL" else f"  {status} {name} {suffix}", file=sys.stderr
                )
        if proc.returncode == 0 and current is None:
            break
        # The child died. Blame the file it was reading, then resume after it.
        victim = current or (pending[len(done)].name if len(done) < len(pending) else None)
        if victim is None:
            break
        msg = f"reader aborted (exit {proc.returncode}): {panic_message(proc.stderr)}"
        (dst / f"{victim}.error").write_text(msg)
        print(f"  FAIL {victim}: {msg}", file=sys.stderr)
        done.add(victim)
        pending = [f for f in pending if f.name not in done]


main()
