# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# /// script
# dependencies = ["pyarrow", "vortex-data"]
# ///
# Read every .vortex file in a directory with the installed vortex-data wheel and dump each one as
# Arrow IPC (<name>.arrow), or record the read error in <name>.error. Each file is read in a child
# process so that a reader abort is recorded rather than killing the run.
import os
import pathlib
import subprocess
import sys


def read_one(src: pathlib.Path, dst: pathlib.Path) -> None:
    import pyarrow.ipc as ipc

    import vortex as vx

    table = vx.open(str(src)).to_arrow().read_all()
    with ipc.new_file(str(dst / f"{src.name}.arrow"), table.schema) as w:
        w.write_table(table)
    print(f"{table.num_rows}")


def main() -> None:
    if sys.argv[1] == "--one":
        read_one(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
        return
    src, dst = map(pathlib.Path, sys.argv[1:3])
    dst.mkdir(parents=True, exist_ok=True)
    import vortex as vx

    print(f"old-reader: vortex-data {getattr(vx, '__version__', '?')}", file=sys.stderr)
    for f in sorted(src.glob("*.vortex")):
        proc = subprocess.run(
            [sys.executable, "-I", __file__, "--one", str(f), str(dst)],
            capture_output=True,
            text=True,
            env={"RUST_BACKTRACE": "0", **os.environ},
        )
        if proc.returncode == 0:
            print(f"  ok   {f.name} ({proc.stdout.strip()} rows)", file=sys.stderr)
        else:
            err = [line for line in proc.stderr.strip().splitlines() if line.strip()]
            # Keep the most informative line: the Python exception, or the first panic message.
            msg = next(
                (
                    line.strip()
                    for line in err
                    if line.startswith(
                        ("pyarrow.lib.", "ValueError", "RuntimeError", "Exception", "TypeError", "AttributeError")
                    )
                ),
                None,
            )
            if msg is None:
                for i, line in enumerate(err):
                    if "panicked at" in line and i + 1 < len(err):
                        msg = err[i + 1].strip()
                        break
            if msg is None:
                msg = err[-1].strip() if err else f"exit {proc.returncode}"
            if proc.returncode < 0 or proc.returncode > 128:
                msg = f"reader aborted (exit {proc.returncode}): {msg}"
            (dst / f"{f.name}.error").write_text(msg)
            print(f"  FAIL {f.name}: {msg}", file=sys.stderr)


main()
