#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import os
import subprocess
import sys

COFF_MACHINES = {b"\x64\x86", b"\x4c\x01"}


def output_path(args: list[str]) -> str | None:
    for arg in args:
        if arg[:3].lower() in ("-fo", "/fo") and len(arg) > 3:
            return arg[3:]
    return None


def zero_timestamp(path: str):
    with open(path, "r+b") as obj:
        if obj.read(2) not in COFF_MACHINES:
            return
        obj.seek(4)
        obj.write(b"\x00\x00\x00\x00")


def main() -> int:
    args = sys.argv[1:]
    status = subprocess.run([os.environ["VORTEX_ML64"], *args]).returncode
    path = output_path(args)
    if status == 0 and path is not None:
        zero_timestamp(path)
    return status


if __name__ == "__main__":
    sys.exit(main())
