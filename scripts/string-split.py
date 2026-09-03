# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""
Run string-bench once per (column, encoder), drop OS cache between runs, merge
outputs
"""

import glob
import re
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BINARY = "target/release_debug/string-bench"
PARTS_DIR = Path("parts")

COLUMNS = ["URL", "l_comment"]
ENCODERS = ["onpair", "fsst"]


def drop_os_caches() -> None:
    try:
        subprocess.run(["sync"], check=True)
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pass


def run_combinations() -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    i = 0
    for column in COLUMNS:
        for encoder in ENCODERS:
            drop_os_caches()

            args = [
                "bash",
                str(SCRIPT_DIR / "bench-taskset.sh"),
                BINARY,
                "--columns",
                f"^{re.escape(column)}$",
                "--encoders",
                encoder,
                "-d",
                "gh-json",
                "-o",
                str(PARTS_DIR / f"{i}.gh.json"),
            ]
            print("+", " ".join(args), flush=True)
            subprocess.run(args, check=True)
            i += 1


def merge(pattern: str, out_path: str) -> None:
    lines: list[str] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    lines.append(line)
    Path(out_path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def main() -> None:
    run_combinations()
    merge(f"{PARTS_DIR}/*.gh.json", "results.json")


if __name__ == "__main__":
    main()
