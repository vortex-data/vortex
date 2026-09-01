# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""
Run compress-bench once per dataset, drop OS cache between datasets,
merge outputs.
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BINARY = "target/release_debug/compress-bench"
PARTS_DIR = Path("parts")

# TODO(myrrc): we should also drop CUDA allocator caches


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


def list_datasets(gpu_decompress: bool) -> list[str]:
    cmd = [BINARY, "--print-datasets"]
    if gpu_decompress:
        cmd.append("--gpu-decompress")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def run_datasets(
    formats: str, emit_ingest_records: bool, gpu_decompress: bool
) -> tuple[list[str], list[Path]]:
    shutil.rmtree(PARTS_DIR, ignore_errors=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    outputs: list[Path] = []
    for i, dataset in enumerate(list_datasets(gpu_decompress)):
        drop_os_caches()

        args = [
            "bash",
            str(SCRIPT_DIR / "bench-taskset.sh"),
            BINARY,
            "--datasets",
            f"^{re.escape(dataset)}$",
        ]
        if gpu_decompress:
            # GPU mode fixes its own formats
            args.append("--gpu-decompress")
        else:
            args += ["--formats", formats]
        output = PARTS_DIR / f"{i}.gh.json"
        args += ["-d", "gh-json", "-o", str(output)]
        if emit_ingest_records:
            args += ["--ingest-jsonl", str(PARTS_DIR / f"{i}.ingest.jsonl")]
        print("+", " ".join(args), flush=True)

        result = subprocess.run(args, check=False)
        if result.returncode != 0:
            failures.append(dataset)
        elif output.exists():
            outputs.append(output)
    return failures, outputs


def merge(paths: list[Path], out_path: str) -> None:
    lines: list[str] = []
    for path in sorted(paths):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    lines.append(line)
    Path(out_path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        default="arrow-ipc,parquet,vortex",
        help="comma-separated formats to forward to compress-bench",
    )
    parser.add_argument(
        "--emit-ingest-records",
        action="store_true",
        help="merge --ingest-jsonl records into results.ingest.jsonl",
    )
    parser.add_argument(
        "--gpu-decompress",
        action="store_true",
        help="run the GPU decompression suite (forwards --gpu-decompress, ignores --formats)",
    )
    args = parser.parse_args()

    failures, outputs = run_datasets(args.formats, args.emit_ingest_records, args.gpu_decompress)
    merge(outputs, "results.json")
    if args.emit_ingest_records:
        ingest_outputs = [
            path.with_name(path.name.replace(".gh.json", ".ingest.jsonl")) for path in outputs
        ]
        merge([path for path in ingest_outputs if path.exists()], "results.ingest.jsonl")

    if failures:
        print("Dropped failed datasets: " + ", ".join(failures), flush=True)


if __name__ == "__main__":
    main()
