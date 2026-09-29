#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Minimize disjoint corpus shards in parallel, then merge their coverage union."""

import argparse
import asyncio
import os
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Merge:
    name: str
    inputs: list[Path]
    directory: Path
    total: int
    processed: int = 0
    offset: int = 0
    started: float = 0
    finished: bool = False

    @property
    def output(self) -> Path:
        return self.directory / "output"

    @property
    def control(self) -> Path:
        return self.directory / "merge.txt"

    def read_progress(self) -> None:
        if not self.control.exists():
            return
        # Follow complete records only. libFuzzer flushes COV after each input;
        # its console pulses become very infrequent on million-input corpora.
        with self.control.open("rb") as stream:
            stream.seek(self.offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break
                self.processed += line.startswith(b"COV ")
                self.offset = stream.tell()

    def report(self) -> None:
        self.read_progress()
        elapsed = time.monotonic() - self.started
        rate = self.processed / elapsed if elapsed else 0
        remaining = max(0, self.total - self.processed)
        eta = f"{remaining / rate:.0f}s" if rate else "unknown"
        print(
            f"{self.name}: {self.processed}/{self.total} inputs, "
            f"{rate:.1f} inputs/s, elapsed {elapsed:.0f}s, ETA {eta}",
            flush=True,
        )


def corpus_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def shard_corpus(files: list[Path], directory: Path, shards: int) -> list[Merge]:
    merges = []
    for index in range(min(shards, len(files))):
        shard = directory / f"shard-{index}"
        inputs = shard / "inputs"
        inputs.mkdir(parents=True)
        merges.append(Merge(f"shard-{index}", [inputs], shard, 0))
    # Round-robin by size spreads the largest inputs across workers. Hard links
    # avoid duplicating the corpus on disk and leave the original untouched.
    for index, path in enumerate(sorted(files, key=lambda path: (path.stat().st_size, path))):
        merge = merges[index % len(merges)]
        os.link(path, merge.inputs[0] / str(index))
        merge.total += 1
    return merges


async def run_merge(merge: Merge, binary: Path, artifacts: Path, env: dict[str, str]) -> bool:
    merge.output.mkdir(parents=True)
    merge.started = time.monotonic()
    print(f"Starting {merge.name}: {merge.total} inputs", flush=True)
    process = None
    failed_input = False
    try:
        process = await asyncio.create_subprocess_exec(
            str(binary),
            "-merge=1",
            str(merge.output),
            *(str(path) for path in merge.inputs),
            f"-merge_control_file={merge.control}",
            f"-artifact_prefix={artifacts}/",
            "-rss_limit_mb=0",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            start_new_session=True,
            # Crash reports may contain very long lines (e.g. debug array dumps).
            limit=16 * 1024 * 1024,
        )
        assert process.stdout is not None
        with (merge.directory / "merge.log").open("wb") as log:
            async for line in process.stdout:
                log.write(line)
                if b"caused a failure" in line or b"will be skipped as unlucky" in line:
                    failed_input = True
                if b"MERGE-OUTER:" in line or b"ERROR:" in line:
                    print(f"[{merge.name}] {line.decode(errors='replace').rstrip()}", flush=True)
        status = await process.wait()
        merge.report()
        if status != 0 or failed_input or merge.processed != merge.total:
            print(f"::error::{merge.name} failed or skipped inputs; see {merge.directory / 'merge.log'}")
            return False
        return True
    finally:
        if process is not None and process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        merge.finished = True


async def run_phase(merges: list[Merge], binary: Path, artifacts: Path, env: dict[str, str]) -> bool:
    tasks = [asyncio.create_task(run_merge(merge, binary, artifacts, env)) for merge in merges]
    try:
        pending = set(tasks)
        while pending:
            _, pending = await asyncio.wait(pending, timeout=30)
            for merge in merges:
                if not merge.finished:
                    merge.report()
        return all(await asyncio.gather(*tasks))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def minimize(binary: Path, corpus: Path, work_dir: Path, artifacts: Path, shards: int) -> bool:
    files = corpus_files(corpus)
    if not files:
        print("Corpus is empty; nothing to minimize")
        return True
    started = time.monotonic()
    original_bytes = sum(path.stat().st_size for path in files)
    # Merge control files are specific to the binary and input snapshot. Each
    # invocation starts fresh; no state from an interrupted run is reused.
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for assignment in os.environ.get("EXTRA_ENV", "").split():
        key, value = assignment.split("=", 1)
        env[key] = value
    merges = shard_corpus(files, work_dir, shards)
    if not await run_phase(merges, binary, artifacts, env):
        return False
    if len(merges) > 1:
        inputs = [merge.output for merge in merges]
        final = Merge("final", inputs, work_dir / "final", sum(len(corpus_files(path)) for path in inputs))
        if final.total == 0 or not await run_phase([final], binary, artifacts, env):
            print("::error::Final merge failed or all shards were empty; retaining original corpus")
            return False
    else:
        final = merges[0]
    minimized = corpus_files(final.output)
    if not minimized:
        print("::error::Refusing to replace a non-empty corpus with an empty corpus")
        return False
    minimized_bytes = sum(path.stat().st_size for path in minimized)
    summary = (
        f"Minimized {len(files)} inputs ({original_bytes} bytes) to {len(minimized)} "
        f"({minimized_bytes} bytes) in {time.monotonic() - started:.0f}s using {len(merges)} shards"
    )
    print(summary, flush=True)
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a") as output:
            output.write(f"### {corpus.name} corpus minimization\n\n{summary}\n")
    # Only publish a fully completed, crash-free merge. The remote update still
    # uses the original ETag in checkpoint-fuzz-corpus.sh.
    shutil.rmtree(corpus)
    final.output.rename(corpus)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards must be positive")
    if not args.binary.is_file() or not args.corpus.is_dir():
        parser.error("the compiled binary and corpus directory must exist")
    success = asyncio.run(minimize(args.binary.resolve(), args.corpus, args.work_dir, args.artifacts, args.shards))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
