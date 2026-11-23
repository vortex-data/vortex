# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Run an existing benchmark against the shared RAM-backed S3 endpoint."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment(endpoint: str) -> dict[str, str]:
    # Each child has dummy credentials; no ambient profile, session, web identity or AWS endpoint leaks in.
    env = {key: value for key, value in os.environ.items() if not key.startswith("AWS_")}
    env.update(
        AWS_ACCESS_KEY_ID="benchmark",
        AWS_SECRET_ACCESS_KEY="benchmark",
        AWS_REGION="us-east-1",
        AWS_DEFAULT_REGION="us-east-1",
        AWS_EC2_METADATA_DISABLED="true",
        AWS_ALLOW_HTTP="true",
        AWS_VIRTUAL_HOSTED_STYLE_REQUEST="false",
        VORTEX_BENCH_S3_SIMULATED="1",
    )
    for key in ("AWS_ENDPOINT", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "AWS_S3_ENDPOINT"):
        env[key] = endpoint
    bypass = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
    env["NO_PROXY"] = env["no_proxy"] = ",".join(filter(None, [bypass, "127.0.0.1", "localhost", "::1"]))
    return env


def command(arguments: list[str], kind: str, bucket: str) -> list[str]:
    if not arguments:
        raise ValueError("Pass a benchmark command after --")
    if any("remote-data-dir" in arg for arg in arguments):
        raise ValueError("The simulator supplies remote-data-dir; remove the conflicting option")
    uri = f"s3://{bucket}/"
    if kind == "sql":
        return [*arguments, "--opt", f"remote-data-dir={uri}"]
    if kind == "probe":
        return [*arguments, "--source", uri]
    return [*arguments, "--remote-data-dir", uri]


def stop(process: subprocess.Popen, group: bool = False):
    if process.poll() is not None:
        return
    if group:
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if group:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=10)


def label_results(paths: list[Path], profile_sha256: str | None):
    # A profile update starts a new baseline, including for random-access rows without storage metadata.
    storage = f"s3-simulated-{profile_sha256}" if profile_sha256 else "s3-simulated-zero-delay"
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"No benchmark results in {path}")
        for row in rows:
            if row.get("kind") == "query_measurement":
                # Postgres represents the S3 protocol with an enum; dataset_variant separates the model.
                row["storage"] = "s3"
                row["dataset_variant"] = "/".join(filter(None, [row.get("dataset_variant"), storage]))
            elif row.get("kind"):
                dataset, separator, pattern = row["dataset"].partition("/")
                row["dataset"] = f"{dataset}@{storage}{separator}{pattern}"
            else:
                row["storage"] = storage
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise ValueError("Output manifest already exists; choose a fresh path")
    if args.kind != "compress" and args.data_root is None:
        raise ValueError("SQL/random-access require --data-root with prepared files")
    if not 1 <= args.timeout <= 86400:
        raise ValueError("Timeout must be between one second and one day")
    cmd = command(args.command, args.kind, args.bucket)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "storage": "s3-simulated",
        "kind": args.kind,
        "command": cmd,
        "data_root": str(args.data_root.resolve()) if args.data_root else None,
        "profile": str(args.profile.resolve()) if args.profile else None,
        "profile_sha256": digest(args.profile) if args.profile else None,
        "zero_delay": args.zero_delay,
        "server_sha256": digest(args.server),
        "ci_run_id": os.environ.get("GITHUB_RUN_ID"),
        "ci_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "instance_id": os.environ.get("RUNS_ON_INSTANCE_ID"),
        "started_unix_ms": round(time.time() * 1000),
        "complete": False,
        "metrics_scope": "whole command, including preparation, registration and warmups",
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    server, child = None, None
    with tempfile.TemporaryDirectory(prefix="s3-simulated-") as temporary:
        ready = Path(temporary) / "ready.json"
        server_command = [str(args.server.resolve()), "--ready-file", str(ready), "--bucket", args.bucket]
        if args.profile:
            server_command.extend(["--profile", str(args.profile.resolve())])
        else:
            server_command.append("--zero-delay")
        if args.data_root:
            server_command.extend(["--root", str(args.data_root.resolve())])
        if args.max_resident_bytes:
            server_command.extend(["--max-resident-bytes", str(args.max_resident_bytes)])
        try:
            server = subprocess.Popen(server_command)
            deadline = time.monotonic() + 300
            while not ready.exists():
                if server.poll() is not None:
                    raise RuntimeError(f"S3 simulator exited with {server.returncode} before readiness")
                if time.monotonic() >= deadline:
                    raise TimeoutError("RAM snapshot did not become ready within five minutes")
                time.sleep(0.05)
            manifest["server"] = json.loads(ready.read_text())
            endpoint = manifest["server"]["endpoint"]
            child = subprocess.Popen(cmd, env=environment(endpoint), start_new_session=True)
            manifest["exit_code"] = child.wait(timeout=args.timeout)
            with urllib.request.urlopen(endpoint + "/_vortex/metrics", timeout=10) as response:
                manifest["metrics"] = json.load(response)
            if args.profile and digest(args.profile) != manifest["profile_sha256"]:
                raise ValueError("Profile changed during the benchmark")
            if manifest["metrics"]["get"] == 0:
                raise ValueError("Benchmark completed without reading from simulated S3")
            if child.returncode == 0:
                label_results(args.results, manifest["profile_sha256"])
            manifest["complete"] = child.returncode == 0
            return child.returncode
        except BaseException as error:
            manifest["error"] = str(error)
            raise
        finally:
            if child is not None:
                stop(child, group=True)
            if server is not None:
                stop(server)
            manifest["finished_unix_ms"] = round(time.time() * 1000)
            args.output.write_text(json.dumps(manifest, indent=2) + "\n")
            metrics = manifest.get("metrics", {})
            args.output.with_suffix(".md").write_text(
                "\nSimulated S3: RAM-backed HTTP endpoint, frozen request-derived profile "
                f"`{manifest['profile_sha256'] or 'zero-delay'}`. "
                f"GETs: {metrics.get('get', 'unavailable')}; "
                f"bytes read: {metrics.get('bytes', 'unavailable')}; "
                f"peak in-flight: {metrics.get('peak_inflight', 'unavailable')}. "
                "Counters cover the complete command, including registration and warmups.\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--profile", type=Path)
    model.add_argument("--zero-delay", action="store_true", help="Explicit loopback/client overhead baseline")
    parser.add_argument("--server", type=Path, default=Path("target/release_debug/s3-simulate"))
    parser.add_argument("--data-root", type=Path, help="Prepared dataset root, mirrored verbatim at the bucket root")
    parser.add_argument("--kind", choices=("sql", "random-access", "compress", "probe"), required=True)
    parser.add_argument("--bucket", default="vortex-bench")
    parser.add_argument("--max-resident-bytes", type=int)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--output", type=Path, required=True, help="Separate simulation provenance/metrics manifest")
    parser.add_argument(
        "--results",
        type=Path,
        action="append",
        default=[],
        help="JSONL results to label with the frozen profile identity",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
