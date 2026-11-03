# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""
Compare divan micro-benchmarks between two checkouts of the repository.

`build` compiles the requested bench targets in the current checkout and copies the
executables into an output directory. `run` executes the base and head executables on the
same machine in alternating rounds, parses divan's output and writes a Markdown report.

A benchmark spec is `<crate>`, `<crate>/<bench>` or `<crate>/<bench>/<filter>`. A bare
crate selects all of its bench targets. The filter is passed to divan, which matches it as a
regex against benchmark names. Specs are separated by commas or whitespace.
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
UNITS_NS = {"ps": 1e-3, "ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}
VALUE_RE = re.compile(r"^(?P<num>[\d.]+)\s*(?P<unit>ps|ns|µs|us|ms|s)$")
TREE_CHARS = "│├╰─ "


@dataclass(frozen=True)
class Spec:
    package: str
    bench: str | None
    filter: str | None


def parse_specs(raw: str) -> list[Spec]:
    specs = []
    for item in re.split(r"[,\s]+", raw.strip()):
        if not item:
            continue
        parts = item.split("/", 2)
        specs.append(Spec(parts[0], parts[1] if len(parts) > 1 else None, parts[2] if len(parts) > 2 else None))
    if not specs:
        raise SystemExit("no benchmarks given")
    return specs


def bench_targets(spec: Spec) -> list[tuple[str, list[str]]]:
    """Resolve a spec to (bench name, required features) pairs using cargo metadata."""
    metadata = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    package = next((p for p in metadata["packages"] if p["name"] == spec.package), None)
    if package is None:
        return []
    targets = [
        (t["name"], t.get("required-features", []))
        for t in package["targets"]
        if "bench" in t["kind"] and (spec.bench is None or t["name"] == spec.bench)
    ]
    return targets


def build(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    failed = False
    for spec in parse_specs(args.benchmarks):
        targets = bench_targets(spec)
        if not targets:
            print(f"::warning::{spec.package}/{spec.bench or '*'} not found in this checkout", file=sys.stderr)
            continue
        for bench, features in targets:
            cmd = [
                "cargo",
                "bench",
                "--locked",
                "--no-run",
                "--message-format=json",
                "-p",
                spec.package,
                "--bench",
                bench,
            ]
            if features:
                cmd += ["--features", ",".join(features)]
            print("+", " ".join(cmd), file=sys.stderr, flush=True)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                print(f"::warning::failed to build {spec.package}/{bench}", file=sys.stderr)
                failed = True
                continue
            for line in proc.stdout.splitlines():
                msg = json.loads(line)
                if (
                    msg.get("reason") == "compiler-artifact"
                    and msg.get("executable")
                    and "bench" in msg["target"]["kind"]
                ):
                    dest = out / f"{spec.package}--{bench}"
                    shutil.copy2(msg["executable"], dest)
                    manifest[f"{spec.package}/{bench}"] = dest.name
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if failed and args.strict:
        raise SystemExit(1)


def parse_divan(output: str) -> dict[str, float]:
    """Parse divan's tree table into {benchmark path: median ns}."""
    results = {}
    stack: list[str] = []
    # Column separators sit at fixed offsets given by the header row. The tree drawing on the
    # left uses the same `│` character, so the line cannot simply be split on it.
    seps: list[int] = []
    for line in output.splitlines():
        if "fastest" in line and "median" in line and "│" in line:
            seps = [i for i, c in enumerate(line) if c == "│"]
            stack = []
            continue
        if not seps or len(line) <= seps[0] or line[seps[0]] != "│":
            continue
        first = line[: seps[0]]
        stripped = first.lstrip(TREE_CHARS)
        depth = (len(first) - len(stripped)) // 3
        tokens = stripped.rstrip().rsplit(None, 2)
        name = stripped.strip()
        if len(tokens) == 3 and VALUE_RE.match(f"{tokens[1]} {tokens[2]}"):
            name = tokens[0]
        del stack[depth - 1 :]
        stack.append(name)
        if len(seps) > 2:
            median = VALUE_RE.match(line[seps[1] + 1 : seps[2]].strip())
            if median:
                results["::".join(stack)] = float(median["num"]) * UNITS_NS[median["unit"]]
    return results


def run_one(exe: Path, filter: str | None, extra: list[str], pin: bool) -> dict[str, float]:
    cmd = [str(exe), "--bench", *extra]
    if filter:
        cmd.append(filter)
    if pin:
        cmd = ["bash", str(SCRIPT_DIR / "bench-taskset.sh"), *cmd]
    print("+", " ".join(cmd), file=sys.stderr, flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, env={**os.environ, "RUST_BACKTRACE": "1"})
    sys.stderr.write(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"{exe.name} exited with {proc.returncode}")
    return parse_divan(proc.stdout)


def fmt_ns(ns: float) -> str:
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.3g} {unit}"
    return f"{ns:.3g} ns"


def run(args: argparse.Namespace) -> None:
    dirs = {"base": Path(args.base), "head": Path(args.head)}
    manifests = {side: json.loads((d / "manifest.json").read_text()) for side, d in dirs.items()}
    extra = args.divan_args.split()
    # samples[side][benchmark] -> one median per round
    samples: dict[str, dict[str, list[float]]] = {"base": {}, "head": {}}
    errors = []

    for spec in parse_specs(args.benchmarks):
        keys = sorted(
            k
            for k in manifests["head"].keys() | manifests["base"].keys()
            if k.split("/")[0] == spec.package and (spec.bench is None or k.split("/")[1] == spec.bench)
        )
        for key in keys:
            for round_idx in range(args.rounds):
                # Alternate which side runs first so drift over time affects both equally.
                order = ["base", "head"] if round_idx % 2 == 0 else ["head", "base"]
                for side in order:
                    if key not in manifests[side]:
                        continue
                    try:
                        parsed = run_one(dirs[side] / manifests[side][key], spec.filter, extra, args.pin)
                    except RuntimeError as e:
                        errors.append(f"{side} `{key}`: {e}")
                        continue
                    for name, ns in parsed.items():
                        samples[side].setdefault(f"{key}::{name}", []).append(ns)

    summary = {side: {k: statistics.median(v) for k, v in s.items()} for side, s in samples.items()}
    Path(args.json).write_text(json.dumps({"rounds": args.rounds, "samples": samples, "median_ns": summary}, indent=2))
    Path(args.report).write_text(report(summary, args, errors))
    if errors:
        raise SystemExit("\n".join(errors))


def report(summary: dict[str, dict[str, float]], args: argparse.Namespace, errors: list[str]) -> str:
    base, head = summary["base"], summary["head"]
    threshold = args.threshold / 100
    rows = []
    ratios = []
    for name in sorted(base.keys() | head.keys()):
        b, h = base.get(name), head.get(name)
        if b is None or h is None:
            status = "🆕 head only" if b is None else "🗑️ base only"
            rows.append((name, fmt_ns(b) if b else "—", fmt_ns(h) if h else "—", "—", status))
            continue
        ratio = h / b
        ratios.append(ratio)
        if ratio > 1 + threshold:
            status = "🔴 slower"
        elif ratio < 1 - threshold:
            status = "🟢 faster"
        else:
            status = "⚪"
        rows.append((name, fmt_ns(b), fmt_ns(h), f"{(ratio - 1) * 100:+.1f}%", status))

    lines = [
        f"# Benchmark comparison: `{args.head_label}` vs `{args.base_label}`",
        "",
        f"Benchmarks: `{args.benchmarks.strip()}` · rounds: {args.rounds} · threshold: ±{args.threshold:g}%",
        "",
    ]
    if ratios:
        geomean = statistics.geometric_mean(ratios)
        slower = sum(r > 1 + threshold for r in ratios)
        faster = sum(r < 1 - threshold for r in ratios)
        lines += [
            f"**Geomean change: {(geomean - 1) * 100:+.2f}%** · 🔴 {slower} slower · 🟢 {faster} faster · "
            f"⚪ {len(ratios) - slower - faster} unchanged",
            "",
        ]
    if errors:
        lines += ["**Errors:**", "", *[f"- {e}" for e in errors], ""]
    if not rows:
        lines.append("No benchmark results were parsed.")
        return "\n".join(lines) + "\n"
    changed = [r for r in rows if r[4] != "⚪"]
    table_header = ["| Benchmark | Base (median) | Head (median) | Change | |", "|---|---:|---:|---:|---|"]
    if changed:
        lines += [
            "## Changes",
            "",
            *table_header,
            *[f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in changed],
            "",
        ]
    lines += [
        "<details><summary>All results</summary>",
        "",
        *table_header,
        *[f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows],
        "",
        "</details>",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build bench executables of the current checkout")
    p_build.add_argument("--benchmarks", required=True)
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--strict", action="store_true", help="fail if any target fails to build")
    p_build.set_defaults(func=build)

    p_run = sub.add_parser("run", help="run base and head executables and write a report")
    p_run.add_argument("--benchmarks", required=True)
    p_run.add_argument("--base", required=True, help="directory written by `build` for the base")
    p_run.add_argument("--head", required=True, help="directory written by `build` for the head")
    p_run.add_argument("--rounds", type=int, default=3)
    p_run.add_argument("--threshold", type=float, default=5.0, help="percent change reported as a regression")
    p_run.add_argument("--divan-args", default="", help="extra arguments passed to every divan executable")
    p_run.add_argument("--pin", action="store_true", help="run under scripts/bench-taskset.sh")
    p_run.add_argument("--base-label", default="base")
    p_run.add_argument("--head-label", default="head")
    p_run.add_argument("--json", default="bench-compare.json")
    p_run.add_argument("--report", default="bench-compare.md")
    p_run.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
