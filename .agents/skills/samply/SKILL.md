---
name: samply
description: Iterate on Vortex performance with Samply and benchmark evidence. Use when profiling or optimizing Vortex benchmarks, especially vx-bench or direct benchmark-binary runs, when analyzing profile.json.gz Firefox profiler output, when comparing benchmark runs, or when investigating performance regressions and hotspots.
---

# Samply

## Overview

Use Samply profiles and narrow benchmark runs to find, validate, and improve Vortex hot paths.
Keep the loop evidence-driven: benchmark first, profile the exact slow target, make one scoped
change, rerun the same target, and only then broaden verification.

## Share Evidence Early

Do not disappear into profile spelunking while useful output is already available. As soon as a
benchmark finishes, report the comparison table or timing lines to the user before starting deeper
analysis. As soon as a profile summary is available, report the top threads/functions/stacks before
reading more code. Then continue investigating with those facts visible.

For long performance sessions, use this cadence:

1. benchmark command starts: say what target, format, query, and toggles are being measured;
2. benchmark command finishes: immediately show run ID, timing comparison, and output path;
3. profile command finishes: immediately show profile path and first stack/function summary;
4. deeper analysis begins: state the concrete hotspot or hypothesis being checked.

## Standard Loop

1. Check branch state and changed surface:

   ```bash
   git status --short
   git branch --show-current
   git diff --stat develop...HEAD
   ```

2. Run a focused benchmark before profiling. Prefer one query, one engine, one format, and enough
   iterations to smooth obvious noise. If the experiment has a runtime env toggle, prefix every
   benchmark and profile command with the same env setting; this is often faster than recompiling
   and makes A/B comparisons clearer.

   ```bash
   FEATURE_TOGGLE=1 UV_CACHE_DIR=/private/tmp/vortex-uv-cache \
     uv run --project bench-orchestrator vx-bench run <benchmark> \
     -e <engine> \
     -f <format> \
     -q <query> \
     -i 5 \
     -l <label> \
     --output /private/tmp/<label>.jsonl \
     --verbose
   ```

   Useful variants:

   ```bash
   # Compare two formats in the same run.
   FEATURE_TOGGLE=1 UV_CACHE_DIR=/private/tmp/vortex-uv-cache \
     uv run --project bench-orchestrator vx-bench run <benchmark> \
     -e <engine> -f <baseline-format>,<candidate-format> -q <query> -i 5 -l <label>

   # Reuse already-built benchmark binaries when only rerunning the command.
   FEATURE_TOGGLE=1 UV_CACHE_DIR=/private/tmp/vortex-uv-cache \
     uv run --project bench-orchestrator vx-bench run <benchmark> \
     -e <engine> -f <format> -q <query> -i 5 -l <label> --no-build
   ```

3. Record a focused Samply profile. `vx-bench --samply` works, but direct `samply record` gives
   control over output paths and prevents the browser UI from opening:

   ```bash
   FEATURE_TOGGLE=1 samply record --save-only --unstable-presymbolicate --rate 1000 \
     --output /private/tmp/<label>.profile.json.gz \
     -- target/release_debug/<benchmark-binary> <benchmark> \
       --display-format gh-json \
       --iterations 5 \
       --hide-progress-bar \
       --formats <format> \
       --queries <query>
   ```

   Put environment assignments before `samply record`, as shown above. Do not put them after the
   `--` separator; everything after `--` is the command Samply launches and profiles. Profiling
   through a system helper such as `env`, `sleep`, `/bin/true`, or system Python is a bad sanity
   check on macOS because signed system executables can block Samply's task-port handoff. Use a
   locally built benchmark binary directly, for example:

   ```bash
   samply record --save-only --output /private/tmp/samply-help.profile.json.gz \
     -- target/release_debug/datafusion-bench --help
   ```

   In Codex on macOS, `Encountered an error during profiling: Unknown(1100)` usually means Samply
   was blocked by the sandbox before the profiled command started. Rerun the same `samply record`
   command with escalated permissions instead of debugging the benchmark or changing the query.

   `bench-orchestrator` builds `target/release_debug/datafusion-bench` by default with
   `RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes"` and the
   `unstable_encodings` feature. That profile has debug info and frame pointers, which is usually
   better for Samply than an arbitrary release binary.

4. Summarize the profile without opening Firefox Profiler. Run a small summary first and report it
   immediately, then run a wider summary only if needed:

   ```bash
   python3 .agents/skills/samply/scripts/profile_summary.py \
     /private/tmp/<label>.profile.json.gz \
     --binary target/release_debug/<benchmark-binary> \
     --symbolicate \
     --weight-mode cpu \
     --top 12 \
     --threads 2 \
     --stacks 4 \
     --stack-depth 10
   ```

   Wider follow-up:

   ```bash
   python3 .agents/skills/samply/scripts/profile_summary.py \
     /private/tmp/<label>.profile.json.gz \
     --binary target/release_debug/<benchmark-binary> \
     --symbolicate \
     --weight-mode cpu \
     --top 30 \
     --threads 6 \
     --stacks 12
   ```

5. Inspect code near the actual hot path. Load the benchmark query text or workload definition
   when it matters; do not rely on memory for the query shape.

6. Make one narrow change, rerun the focused benchmark/profile, and record the before/after command
   lines and results. Do not update expected results or broaden the benchmark until the narrow
   target explains the change.

## Samply JSON Schema

Samply writes Firefox-profiler JSON, often compressed as `profile.json.gz`.

- Top-level keys commonly include `meta`, `libs`, `threads`, `pages`, `counters`, and
  `profilerOverhead`.
- `meta.product` names the process, `meta.interval` is the sampling interval in milliseconds, and
  `meta.startTime` is an epoch timestamp in milliseconds.
- `libs[]` records loaded binaries with `name`, `path`, `debugPath`, `codeId`, `breakpadId`, and
  `arch`. Use this to verify symbol files still match the profile.
- Each `threads[]` entry has `name`, `tid`, `samples`, `stackTable`, `frameTable`, `funcTable`,
  `resourceTable`, and `stringArray`.
- `samples.length` is the number of stored sample rows. `samples.stack[]` points into
  `stackTable`. `samples.weight[]` is the number of collapsed samples represented by that row; use
  weight instead of row count when present. `samples.threadCPUDelta[]` is per-thread CPU delta in
  microseconds when present.
- `stackTable` is a linked list: `stackTable.frame[i]` is the current frame and
  `stackTable.prefix[i]` points to the caller stack. Follow prefixes to `null` and reverse to get
  root-to-leaf order.
- `frameTable.func[frame]` points into `funcTable`; `funcTable.name[func]` points into
  `stringArray`. `funcTable.resource[func]` points into `resourceTable`, whose `name` or `lib`
  fields point back into `stringArray`.

## Symbolication

If function names are raw addresses such as `0x3db28a0`, the profile is not symbolicated. Before
using `atos`, verify the binary UUID matches the profile:

```bash
gzip -cd profile.json.gz | jq '.libs[] | select(.name=="datafusion-bench") | {path, codeId, breakpadId}'
dwarfdump --uuid target/release_debug/datafusion-bench
```

If the UUID/code ID does not match, do not trust symbol names from the current binary. Re-profile,
or keep the exact binary plus the `.syms.json` sidecar emitted by `--unstable-presymbolicate`.

On macOS, Samply stores app addresses as offsets. Add the Mach-O text load address when using
`atos` manually:

```bash
atos -o target/release_debug/datafusion-bench -l 0x100000000 0x103db28a0
```

For a raw offset `0x3db28a0`, the address passed to `atos` is `0x100000000 + 0x3db28a0`.

## Reading Profiles

- Treat the main thread as orchestration unless its CPU delta is high. In DataFusion runs, useful
  CPU time is usually on `tokio-rt-worker` threads.
- Sort threads by total sample weight and CPU delta. Many idle worker threads can have high wall
  time but near-zero CPU.
- A stack with many samples may mean the operation is slow, or it may mean it is called many times.
  Samply alone usually cannot distinguish those. Pair hot stacks with counters, metrics, or logs:
  operation count, byte count, rows decoded, cache hits/misses, per-operation max/median duration,
  and lock wait/hold time.
- Prefer inclusive stacks to understand which subsystem owns time, then self frames to find tight
  loops.
- Look for repeated work: schema/dtype cloning, filter evaluation, decompression, canonicalization,
  pruning, segment reads, allocation, string parsing, and DataFusion physical-expression overhead.
- For I/O stacks, check `datafusion-bench --show-metrics` before assuming contention. Compare
  `vortex.io.read.duration_count`, `vortex.io.read.total_size`, `io.requests.individual`,
  `io.requests.coalesced`, segment cache misses, and max read duration against the baseline.
- If profile output only shows addresses and symbolication is blocked by a UUID mismatch, you can
  still use thread CPU, stack repetition, and library ownership, but re-profile before making a
  code-level claim.

## Reporting

Summaries should include:

- benchmark command and profile command;
- branch and binary profile used;
- before/after timings or run IDs;
- top hot threads/functions/stacks;
- confirmed facts versus inferences;
- checks run and checks skipped.
