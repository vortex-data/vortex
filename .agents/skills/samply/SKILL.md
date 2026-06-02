---
name: samply
description: Analyze Samply Firefox-profiler output, record focused profiles, summarize hot threads/stacks, inspect symbolication, and compare profile evidence before and after a performance change.
---

# Samply

## Overview

Use this skill when a task involves Samply recordings or Firefox-profiler JSON, especially
`profile.json.gz` files, `samply record`, `samply load`, symbolication, thread timeline skew, or
hot stack interpretation. Keep the loop evidence-driven: establish a focused baseline, record the
exact target, summarize the profile before deep code reading, make one scoped change, and rerun the
same target.

This skill is intentionally project-agnostic. Any project-specific benchmark harness, environment
variables, metrics, or logging commands should live in a separate benchmark skill or in the current
task context.

## Share Evidence Early

Do not disappear into profile spelunking while useful output is already available. As soon as a
timing run finishes, report the timing lines or comparison table before starting deeper analysis.
As soon as a profile summary is available, report the top threads/functions/stacks before reading
more code. Then continue investigating with those facts visible.

For long performance sessions, use this cadence:

1. timing command starts: say what target, input, mode, and runtime toggles are being measured;
2. timing command finishes: immediately show timing results and output path;
3. profile command finishes: immediately show profile path, a `samply load` command the user can
   run to inspect it in Firefox Profiler, and the first stack/function summary;
4. deeper analysis begins: state the concrete hotspot or hypothesis being checked.

## Standard Loop

1. Check branch state and changed surface when working in a repository:

   ```bash
   git status --short
   git branch --show-current
   git diff --stat
   ```

2. Run a focused timing command before profiling. Prefer one executable, one workload/input, one
   mode, and enough iterations to smooth obvious noise. If the experiment has a runtime
   environment toggle, prefix every timing and profile command with the same env setting; this is
   often faster than recompiling and makes A/B comparisons clearer.

   Generic shape:

   ```bash
   FEATURE_TOGGLE=1 <timing-command> --iterations 5 --output /tmp/<label>.jsonl
   ```

   If a project has an existing benchmark harness, use that harness for the timing baseline and
   copy its exact target arguments into the profiled command.

3. Record a focused Samply profile. Prefer recording without `--unstable-presymbolicate` first,
   then symbolicate offline with the scripts below. This avoids chasing misleading
   pre-symbolicated stacks when unwinding or symbol lookup gets confused.

   ```bash
   FEATURE_TOGGLE=1 samply record --save-only --rate 1000 \
     --output /tmp/<label>.profile.json.gz \
     -- /absolute/path/to/<binary> <args>
   ```

   Put environment assignments before `samply record`, as shown above. Do not put them after the
   `--` separator; everything after `--` is the command Samply launches and profiles.

   On macOS, profiling through a system helper such as `env`, `sleep`, `/bin/true`, or system
   Python can be a bad sanity check because signed system executables may block Samply's task-port
   handoff. Prefer a locally built binary or a user-owned executable.

   In a sandboxed agent environment on macOS, `Encountered an error during profiling:
   Unknown(1100)` usually means Samply was blocked before the profiled command started. Rerun the
   same `samply record` command with the required execution permissions instead of changing the
   workload.

   Use a profile with debug information when stack quality matters. If the symbols or unwinding
   look suspect, rebuild with the project's highest-quality profiling/debug-symbol profile and
   record again.

   If a profile shows impossible-looking ancestry, such as hot execution frames nested under
   unrelated `Drop::drop` frames or otherwise nonsensical async stacks, do not trust the stack
   summary. First verify the binary UUID matches the profile, remove presymbolication from the
   recording command, and rebuild with better debug symbols if needed.

   After the profile is recorded, immediately show the user the command to open it themselves:

   ```bash
   samply load /tmp/<label>.profile.json.gz
   ```

   `samply load` starts a local Firefox Profiler server and opens the browser UI. If the user only
   wants the URL or the environment cannot open a browser, use:

   ```bash
   samply load --no-open /tmp/<label>.profile.json.gz
   ```

   Then report the printed local URL.

4. Summarize the profile without opening Firefox Profiler. Run a small summary first and report it
   immediately, then run a wider summary only if needed:

   ```bash
   python3 .agents/skills/samply/scripts/profile_summary.py \
     /tmp/<label>.profile.json.gz \
     --binary /absolute/path/to/<binary> \
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
     /tmp/<label>.profile.json.gz \
     --binary /absolute/path/to/<binary> \
     --symbolicate \
     --weight-mode cpu \
     --top 30 \
     --threads 6 \
     --stacks 12
   ```

   For timeline skew, quantify worker occupancy instead of relying only on the visual timeline:

   ```bash
   python3 .agents/skills/samply/scripts/profile_activity.py \
     /tmp/<label>.profile.json.gz \
     --thread-regex '<worker-thread-regex>' \
     --bin-ms 10
   ```

   For a focused inverted call tree over a thread class or time range, use:

   ```bash
   python3 .agents/skills/samply/scripts/profile_inverted_tree.py \
     /tmp/<label>.profile.json.gz \
     --binary /absolute/path/to/<binary> \
     --symbolicate \
     --thread-regex '<worker-thread-regex>' \
     --start-ms <start> \
     --end-ms <end> \
     --contains '<frame-regex>'
   ```

5. Inspect code near the actual hot path. Load the workload definition when it matters; do not rely
   on memory for the workload shape.

6. Make one narrow change, rerun the focused timing/profile command, and record the before/after
   command lines and results. Do not broaden the workload until the narrow target explains the
   change.

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
using `atos`, verify the binary UUID/code ID matches the profile:

```bash
gzip -cd /tmp/<label>.profile.json.gz | jq '.libs[] | {name, path, codeId, breakpadId}'
dwarfdump --uuid /absolute/path/to/<binary>
```

If the UUID/code ID does not match, do not trust symbol names from the current binary. Re-profile,
or keep the exact binary plus any symbol sidecar emitted by the recorder.

On macOS, Samply stores app addresses as offsets. Add the Mach-O text load address when using
`atos` manually:

```bash
atos -o /absolute/path/to/<binary> -l 0x100000000 0x103db28a0
```

For a raw offset `0x3db28a0`, the address passed to `atos` is `0x100000000 + 0x3db28a0`.

When using the bundled scripts, pass `--symbol-lib <library-name>` if the binary name in
`profile.libs[]` differs from the basename of `--binary`.

## Reading Profiles

- Treat the main thread as orchestration unless its CPU delta is high. Useful CPU time is often on
  worker threads, but thread naming is runtime-specific.
- Sort threads by total sample weight and CPU delta. Many idle worker threads can have high wall
  time but near-zero CPU.
- Use `profile_activity.py` when the Firefox Profiler timeline shows empty space. Good parallel
  traces keep worker occupancy high through the timed region; a low-occupancy tail points to
  scheduling skew, stragglers, dependency ordering, partition imbalance, blocking, or insufficient
  work admission.
- Use `profile_inverted_tree.py` with `--contains` for allocation frames, blocking frames, or a hot
  leaf function to see the caller contexts that produce the samples.
- A stack with many samples may mean the operation is slow, or it may mean it is called many times.
  Samply alone usually cannot distinguish those. Pair hot stacks with counters, metrics, or logs:
  operation count, byte count, rows/items processed, cache hits/misses, per-operation max/median
  duration, and lock wait/hold time.
- Prefer inclusive stacks to understand which subsystem owns time, then self frames to find tight
  loops.
- Look for repeated work: allocation, parsing, cloning, serialization/deserialization, expression
  evaluation, decompression, canonicalization, redundant I/O, and synchronization overhead.
- If profile output only shows addresses and symbolication is blocked by a UUID mismatch, you can
  still use thread CPU, stack repetition, and library ownership, but re-profile before making a
  code-level claim.

## Reporting

Summaries should include:

- timing command and profile command;
- command for the user to open the profile, usually `samply load <profile.json.gz>`;
- branch, binary, and profile used;
- before/after timings or run IDs;
- top hot threads/functions/stacks;
- confirmed facts versus inferences;
- checks run and checks skipped.
