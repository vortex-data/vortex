# 02 — Stability & Repeatability of Microbenchmark Runs

> Slice owner bullet: **"Very stable and repeatable runs across machines so I can see perf
> changes."**
>
> This is the crux of the whole CodSpeed-replacement project. CodSpeed solves it by *not*
> measuring wall-clock time for its primary signal: it runs benches under a modified Valgrind
> (Callgrind) and counts CPU instructions / simulated cache events. Those counts are
> deterministic and machine-independent, so the same commit produces the same number on a
> laptop and on a noisy CI VM. Replacing CodSpeed means we must reproduce that property
> ourselves (or accept a noisier substitute). This document recommends how.

## 1. Where Vortex is today (ground truth)

Read directly from the repo:

### CI workflow — `.github/workflows/codspeed.yml`

- **CPU shards (8 of them)** run `cargo codspeed build ... --profile bench`, then
  `bash scripts/bench-taskset.sh cargo codspeed run` under `CodSpeedHQ/action` with
  **`mode: "simulation"`** (instruction counting via Valgrind/Callgrind). Build uses
  `RUSTFLAGS: -C target-feature=+avx2`. Runners are `runs-on` self-hosted `amd64-medium`
  images, tagged per-shard.
- **CUDA shards (3 of them)** run `cargo codspeed build -m walltime` and
  `CodSpeedHQ/action` with **`mode: "walltime"`** on `g5` GPU runners. Instruction counting
  is meaningless for GPU kernels, so wall-time is the only option there.
- Both call `scripts/setup-benchmark.sh` (as root) for runner tuning, and a `system-info`
  composite action that dumps `lscpu`, CPU flags, memory and disk info into the log.

### Runner tuning — `scripts/setup-benchmark.sh` (runs as root, exactly does)

1. **Disables turbo / frequency boost**: writes `1` to
   `/sys/devices/system/cpu/intel_pstate/no_turbo` and `0` to `.../cpufreq/boost` (best
   effort, `|| true`).
2. **Discourages swap**: `sysctl vm.swappiness=0` and `swapoff -a`.
3. **Disables NUMA balancing**: `sysctl kernel.numa_balancing=0`.
4. **Disables ASLR**: `sysctl kernel.randomize_va_space=0` — important for repeatable
   addresses / cache layout.
5. **Disables desktop autogroup scheduling**: `kernel.sched_autogroup_enabled=0`.
6. Quiets kernel logging (`dmesg -n 1`), stops/masks `apparmor`, `ModemManager`,
   `irqbalance`, apt timers, `motd-news`, `apport`.
7. **CPU partitioning**: `HOUSEKEEPING_CPUS=0-1`, `BENCH_CPUS=2-(N-1)`. Pins all writable
   `/proc/irq/*/smp_affinity_list` to housekeeping CPUs 0–1.
8. Persists `HOUSEKEEPING_CPUS` / `BENCH_CPUS` to `/tmp/vortex-benchmark.env` for the
   non-root run step.

### Pinning wrapper — `scripts/bench-taskset.sh`

- Sources `/tmp/vortex-benchmark.env`. If `BENCH_CPUS` is unset, derives it: with `numactl`
  present, takes all NUMA-node-0 CPUs except 0–1; otherwise `2-(nproc-1)`.
- Execs the benchmark under **`numactl --physcpubind=$BENCH_CPUS --membind=0`** when
  `numactl` exists, else **`taskset -c $BENCH_CPUS`**. So benches are already CPU-pinned and
  memory-node-pinned, isolated from the housekeeping CPUs that absorb IRQs and the OS.

**Takeaway:** Vortex *already* implements essentially the full walltime-stabilization
playbook (turbo off, ASLR off, IRQ steering, CPU+memory pinning, swap off). That work is not
CodSpeed-specific and we keep it regardless of which tool we pick.

### Benchmark frameworks actually in use

`grep` across `vortex-*/benches`, `encodings/*/benches`, `vortex/benches`:

- **59 bench files use `divan`**, wired through the workspace dependency
  `divan = { package = "codspeed-divan-compat", version = "4.0.4" }` (`Cargo.toml:152`). All
  CPU microbenches (e.g. `vortex-array/benches/take_primitive.rs` → `use divan::Bencher;`,
  `divan::main()`) are divan benches. These are the ones run in **simulation** mode.
- **13 bench files use `criterion`** — all under `vortex-cuda/benches`, wired through
  `criterion = { package = "codspeed-criterion-compat-walltime", version = "4.3.0" }`
  (`vortex-cuda/Cargo.toml:48`). These run in **walltime** mode.
- The workspace also declares a *plain* `criterion = "0.8"` (`Cargo.toml:127`) and plain
  `divan` is the compat shim — i.e. the `codspeed-*-compat` crates are drop-in replacements
  for upstream `divan` / `criterion`, selected at the dependency level. `cfg(codspeed)` is a
  declared check-cfg (`Cargo.toml:327`).
- `[[bench]] harness = false` is set for every bench target (e.g.
  `vortex-array/Cargo.toml`), as required for custom harnesses.

So the migration surface is: **swap the `package = "codspeed-*-compat*"` dependency targets**,
plus the CI workflow. The bench *source files* mostly stay as-is because the compat crates
expose the same `divan`/`criterion` API. This is the single most important architectural fact
for the rest of this plan.

### Macro-bench measurement model — `vortex-bench/src/measurements.rs`, `output.rs`

Separate from microbenches: `vortex-bench` models query/compression benchmarks as wall-time
`Duration` runs with **median** reduction (`TimingMeasurement::median_time`,
`QueryMeasurement::median_run`) plus memory deltas/peaks (`MemoryMeasurement`). Output is JSON
keyed by `commit_id` (`GIT_COMMIT_ID`) and host `Triple` (arch/os/env), written to
`target/vortex-bench/<id>/results.json` (`output.rs`). This is our model for "walltime done
properly with robust statistics + provenance" and we should reuse its median/JSON conventions
for any walltime microbench signal we keep.

## 2. The core decision: instruction-counting vs walltime

| Property | Instruction/sim counting (Callgrind/Cachegrind) | Walltime (even on tuned runners) |
|---|---|---|
| Determinism | Effectively deterministic; same binary → same count | Stochastic; needs many iterations |
| Cross-machine comparability | High — counts are HW-independent | Low — tied to uarch, freq, caches |
| Variance in CI | <1% claimed by CodSpeed for sim mode [1][6] | Typically several %, can spike on shared VMs |
| Measures the *right* thing? | Proxy: instructions ≠ time; misses memory-latency, SIMD throughput, branch-predictor, real cache *timing*, syscalls/I-O | Yes — actual elapsed time incl. SIMD, memory stalls, syscalls |
| Speed | One pass, but Valgrind runs ~10–50× slower | Many iterations × warmup, but native speed |
| GPU support | None (CPU simulation only) | Required for CUDA |

Conclusion in one line: **instruction counts give us the stability/repeatability the bullet
demands; walltime tells us the truth but is noisy.** We want both, with the stable signal
gating regressions.

### Tool A — `iai-callgrind` / `gungraun` (the OSS equivalent of CodSpeed simulation)

`iai-callgrind` is the leading open-source instruction-count harness; the project has since
been **renamed `gungraun`** (same maintainers, repo `iai-callgrind/iai-callgrind` →
`gungraun/gungraun`) [4][7]. How it works:

- Drives Valgrind's **Callgrind** (default) or **Cachegrind** (since 0.15.0) to count, per
  benchmarked function: **Instructions (Ir)**, **L1/L2/RAM hits**, total reads+writes, and an
  **Estimated Cycles** figure using the textbook formula
  `Est.Cycles = L1Hits + 5×L2Hits + 35×RAMHits` [4][1]. DHAT heap profiling integrated since
  0.16.0 [4]. Multi-threaded/multi-process benching since 0.14.0 [4].
- **Stability mechanism:** the harness splits the *runner* binary from the *library under
  test* and counts only events inside the benchmark function, excluding setup. Because it
  counts simulated events, not time, results are "comparable between different systems
  completely negating the noise of the environment," and **each benchmark runs once** —
  usually faster wall-clock than statistical walltime harnesses despite Valgrind overhead [4].
- **Output:** per-bench absolute metrics plus diff vs the previous run (`% change` and
  multiplier), stored under `target/iai/...`; compatible with `callgrind_annotate` and
  `kcachegrind` for drill-down [4].
- **Rust integration:** `#[library_benchmark]` + `#[bench::case(args)]` attributes,
  `library_benchmark_group!`, `main!(...)`; `harness = false`. Requires the
  `iai-callgrind-runner` binary installed (version must match the lib crate) and **Valgrind on
  PATH** [4].
- **Regression gating built in:** `RegressionConfig::default().limits([(EventKind::Ir, 5.0)])`
  or env `IAI_CALLGRIND_REGRESSION='Ir=5'` + `IAI_CALLGRIND_REGRESSION_FAIL_FAST=yes` [4].
  This is a *self-hosted* fail-on-regression — no SaaS needed.
- **Fully standalone**, dual MIT/Apache-2.0. No account, no token, no upload.
- **Migration cost (the catch):** iai-callgrind uses its *own* macro API. It is **not** a
  drop-in for `divan`/`criterion`. Our 59 divan benches would need their bench bodies
  re-expressed as `#[library_benchmark]` functions. The bodies are mostly "build inputs, call
  the kernel" so the port is mechanical but not free. Divan's parameter matrices
  (`const NUM_INDICES: &[usize]`, etc.) map to `#[bench::name(arg)]` cases.

### Tool B — `cargo-codspeed` minus the SaaS

`cargo-codspeed` and the `codspeed` runner are **open source (MIT/Apache-2.0)** [8][3]. The
moving parts:

- `codspeed-divan-compat` / `codspeed-criterion-compat` are compat shims: when **not** built
  under `cfg(codspeed)` they behave like vanilla divan/criterion; under instrumentation they
  emit CodSpeed's measurement hooks.
- `cargo codspeed build` compiles the bench binaries; `cargo codspeed run` executes them. The
  CodSpeed **runner** wraps that run in Valgrind/Callgrind (simulation) or perf-style walltime
  and emits profile artifacts. By default it runs in instrumentation mode; `--mode` selects
  sim vs walltime [run docs].
- **What `CodSpeedHQ/action` actually adds = upload + gating UI.** The action authenticates
  with `CODSPEED_TOKEN` (or OIDC `id-token: write`), posts results to codspeed.io, and the PR
  comment / regression verdict come from the *cloud* comparing against the base branch [2][5].
  The instrumentation itself is local.
- **Can we keep the runner, drop the backend?** Partially. The runner can *produce* the
  Valgrind profile data locally, and public repos / tokenless runs are tolerated [search:FAQ].
  But the **comparison, history, and pass/fail regression verdict live server-side** — that is
  the product. There is no supported, documented "emit a stable JSON of instruction counts and
  diff it yourself" path baked into cargo-codspeed today. We would be reverse-engineering the
  Callgrind `.out` parsing and building our own diff/gate — which is exactly what
  iai-callgrind already ships as a first-class, documented feature.
- Verdict: reusing the CodSpeed OSS layer *without* the SaaS means re-implementing the gating
  half ourselves on top of an undocumented artifact format. iai-callgrind gives the same
  Valgrind-based numbers **with** gating already built and documented. Prefer iai-callgrind for
  the self-hosted instruction-count path.

### Tool C — Walltime with heavy stabilization (what we already have + `perf`)

We keep the `setup-benchmark.sh` / `bench-taskset.sh` tuning and add the standard robustness
layers:

- **Dedicated pinned runners** (already self-hosted `runs-on`), **CPU isolation** via
  `taskset`/`numactl` (already done), **turbo/boost off, ASLR off, IRQ steering** (already
  done).
- Optionally `isolcpus=`/`nohz_full=` at the kernel cmdline for the bench CPUs (stronger than
  IRQ steering alone) — a future hardening, not required day one.
- **Count cycles/instructions with `perf stat`** instead of (or alongside) elapsed time:
  hardware counters for `cycles` and `instructions` are less jittery than nanoseconds and let
  us sanity-check the simulated counts against real hardware. Divan can be configured to
  measure via the OS; a `perf stat -e instructions,cycles` wrapper around the bench binary is
  the lightest integration.
- **Statistics:** many iterations + warmup, reduce with **median** and report dispersion as
  **MAD** (median absolute deviation) rather than mean/stddev (robust to the occasional
  scheduler spike). This mirrors what `vortex-bench` already does with `median_time()` /
  `median_run()`.

## 3. Recommended approach — a hybrid, with instruction counts as the gate

**Primary (regression gate): instruction counts via `iai-callgrind`/`gungraun`, self-hosted.**
This directly satisfies the bullet: deterministic, machine-independent, <1%-class variance,
no SaaS dependency, with built-in `RegressionConfig` thresholds. It replaces today's
`mode: "simulation"` CPU shards.

**Secondary (truth signal, non-gating or soft-gating): walltime on the tuned pinned runners**,
reduced with median/MAD, optionally backed by `perf stat` cycle/instruction counters. This
catches things instruction counts *miss* — SIMD throughput (`+avx2`!), memory-latency-bound
kernels, allocator behavior — which matter a lot for a columnar/compression engine. Keep the
existing divan benches for this, run them natively (no Valgrind) on the isolated CPUs.

**CUDA stays walltime-only** (instruction counting can't model GPUs) — keep the existing
`codspeed-criterion-compat-walltime` benches; just point their gating at our own store instead
of CodSpeed (covered by the diff-view slice).

Justification: the bullet asks for *stable and repeatable* so we can *see* perf changes — that
is an instruction-count strength and a walltime weakness, so instruction counts gate. But
instruction count is a proxy; for a SIMD/compression codebase we cannot let a kernel regress in
real cycles while its instruction count is flat (e.g. losing vectorization keeps Ir similar but
tanks throughput). Hence walltime/`perf` remains as a watched secondary. This is the same
two-mode split CodSpeed itself offers (simulation + walltime), just self-hosted.

## 4. Noise budget & regression thresholds

| Signal | Realistic variance on our tuned runners | Suggested regression threshold |
|---|---|---|
| iai-callgrind `Ir` (instructions) | Near-deterministic; sub-0.1% run-to-run typical, <1% claimed even in noisy CI [1][6] | **Hard fail at > +1.0% Ir** (alarm at +0.5%) |
| iai-callgrind Estimated Cycles | Slightly noisier than Ir (cache model) | Alarm only, ~ +2–3% |
| Walltime median (divan, isolated CPUs) | A few % typical; depends on kernel | **Soft fail at > +5–7%** median, require sustained over ≥2 runs |
| `perf` cycles | Lower than walltime ns, higher than sim Ir | Alarm ~ +3% |

Threshold-setting principle: gate on the **lowest-variance signal with the tightest band**
(instruction count). For walltime, prefer **confidence on the median across N≥10 iterations
with MAD-based bounds** over a flat percentage, and require a regression to reproduce across
two consecutive commits/runs before failing CI to avoid flake-driven false alarms. These
numbers feed the diff-view slice (bullet 4): the diff view should show Ir delta as the
headline number (tight band, red/green) and walltime median±MAD as advisory.

Calibration step before turning gates on: run the chosen benches ~20× on the actual runner
image at a fixed commit and measure observed variance per metric; set each threshold at roughly
`max(observed p99 noise, documented floor)`.

## 5. Determinism gotchas (must control for stable instruction counts AND comparable builds)

- **ASLR** — already disabled in `setup-benchmark.sh` (`randomize_va_space=0`). Keep it;
  Valgrind also neutralizes most layout noise, but matching addresses helps annotation diffs.
- **Allocator** — instruction counts are sensitive to the allocator's code path. Pin one
  allocator across the matrix (the workspace default), and be aware that allocator fast/slow
  paths change Ir. Avoid switching jemalloc/mimalloc between base and PR.
- **Codegen / RUSTFLAGS** — CI builds with `-C target-feature=+avx2`. Instruction counts are
  only comparable between builds compiled **identically**. Pin `RUSTFLAGS`, profile
  (`--profile bench`), toolchain (`NIGHTLY_TOOLCHAIN`/stable), and feature set per shard. A
  base-vs-PR comparison must use the *same* flags, or the Ir delta is meaningless.
- **`target-cpu` vs the arch matrix** — `+avx2` (feature) is fine and portable across x86-64
  AVX2 machines; `target-cpu=native` would make builds host-specific and break cross-machine
  comparability — **do not** use `native` for benches. For the broader arch matrix (bullet 1),
  treat each `(arch, target-feature)` pair as a *separate* benchmark series; never compare Ir
  across architectures (arm64 vs x86-64 instruction counts are not comparable).
- **Dataset size / RNG** — many benches use random inputs (`rand`, `Zipf` in
  `take_primitive.rs`). Seed RNGs deterministically so input data is identical run-to-run;
  otherwise instruction counts wobble with the data shape. This is the single most common
  source of "stable tool, unstable number."
- **Parallelism** — multi-threaded benches make instruction counts depend on scheduling.
  iai-callgrind supports multi-threaded counting (≥0.14) but prefer single-threaded
  microbenches for the gating signal; reserve threaded behavior for the walltime signal. The
  walltime path already pins to `BENCH_CPUS` and `--membind=0`.
- **Valgrind/toolchain versions** — pin the Valgrind version in the runner image; Callgrind
  output can shift across Valgrind releases. Cache the `iai-callgrind-runner` install (we
  already cache `cargo-codspeed` via `taiki-e/cache-cargo-install-action`).

## 6. Concrete migration steps for Vortex

1. **Add deps.** Introduce `iai-callgrind` (or `gungraun`) + `iai-callgrind-runner` as
   workspace dev-dependencies. Keep plain `divan`/`criterion` for the walltime path; the goal
   is to *remove* the `codspeed-*-compat` package indirection.
2. **Flip the compat shims back to upstream** for the walltime signal: change
   `Cargo.toml:152` `divan = { package = "codspeed-divan-compat", ... }` → upstream `divan`,
   and `vortex-cuda/Cargo.toml:48`
   `criterion = { package = "codspeed-criterion-compat-walltime", ... }` → upstream
   `criterion`. Drop the `cfg(codspeed)` check-cfg (`Cargo.toml:327`) once no compat code
   references it.
3. **Pick the gated subset.** Not all 59 divan benches need instruction-count gating; choose
   the hot-path kernels (take/filter/compare/compress, FastLanes bitpacking, FSST, ALP,
   RunEnd). Port those to `#[library_benchmark]` iai-callgrind targets (new
   `benches/*_iai.rs` with `harness = false`), mapping divan const-arrays to `#[bench::]`
   cases. Seed all RNG deterministically during the port.
4. **Keep the rest as native divan walltime benches**, run on the isolated CPUs via the
   existing `bench-taskset.sh`, reducing with median/MAD; reuse `vortex-bench` JSON
   conventions (`commit_id`, host `Triple`) for storage.
5. **Rework `.github/workflows/codspeed.yml`:**
   - Replace the `CodSpeedHQ/action` step in CPU shards with: install Valgrind +
     `iai-callgrind-runner` (cached), `cargo bench` the iai targets under
     `bash scripts/bench-taskset.sh`, set `IAI_CALLGRIND_REGRESSION='Ir=1'` (+ fail-fast) for
     the gate, and upload the `target/iai` artifacts to *our* store (diff-view slice).
   - Keep `setup-benchmark.sh` and `bench-taskset.sh` verbatim — they are tool-agnostic.
   - Keep `-C target-feature=+avx2`, `--profile bench`, the 8-way sharding, and
     `system-info`/`runs-on` runner selection.
   - CUDA shards: keep walltime, repoint result upload from CodSpeed to our store.
6. **Calibrate thresholds** (section 4) on the runner image, then enable gating.
7. **Validation (no cargo here, per instructions):** when implementing, run narrow
   `cargo bench -p <crate> --bench <name>` locally with Valgrind installed to confirm Ir is
   stable across two runs at the same commit before wiring the gate.

## 7. Open questions / decisions for the user

1. **iai-callgrind vs gungraun:** the project renamed to `gungraun`. Track the new crate
   name/repo or pin the last `iai-callgrind` release? (Affects dep names in step 1.)
2. **Callgrind vs Cachegrind as default tool:** Callgrind gives callstack/flamegraph data;
   Cachegrind gives richer cache numbers and is faster. Gate on Callgrind `Ir`, or switch
   default to Cachegrind (`--default-tool`)?
3. **How much of the 59 divan benches to port** to instruction-count gating vs leave as
   walltime-only? (Porting cost vs coverage.)
4. **Is losing the CodSpeed PR-comment UX acceptable**, given we must build the diff
   view/history ourselves (the diff-view slice)? Confirm we are not retaining any CodSpeed
   SaaS.
5. **Allocator policy** for benches — pin which allocator, and forbid per-PR allocator swaps
   in benched code paths.
6. **Kernel-level isolation** (`isolcpus`/`nohz_full`) — worth baking into the runner image
   for the walltime signal, or is `taskset`+IRQ-steering enough?
7. **Walltime gating strictness** — advisory-only, or soft-fail at a median+MAD bound? This
   determines whether a SIMD regression with flat Ir can block a PR.

## Sources

- [1] CodSpeed action / instruments — simulation mode, <1% variance:
  https://github.com/marketplace/actions/codspeed-performance-analysis
- [2] CodSpeedHQ/action (modes, token/OIDC upload): https://github.com/CodSpeedHQ/action
- [3] cargo-codspeed crate: https://crates.io/crates/cargo-codspeed
- [4] iai-callgrind / gungraun (metrics, est-cycles formula, regression config, standalone):
  https://github.com/clockworklabs/iai-callgrind and https://github.com/iai-callgrind/iai-callgrind
- [5] CodSpeed walltime instrument (bare-metal Macro Runners, parallelism caveat):
  https://codspeed.io/docs/instruments/walltime
- [6] CodSpeed (toolkit overview, <1% variance simulation): https://codspeed.io/
- [7] gungraun (renamed project): https://github.com/iai-callgrind/iai-callgrind
- [8] CodSpeed runner (open source, MIT/Apache-2.0): https://github.com/CodSpeedHQ/runner
- FAQ (local/tokenless runs): https://codspeed.io/docs/faq
- Iai-Callgrind crate page: https://crates.io/crates/iai-callgrind
