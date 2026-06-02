<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# CodSpeed Replacement — Design Overview

This directory is a design study for replacing **CodSpeed** (the SaaS that today runs Vortex's
microbenchmarks and hosts their PR comment + diff dashboard) with a self-hosted stack built on
infrastructure the project **already owns**. It is split into five slices, one per requested
capability:

| # | Slice | Doc |
|---|-------|-----|
| 1 | Benchmarks on different CPU/arch via `runs-on` | [`01-multiarch-runners.md`](01-multiarch-runners.md) |
| 2 | Stable, repeatable runs across machines | [`02-stability-repeatability.md`](02-stability-repeatability.md) |
| 3 | Storage for each run | [`03-storage.md`](03-storage.md) |
| 4 | GitHub comment + diff view | [`04-pr-comment-diff-view.md`](04-pr-comment-diff-view.md) |
| 5 | Stackcharts with per-run diffs | [`05-stackcharts-flamegraph-diffs.md`](05-stackcharts-flamegraph-diffs.md) |

This overview synthesizes them: what we keep, the target architecture, the phased plan, and the
cross-cutting decisions that need your call before implementation.

---

## 1. The single most important discovery

Vortex already has ~80% of a CodSpeed replacement in-tree. We are not building from scratch; we are
**lifting microbenchmarks off the CodSpeed SaaS onto the same rails the macro benchmarks already
run on.** Inventory:

- **Runners** — `runs-on` self-hosted GHA runners, including a proven **arm64/Graviton** image
  (`ubuntu24-full-arm64-pre-v2`, already used by `ci.yml`/`package.yml`) and **dedicated pinned
  pools** (`runner=bench-dedicated`, `nightly-bench.yml`'s `i7i.metal-24xl`).
- **Run stabilization** — `scripts/setup-benchmark.sh` (turbo off, ASLR off, NUMA/swap off, IRQ
  steering, CPU partitioning) + `scripts/bench-taskset.sh` (`numactl --physcpubind --membind=0`).
  This is the full walltime-stabilization playbook, tool-agnostic, already in production.
- **Storage** — `benchmarks-website/server/` is an **axum + DuckDB** SSR site with a registry of
  fact "families", an authenticated `POST /api/ingest`, hourly S3 snapshots, and the
  `vortex-ci-benchmark-results` S3 bucket (OIDC `GitHubBenchmarkRole`, `cat-s3.sh` CAS append).
- **PR comment pattern** — `bench-pr.yml` already posts a **sticky** macro-bench comment via
  `thollander/actions-comment-pull-request`, with a Python diff/markdown generator
  (`scripts/compare-benchmark-jsons.py`) and fork-safe gating.
- **Profiling** — Polar Signals Cloud is already wired into `bench.yml`/`bench-pr.yml`/
  `sql-benchmarks.yml` (eBPF pprof, native differential-flamegraph UI). `samply` + a `[profile.samply]`
  profile + a samply skill already exist.
- **Harness** — microbenches build/run through `codspeed-*-compat` shims that are **drop-in
  replacements** for upstream `divan`/`criterion`. Off CodSpeed they are ordinary wall-clock
  benches. (See §2 caveat on which framework is which.)

What CodSpeed uniquely provides today, and therefore what we must replace:

1. **Deterministic, machine-independent measurement** (Valgrind/Callgrind instruction counts in
   `simulation` mode) — the stability guarantee.
2. **The hosted PR comment + regression verdict.**
3. **The hosted diff dashboard + per-bench history.**
4. (For micro) **a profile/flamegraph per run** — CodSpeed shows where time/instructions went.

Slices 2, 4, 3, and 5 respectively replace those four things; slice 1 generalizes the run across
architectures.

---

## 2. Harness caveat to resolve early (cross-slice inconsistency)

The slices disagree slightly on the microbench framework, and it changes slice 5's capture path:

- Slice 2 grepped specifically and found: **all CPU microbenches are `divan`** (59 files, via
  `divan = { package = "codspeed-divan-compat" }`, `Cargo.toml:152`); **only the CUDA microbenches
  are `criterion`** (13 files, `codspeed-criterion-compat-walltime`, `vortex-cuda/Cargo.toml:48`).
- Slices 4/5 in places refer to the CPU microbenches as "criterion."

Treat **slice 2 as authoritative** (it grepped). This matters because the lowest-friction profiler
integration in slice 5 — `pprof-rs`'s `PProfProfiler` — is a **criterion** integration, not a divan
one. So the slice-5 profile pass for CPU benches must use `samply`/`perf` against the divan bench
binary (divan has no built-in pprof hook), not `pprof-rs`'s criterion adapter. **Action: confirm the
divan-vs-criterion split before building slice 5's capture step.**

---

## 3. Target architecture

```
                         ┌──────────────────────────────────────────────┐
   PR / push:develop     │  GHA: microbench.yml  (replaces codspeed.yml) │
   ───────────────────▶  │  matrix = arch{x64,arm64} × shard{1..8}        │  ← slice 1
                         │  runs-on dedicated pinned pool, spot=false     │
                         │  setup-benchmark.sh + bench-taskset.sh         │  ← slice 2 (stabilize)
                         │                                                │
                         │  (a) METRIC pass:                              │
                         │      • iai-callgrind/gungraun  → Ir counts     │  ← slice 2 (GATE signal)
                         │        (deterministic, cross-machine)          │
                         │      • native divan walltime + perf cycles     │  ← slice 2 (truth signal)
                         │  (b) PROFILE pass (curated subset):            │
                         │      • samply/pprof → folded stacks            │  ← slice 5
                         └───────────────┬────────────────────────────────┘
                                         │ artifacts (read-only job; fork-safe)
                                         ▼
        ┌────────────────────────────────────────────────────────────────┐
        │  develop runs → ingest;  PR runs → compare-only                  │
        │                                                                  │
        │  DuckDB fact family `microbenchmarks` (commit,arch,runner,        │  ← slice 3
        │     bench_id,metric∈{instructions,walltime_ns,cycles})            │
        │     via POST /api/ingest  (benchmarks-website/server)            │
        │  S3 vortex-ci-benchmark-results: raw blobs + pprof + diff.svg     │  ← slice 3 + 5
        └───────────────┬───────────────────────────────────┬──────────────┘
                        │                                   │
       workflow_run     ▼                                   ▼
   ┌──────────────────────────────┐         ┌────────────────────────────────────┐
   │ microbench-comment.yml        │         │ benchmarks-website (axum+DuckDB SSR) │
   │ sticky PR comment, per-arch    │ ← s4    │  GET /compare?base&head  (new)       │ ← slice 4
   │ Δ% table, ✅/⚠️/🔴, deep links │         │  GET /micro/{name} chart overlay     │ ← slice 4
   │ + Profiles: diff.svg / PS link │ ← s5    │  embed inferno diff SVG / PS iframe   │ ← slice 5
   └──────────────────────────────┘         └────────────────────────────────────┘
```

---

## 4. Per-slice recommendations (one line each)

1. **Multi-arch runners** — add an `arch × shard` matrix over pinned, on-demand (`spot=false`)
   dedicated pools `bench-micro-x64` (`c7i.4xlarge`) and `bench-micro-arm64` (`c7g.4xlarge`); drop
   `target-cpu=native`, pin `x86-64-v3` / `neoverse-v1`; full 8 shards on x64 + arm64 *smoke* per PR,
   full both-arch on develop/nightly. CUDA stays x64-only.
2. **Stability** — **hybrid**: `iai-callgrind`/`gungraun` instruction counts are the **regression
   gate** (deterministic, machine-independent, built-in thresholds, no SaaS); native divan walltime
   + `perf` cycles on the pinned runners are the **secondary truth signal** (catches SIMD/`+avx2`
   throughput regressions that flat Ir hides). Keep `setup-benchmark.sh`/`bench-taskset.sh` verbatim.
3. **Storage** — add a sixth DuckDB fact family `microbenchmarks` keyed
   `(commit_sha, crate, bench_group, benchmark_id, metric, arch, runner)`, ingested via the existing
   `POST /api/ingest`; raw blobs + profiles to `s3://vortex-ci-benchmark-results/micro/…`. Develop
   ingests; PR compares without ingesting. Additive, no `SCHEMA_VERSION` bump.
4. **PR comment + diff view** — fork-safe **`workflow_run` split** (`microbench.yml` runs read-only +
   uploads artifact; `microbench-comment.yml` posts with write perms from trusted base-branch code).
   Sticky comment grouped by arch with `Benchmark | base | head | Δ% | status`; base **fetched, not
   re-run**, at the true `git merge-base`. Advisory by default; opt-in hard gate behind a
   `perf-sensitive` label. New website routes `GET /compare` + `GET /micro/{name}`.
5. **Stackcharts/diffs** — **keep Polar Signals Cloud for macro** (free diff UI, just add a deep
   link). For micro, add a separate sampling pass → folded stacks → **`inferno-diff-folded` +
   `inferno-flamegraph --negate`** static red/blue **SVG** in S3, embedded in the website and linked
   from the PR comment. Cheapest fully-owned option; revisit self-hosted Parca later.

---

## 5. Phased implementation roadmap

**Phase 0 — Decisions & calibration (no code merged to gate)**
- Resolve the cross-cutting decisions in §6.
- Confirm the divan/criterion split (§2).
- Stand up a `bench-micro-x64` dedicated pool; run a fixed commit ~20× to measure real run-to-run
  variance per metric and set thresholds (slice 2 §4).

**Phase 1 — Metric pipeline off CodSpeed (slices 1 + 2)**
- Port the hot-path kernels (take/filter/compare/compress, FastLanes, FSST, ALP, RunEnd) to
  `iai-callgrind` library benchmarks; deterministically seed all RNG.
- Keep the rest as native divan walltime benches under `bench-taskset.sh`.
- New `microbench.yml` with the `arch × shard` matrix; **run alongside `codspeed.yml`** (no deletion
  yet). Emit per-`(arch,shard)` artifacts. Not gating.

**Phase 2 — Storage (slice 3)**
- Add the `microbenchmarks` family + `POST /api/ingest` path + S3 raw-blob layout.
- develop runs ingest; characterize the stored series vs CodSpeed's numbers for a couple of weeks.

**Phase 3 — Comment + diff view (slice 4)**
- `compare-microbench-jsons.py` + the `workflow_run` comment split (advisory).
- Website `/compare` and `/micro/{name}` routes; deep links live.

**Phase 4 — Profiles (slice 5)**
- Profile pass for a curated subset; inferno diff SVG to S3; PS Cloud deep link for macro; embed in
  comment + website.

**Phase 5 — Cutover**
- After a 2–4 week overlap where our verdicts match CodSpeed's, delete `codspeed.yml`, remove
  `CODSPEED_TOKEN`, and (optionally) promote the gate to a required check once variance is proven.

The phases are independently shippable; nothing requires deleting CodSpeed until Phase 5.

---

## 6. Cross-cutting decisions for you (consolidated)

These appear repeatedly across the slice docs; deciding them unblocks implementation:

1. **Gate signal & strictness.** Confirm the hybrid: instruction counts (iai-callgrind) gate hard at
   ~+1% Ir; walltime is advisory (or soft-fail at median+MAD). Is a SIMD regression with flat Ir
   allowed to block a PR? (slices 2, 4)
2. **`iai-callgrind` vs `gungraun`.** The project was renamed; track the new crate or pin the last
   `iai-callgrind` release? (slice 2)
3. **Port scope.** Which/how many of the 59 divan benches get instruction-count gating vs stay
   walltime-only? (slices 2, 5 — also bounds the profile subset)
4. **Runner provisioning.** Provision dedicated `bench-micro-{x64,arm64}` pools (strongest
   comparability) vs `family=…/spot=false` on-demand? And exact instance types (Graviton3 vs 4,
   metal?). (slice 1)
5. **arm64 PR coverage + AVX-512 series.** x64-full + arm64-smoke per PR (recommended) vs full both?
   Add an explicit AVX-512 matrix variant or not? (slices 1, 3)
6. **Base selection.** True `git merge-base` (recommended, requires microbenches on every develop
   commit) vs latest-successful-develop (current macro behavior). And base data source: website read
   endpoint vs S3 grep. (slices 3, 4)
7. **Profile cost/visual.** Accept one extra sampling pass (profile a curated subset / only on
   regression?). Static inferno SVG now vs invest in self-hosted Parca for interactive micro diffs?
   Keep paying for PS Cloud for macro or consolidate on Parca? Retention for large pprof blobs.
   (slice 5)
8. **CUDA.** Confirm CUDA microbenches stay walltime-only, x64/`g5`-only, with their own (wider)
   threshold band. (slices 1, 2, 4)
9. **Cutover window.** Confirm a 2–4 week dual-run overlap with CodSpeed before decommissioning, and
   that we are not retaining any CodSpeed SaaS. (slices 2, 4)

---

## 7. Top risks

- **Instruction count is a proxy.** A columnar/SIMD engine can regress in real cycles while Ir stays
  flat (lost vectorization). Mitigated by the walltime/`perf` secondary signal — do not drop it.
- **Microbench flake blocking merges.** Hence advisory-by-default + opt-in gate, with thresholds
  calibrated on the actual runner image before any hard gate.
- **Determinism debt.** RNG seeding is the single biggest source of "stable tool, unstable number";
  every ported bench must seed deterministically. Identical RUSTFLAGS/profile/toolchain across
  base-vs-PR is mandatory; never `target-cpu=native`.
- **Divan ≠ criterion for profiling.** The convenient `pprof-rs` criterion hook does not apply to the
  divan CPU benches; confirm the capture path (§2).
- **Capacity vs comparability.** Pinning one instance type + on-demand risks "no capacity"; the
  dedicated-pool route avoids per-run gambles but needs provisioning.

---

*Each slice doc contains the detailed research, YAML/SQL/Python sketches, file/line citations, and
external sources. Start with this overview, then read the slice relevant to the phase you are
implementing.*
