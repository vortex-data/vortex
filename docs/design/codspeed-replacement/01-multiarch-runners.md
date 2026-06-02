# 01 - Multi-arch microbenchmark runners (`runs-on`)

Slice owner bullet: **"Benchmarks on different CPU and arch, runs-on."**

This document covers the runner-selection and matrix mechanics for running the Vortex
microbenchmarks across multiple CPUs/architectures on [runs-on](https://runs-on.com)
self-hosted GitHub Actions runners, replacing CodSpeed's single hidden runner. It is one
slice of the larger CodSpeed-replacement design.

- **In scope:** the `runs-on` runner matrix (amd64 + arm64/Graviton), expressing arch as a
  matrix dimension, propagating arch-specific build flags, how sharding interacts with the
  arch dimension, and a workflow skeleton that emits a results artifact.
- **Out of scope (handoffs):** *how* to keep a single machine's numbers stable (bullet 2 -
  pinning/taskset/turbo are referenced but owned there), and the *results format / storage /
  regression detection* (bullet 3 - we only define the artifact handoff).

## 1. Grounding: what exists today

Current microbench workflow: `.github/workflows/codspeed.yml`.

- `bench-codspeed` job: 8 CPU shards (`matrix.include`, lines 30-37), each a set of
  `-p <packages>`, built and run via `cargo codspeed build/run` in CodSpeed "simulation"
  (instruction-count) mode.
- All 8 shards run on the **same** amd64 runner spec (lines 40-43):
  `runs-on={run_id}/runner=amd64-medium/image=ubuntu24-full-x64-pre-v2/tag=bench-codspeed-{shard}`.
- `RUSTFLAGS: "-C target-feature=+avx2"` is set for the build step (line 60). Note this is
  *not* `target-cpu=native` here - CodSpeed simulation is instruction-counted, so microarch
  matters less. The macro benches (`bench.yml` line 75, `sql-benchmarks.yml` line 346) *do*
  use `-C target-cpu=native`, which is exactly the cross-machine-comparability hazard we must
  avoid for wall-time microbenches (see section 2).
- `bench-codspeed-cuda` job: 3 shards on `family=g5/cpu=8/image=ubuntu24-gpu-x64` (line 80).
  GPU benches are arch-irrelevant for this slice; keep them x64-only.

Runner-selection patterns already in the repo we should reuse verbatim:

- `runs-on/action@v2` with `sccache: s3`, gated on
  `github.repository == 'vortex-data/vortex'` so forks fall back to `ubuntu-latest`
  (`codspeed.yml` lines 45-47, 40-43).
- **arm64 already works in this repo.** `runs-on.yml` defines image
  `ubuntu24-full-arm64-pre-v2` (arch `arm64`, lines 9-13). `ci.yml` line 308 and
  `package.yml` line 117 already run arm64 jobs via
  `runner=arm64-medium/image=ubuntu24-full-arm64-pre-v2`. So the arm64 runner family + AMI is
  proven; this slice mostly wires it into the bench matrix.
- Dedicated/pinned-instance pattern: `bench.yml` uses `runner=bench-dedicated`;
  `sql-benchmarks.yml` line 296 uses `runner=bench-dedicated/instance-type={machine_type}`;
  `nightly-bench.yml` pins `i7i.metal-24xl`. These are the precedents for "same hardware
  every run".

Bench harness facts (matter for arch portability):

- Microbenches use `divan` via `codspeed-divan-compat` (`Cargo.toml` line 152) plus
  `criterion 0.8` (line 127); `cargo-codspeed` drives build/run. `[[bench]]` entries with
  `harness = false` live in `vortex-array`, `vortex-buffer`, `vortex-mask`, `vortex`,
  `vortex-btrblocks`, and each `encodings/*` crate (matching the shard package lists).
- Benches already do **runtime** ISA detection, e.g.
  `vortex-buffer/benches/vortex_bitbuffer.rs` pre-warms
  `is_x86_feature_detected!("avx2"/"avx512f"/"avx512vpopcntdq")`. This is x86-gated with
  `#[cfg(target_arch = "x86_64")]`, so it already compiles cleanly on arm64; an equivalent
  NEON/SVE warm-up would be additive, not required.
- `scripts/setup-benchmark.sh` and `scripts/bench-taskset.sh` are arch-agnostic (they probe
  `intel_pstate` *and* generic `cpufreq/boost`, use `numactl`/`taskset`), so they run on
  Graviton as-is. Stability tuning lives in bullet 2.

## 2. Runner matrix: amd64 + arm64 (Graviton)

### 2.1 runs-on label mechanics (from docs)

`runs-on`'s job-label string selects compute at runtime. Relevant attributes
([job-labels](https://runs-on.com/configuration/job-labels/),
[linux runners](https://runs-on.com/runners/linux/)):

- `runner=<name>` - a predefined size, e.g. `amd64-medium`, `arm64-medium`, or `Ncpu-linux-x64`
  / `Ncpu-linux-arm64`. Default arm64 sizes map to Graviton: `8cpu-linux-arm64 -> c7gd.2xlarge`,
  `16cpu -> c7gd.4xlarge`, `32cpu -> c7g.8xlarge` (Graviton3), `96cpu -> c8g.24xlarge`
  (Graviton4). x64 sizes map to `m7i-flex` / `c7i-flex` / `c7i`.
- `family=<type>` - pin/scope the EC2 instance type. Full name (`family=c7i.2xlarge`),
  prefix (`family=c7`), wildcard (`family=m7i.*`), or a `+`-list (`family=c7g+m7g`).
- `cpu=N` / `ram=N` (also ranges `cpu=4+16`), `image=<name>` (our pre-baked AMIs),
  `spot=false` (force on-demand), `extras=s3-cache`, `tag=<dedup>`.

### 2.2 Pinning the SAME hardware (comparability requirement)

Wall-time microbenches must run on identical silicon every time. Two combined controls:

1. **Pin a single instance type** via `family=<full-type>`, not a flex/range. Flex families
   (`m7i-flex`, `c7i-flex`) and ranges deliberately vary the chosen instance, which is fine
   for build/test but poison for benchmarking. Pin a *fixed-clock* type, e.g.
   `family=c7i.4xlarge` (Intel Sapphire Rapids, x86) and `family=c7g.4xlarge`
   (Graviton3, arm64). `c7i`/`c7g` are non-flex with stable clocks.
2. **Force on-demand** with `spot=false`. Spot can swap pools/hardware and be interrupted;
   docs recommend `spot=false` "if you require a specific instance type, which is often out
   of capacity" ([spot-instances](https://runs-on.com/configuration/spot-instances/)).

**Capacity tradeoff (decision for the user):** pinning one type + on-demand maximizes
comparability but risks "no capacity" launch failures (docs explicitly warn against a single
family). Mitigations, in order of preference:

- **Pre-provisioned dedicated pool / `runner=bench-dedicated`** (already used by `bench.yml`
  and `sql-benchmarks.yml`). This is the strongest guarantee: a warm, reserved host of known
  type. Recommend defining `bench-micro-x64` and `bench-micro-arm64` dedicated runners in the
  private RunsOn config (`runs-on.yml` `_extends: .github-private`, line 1) pinned to
  `c7i.4xlarge` / `c7g.4xlarge`. This keeps the comparability contract identical to the macro
  benches and avoids per-run capacity gambles.
- If a dedicated pool is not provisioned for arm64 yet, fall back to
  `family=c7g.4xlarge/spot=false` and accept occasional capacity retries.

Cost note (from linux-runners pricing): on-demand `c7g` (arm64) is ~15-25% cheaper per vCPU-min
than `c7i` (x64); both are cheap relative to job wall time. The dominant cost lever is shard
count x arch x PR frequency (section 4).

### 2.3 Concrete matrix entries

Recommended per-arch config carried in the matrix (combined with the shard rows from
section 4 via a cross-product or nested matrix):

| arch  | runner family (pinned)            | image                       | dedicated pool name (preferred) |
|-------|-----------------------------------|-----------------------------|---------------------------------|
| x64   | `family=c7i.4xlarge/spot=false`   | `ubuntu24-full-x64-pre-v2`  | `bench-micro-x64`               |
| arm64 | `family=c7g.4xlarge/spot=false`   | `ubuntu24-full-arm64-pre-v2`| `bench-micro-arm64`             |

Both AMIs already exist in `runs-on.yml`. `4xlarge` (16 vCPU) leaves room for
`scripts/setup-benchmark.sh`'s housekeeping/bench CPU split.

## 3. Arch as a matrix dimension + build flags

### 3.1 Express arch as its own dimension

Use a two-dimension matrix: `arch` x `shard`. Carry per-arch *runner* and *RUSTFLAGS* on the
`arch` rows; carry *packages* on the `shard` rows. GitHub expands the cross-product, and
`fail-fast: false` keeps one arch from killing the other.

### 3.2 Build flags per arch (no `target-cpu=native`)

`target-cpu=native` keys codegen to whatever host happened to be picked, which destroys
cross-machine and cross-run comparability. Instead pin an explicit, arch-stable target and an
explicit feature set so the binary is reproducible for a given (arch, instance type):

- **x86_64:** keep today's `-C target-feature=+avx2` as the baseline (matches `codspeed.yml`
  line 60 and the runtime `is_x86_feature_detected!` warm-ups). If we additionally want an
  AVX-512 data point, that is a *separate matrix entry* (e.g. a `flags`/`profile` column),
  not `native`, because `c7i` supports AVX-512 but pinning it explicitly keeps the binary
  identical run-to-run: `-C target-feature=+avx2,+avx512f,+avx512vl,+avx512bw,+avx512vpopcntdq`.
  Prefer `-C target-cpu=x86-64-v3` (stable, portable, == AVX2/BMI2/FMA baseline) over
  `native` when a named uarch is wanted; only add `+avx512*` features deliberately.
- **arm64:** NEON is baseline-mandatory on `aarch64` (always on, no flag needed). For SVE on
  Graviton3+ use a named CPU rather than `native`:
  `-C target-cpu=neoverse-v1` (Graviton3 / `c7g`) or `neoverse-n2`. Keep it pinned to the
  *chosen instance type's* uarch so codegen is deterministic.

Carry the flags string on the `arch` matrix row so the build step is just
`RUSTFLAGS: ${{ matrix.arch.rustflags }}`. The key invariant: **for a fixed (arch, instance
type) the RUSTFLAGS are a constant literal, never `native`.** Coordinate the exact AVX-512
on/off decision with bullet 3 (it changes the result-series identity/baseline).

## 4. Sharding x arch interaction, timeout, cost

- Today: 8 shards, 30 min timeout each, single arch -> 8 jobs/run.
- Naive cross-product: 8 shards x 2 arches = **16 jobs/run**. Plus optional AVX-512 variant
  on x64 would add up to 8 more. The CUDA shards stay x64-only and unchanged.
- Timeout: keep `timeout-minutes: 30` per (arch, shard). arm64 build is from cold-ish cache
  on first rollout; `extras=s3-cache` + the pre-baked AMI keep it comparable to x64.

**Policy recommendation (decision for the user):**

- **On every PR:** run the **full 8 shards on x64 only** (preserves current PR signal and
  cost) plus a **small arm64 smoke subset** (e.g. shards 2 "Arrays" + 4/5 encodings) to catch
  arch-specific regressions early without doubling PR cost.
- **On push to `develop` (and nightly):** run the **full 8 shards x both arches** so the
  benchmark website/history has complete per-arch series. This mirrors how `bench.yml` /
  `nightly-bench.yml` already gate the expensive, complete runs to `develop`/schedule.

Express this with an `arch.pr_subset` flag and an `if:` on the shard, or by composing the
matrix from a JSON input (the `fromJSON(inputs.benchmark_matrix)` pattern in
`sql-benchmarks.yml` line 292) so PR vs develop pass different matrices. Cost then scales as
roughly `(8 x_runs) + (3 arm64_smoke)` per PR vs `16` per develop push.

## 5. Workflow skeleton (YAML sketch)

Replaces the runner/build/run portion of `bench-codspeed`. Results emission (step "Emit
results artifact") is the **handoff to bullet 3** - format/upload owned there; here we only
guarantee a per-`(arch, shard)` artifact exists. yamllint-clean (2-space indent, no tabs,
quoted `on`-less here as it is a sketch fragment).

```yaml
jobs:
  bench-micro:
    name: "Microbench (${{ matrix.arch.id }} shard #${{ matrix.shard.id }})"
    timeout-minutes: 30
    strategy:
      fail-fast: false
      matrix:
        arch:
          - id: x64
            family: "family=c7i.4xlarge/spot=false"
            image: "ubuntu24-full-x64-pre-v2"
            rustflags: "-C target-cpu=x86-64-v3"
          - id: arm64
            family: "family=c7g.4xlarge/spot=false"
            image: "ubuntu24-full-arm64-pre-v2"
            rustflags: "-C target-cpu=neoverse-v1"
        shard:
          - { id: 1, packages: "vortex-buffer vortex-error vortex-mask" }
          - { id: 2, packages: "vortex-array", features: "--features _test-harness" }
          - { id: 3, packages: "vortex" }
          - { id: 4, packages: "vortex-alp vortex-bytebool vortex-datetime-parts" }
          - { id: 5, packages: "vortex-decimal-byte-parts vortex-fastlanes vortex-fsst", features: "--features _test-harness" }
          - { id: 6, packages: "vortex-pco vortex-runend vortex-sequence" }
          - { id: 7, packages: "vortex-sparse vortex-zigzag vortex-zstd" }
          - { id: 8, packages: "vortex-flatbuffers vortex-proto vortex-btrblocks" }
    runs-on: >-
      ${{ github.repository == 'vortex-data/vortex'
          && format('runs-on={0}/{1}/image={2}/tag=bench-micro-{3}-{4}',
                    github.run_id, matrix.arch.family, matrix.arch.image,
                    matrix.arch.id, matrix.shard.id)
          || 'ubuntu-latest' }}
    steps:
      - uses: runs-on/action@v2
        if: github.repository == 'vortex-data/vortex'
        with:
          sccache: s3
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6
      - name: Setup benchmark environment
        run: sudo bash scripts/setup-benchmark.sh
      - uses: ./.github/actions/setup-prebuild
      - uses: ./.github/actions/system-info
      - name: Install cargo-codspeed
        uses: taiki-e/cache-cargo-install-action@66c9585ef5ca780ee69399975a5e911f47905995
        with:
          tool: cargo-codspeed
      - name: Build benchmarks
        env:
          RUSTFLAGS: ${{ matrix.arch.rustflags }}
        run: >-
          cargo codspeed build ${{ matrix.shard.features }}
          $(printf -- '-p %s ' ${{ matrix.shard.packages }}) --profile bench
      - name: Run benchmarks (wall-time, pinned CPUs)
        run: bash scripts/bench-taskset.sh cargo codspeed run
      - name: Emit results artifact  # HANDOFF -> bullet 3 (format/upload owned there)
        uses: actions/upload-artifact@v4
        with:
          name: bench-micro-${{ matrix.arch.id }}-${{ matrix.shard.id }}
          path: target/codspeed/**  # placeholder path; bullet 3 defines schema
```

Notes on the sketch:

- Drops `CodSpeedHQ/action` (the SaaS dependency) and runs benches directly via
  `cargo codspeed run` under `bench-taskset.sh`. Whether we keep `cargo-codspeed` as the
  *runner* (it can run divan/criterion wall-time locally) or switch to plain
  `cargo bench`/divan is a bullet-3/harness decision; the runner-matrix mechanics here are
  identical either way.
- Uses `setup-prebuild` (already arch-aware via its `setup-rust` fallback) so the same step
  works on both AMIs.
- For the PR-vs-develop policy (section 4), wrap the arm64 full set in an `if:` or feed a
  reduced matrix on `pull_request` via the `fromJSON(inputs.benchmark_matrix)` pattern.

## 6. Open questions / decisions for the user

1. **Dedicated pool vs on-demand pinning for arm64.** Do we provision
   `bench-micro-arm64` (and `-x64`) dedicated runners in the private RunsOn config (strongest
   comparability, matches macro benches), or accept `family=c7g.4xlarge/spot=false` with
   occasional capacity retries?
2. **Exact pinned instance types.** `c7i.4xlarge` / `c7g.4xlarge` proposed (Graviton3). Do we
   want Graviton4 (`c8g`) and/or a metal type (like the macro benches' `i7i.metal-24xl`) for
   even tighter clock stability? More cores cost more but improve isolation.
3. **AVX-512 as a separate series?** x64 baseline is `x86-64-v3` (AVX2). Add an explicit
   AVX-512 matrix variant, or leave AVX-512 off entirely? Affects job count and bullet-3
   baselines.
4. **PR coverage of arm64.** Full 8 shards on both arches per PR (16 jobs, higher cost/latency)
   vs x64-full + arm64-smoke on PRs and full-both on develop (recommended). Confirm the smoke
   shard selection.
5. **Wall-time vs instruction-count.** CodSpeed gave deterministic instruction counts. Moving
   to multi-arch wall-time means run-to-run noise; how many iterations / what
   stability budget (coordinate with bullet 2) before a delta is "real"? Owned partly by
   bullets 2 and 3, but it constrains the timeout/cost here.
6. **CUDA shards.** Confirm GPU benches stay x64-only (`g5`) and out of the arch matrix.

## Sources

- [RunsOn job labels](https://runs-on.com/configuration/job-labels/)
- [RunsOn Linux runners (instance/label/price map)](https://runs-on.com/runners/linux/)
- [RunsOn spot instances](https://runs-on.com/configuration/spot-instances/)
- Repo: `.github/workflows/codspeed.yml`, `bench.yml`, `nightly-bench.yml`,
  `sql-benchmarks.yml`, `ci.yml`, `package.yml`; `.github/runs-on.yml`;
  `.github/actions/{setup-prebuild,setup-rust,system-info}`;
  `scripts/setup-benchmark.sh`, `scripts/bench-taskset.sh`;
  `Cargo.toml` (divan/criterion); `vortex-buffer/benches/vortex_bitbuffer.rs`.
