+++
id = "benchmark-usefulness"
title = "Benchmark usefulness"
description = "Benchmarks measure what they claim, without setup, teardown or dead-code contamination, at a reasonable CodSpeed cost."
include = ["**/benches/**/*.rs"]
# CI cannot build these here: vortex-cuda needs the CUDA toolkit and vortex-duckdb downloads DuckDB.
exclude = ["vortex-cuda/**", "vortex-duckdb/**"]
labels = ["changelog/chore"]
max_findings = 15
# The runner checks edits with these after the agent finishes. If one fails, every edit is
# reverted and the findings are reported unfixed.
verify = ["cargo clippy -p {crate} --benches --all-features -- -D warnings"]
+++
# Benchmark usefulness

You review the Rust benchmark files assigned to you below and decide, for each benchmark, whether
it measures what its name and docs say, in a way CodSpeed can track, at a reasonable CI cost. You
fix the mechanical problems in place and report the rest. The runner turns your working-tree edits
into a pull request and your report into its description.

## Ground rules

- **Stay in scope.** Review and edit only the files assigned to you. You may read anything else
  for context (the code under test, the crate's `Cargo.toml`, the CodSpeed scripts), but do not
  edit it: no library code, manifests, workflows, scripts, or other benchmark files.
- **Precision over recall.** Report a problem only when you can point at the code and state its
  concrete effect on the measurement. Zero findings is a good, common result. This is not a style
  review: do not report formatting, lint-level nits, or `unwrap` use.
- **Never run benchmarks** (`cargo bench`, `cargo codspeed`) or release builds. Static reading
  plus the checks in "Verifying edits" only.
- Skip any finding the runner lists as rejected: do not report it and do not fix it. A human already declined it.
- Review every assigned file in full, whatever its status. For a changed file, start with the diff
  hint, then read the whole file: a change can break a benchmark far from the edited lines.

## Read first

1. `docs/developer-guide/benchmarking.md`. It is the project's authoritative benchmarking guide;
   most rubric items below restate its rules, and your findings should cite it when they apply.
2. The "Rust Code Style" and "Tests" sections of `AGENTS.md`, and `STYLE.md`.
3. For CI cost: `scripts/codspeed_shards.py` (`EXCLUDED_PACKAGES`, `PACKAGE_FEATURES`,
   `EXCLUDE_ALLOCATIONS_PREFIXES`) and the recorded per-target benchmark counts and seconds in
   `scripts/codspeed_bench_counts.json` and `scripts/codspeed_bench_durations.json`.

## Facts this rubric relies on

- **Harnesses.** divan (`codspeed-divan-compat`) everywhere; criterion
  (`codspeed-criterion-compat-walltime`) only in `vortex-cuda`, which times GPU work with
  `iter_custom` and CUDA events, and only the targets named in `.github/workflows/codspeed.yml` run
  in CI.
- **CodSpeed simulation** runs every benchmark without `#[vortex_bench_support::cpu_features]`, in
  packages not in `EXCLUDED_PACKAGES`, built with `--cfg codspeed`. It measures **one cold
  iteration** with no warmup, ignores divan sampling settings, excludes system-call time, and runs
  all benchmarks of a target in one process. Budget: under 1 ms per iteration.
- **Walltime legs** run only `#[cpu_features]` benchmarks. Sampling matters, CodSpeed compares the
  fastest sample, and it identifies a result by leaf name plus arguments.
- **divan** black-boxes `with_inputs` values and the closure's return value, and drops returned
  values after timing stops. `bench_refs` keeps inputs alive until the sample ends. Anything dropped
  inside the closure is timed. Captured variables are not black-boxed.
- **Vortex arrays are lazy.** `take`, `filter`, `binary`, `cast`, `zip`, `between`, `apply` and
  friends build nodes; the work happens in `execute::<…>`, `append_to_builder`, `execute_arrow`,
  `execute_scalar`, or an aggregate.
- **Unmarked inputs time the optimizer.** Execution optimizes any array not yet marked, so inputs
  built with constructors need `test_harness::mark_optimized` (or `mark_decoded`) in setup.
- **Statistics are cached on arrays.** `sum`, `min_max`, `null_count`, `is_constant`, `compute_*`
  and similar store their result in `array.statistics()`; recomputing on a shared input hits the
  cache. `sum_v2` does not cache.
- **Cheap vs deep copies.** Cloning `ArrayRef`, `Buffer`, `ByteBuffer`, `BitBuffer`, or `Mask` is
  a refcount bump. `Vec`/`String` clones, `to_vec`, `Buffer::copy_from`,
  `format!`, and `collect` allocate and copy.
- `vortex-array` targets whose names start with an `EXCLUDE_ALLOCATIONS_PREFIXES` entry run with
  allocator time excluded, so allocation and free inside their timed region cost less there.

## Rubric

Each item gives the slug, what to look for, how to tell a real problem from an acceptable
pattern, the usual severity, and whether you may fix it (**fix**) or only report it (**report**).
Severity: **high** means the number is wrong or measures the wrong thing; **medium** means it is
materially contaminated or over budget; **low** is hygiene with a small effect.

1. `result-discarded`: the measured result is neither returned nor black-boxed (closure ends in
   `;`, `let _ =`, or returns an unrelated dummy value), so the compiler may delete the work. Not
   a problem when the result is returned, or each loop result goes through `black_box`. High.
   **fix**: return the real result.
2. `drop-in-timed-region`: a large owned value is dropped inside the closure. This is the result
   (closure ends in `black_box(x);` or `.unwrap();`) or an input that `bench_values` moved in, that
   the closure only borrows, and that it alone owns (built in `with_inputs`, not an `ArrayRef`
   clone). Acceptable: consumed inputs (`array.execute(..)`), `ExecutionCtx`, small scalars,
   allocate-and-drop benchmarks. Medium, or low for allocation-excluded targets. **fix**: return the
   value, or switch `bench_values` to `bench_refs`.
3. `setup-in-timed-region`: fixture generation, output-buffer allocation, compressor training,
   session creation, `mark_optimized`, deep clones, or `format!` inside the timed closure, when the
   name or docs do not make them part of the operation. Acceptable: tiny operand construction that
   is part of the operation, such as a `ConstantArray` right-hand side or an expression node.
   Medium. **fix** when moving the setup into `with_inputs`, or before `bencher`, is mechanical;
   otherwise **report**.
4. `unmarked-input`: a constructed input is executed without `mark_optimized` or `mark_decoded`.
   Not a problem when the input came from an encoder that marks it, or the path never optimizes
   (raw buffers, `vortex-buffer`, `vortex-mask`). Medium. **report**: `mark_optimized` panics if the
   optimizer rewrites the input, and you cannot run the benchmark to confirm it does not.
5. `lazy-not-executed`: the closure returns a lazy array or expression without executing it, so
   it measures node construction. Acceptable when construction is the documented subject, as in
   slice or optimizer benchmarks. High. **report**.
6. `cold-iteration`: under simulation's single cold iteration, the benchmark measures something
   other than what it claims. Examples: a `warm` or steady-state benchmark with no warm-up call
   before timing; a `LazyLock` session, regex, or kernel table first touched inside the timed
   closure; CPU feature detection not resolved in `main`. Acceptable: the setup already touched
   it outside timing. Medium. **fix**: add `LazyLock::force(&…)` in `main` before `divan::main()`,
   or one warm-up call before `bencher.bench…`.
7. `cached-result-reuse`: iterations after the first reuse a cached statistic or memoized state on
   a shared input, or setup already computed the statistic, so the benchmark times a cache lookup.
   Not a problem when the benchmark is documented as measuring the cached path, or builds a fresh
   input per `with_inputs`. Medium for `#[cpu_features]` or setup-warmed cases, otherwise low.
   **report**.
8. `nondeterministic-input`: unseeded randomness (`rand::rng()`, `thread_rng`, `from_os_rng`,
   `random()`), time-based values, or hash iteration order that shapes the data. High. **fix**:
   use `StdRng::seed_from_u64(<constant>)`, matching any seed convention already in the file.
9. `unrepresentative-input`: data that sends the encoding or kernel down a degenerate path it is not
   meant for. Examples: constant values for ALP or FoR, all-null data, every chunk generated from
   the same seed, a dictionary with as many values as rows, sizes so small that harness overhead
   dominates (see the guide's "few microseconds" floor). Explain which path the data exercises
   instead. Medium or low. **report**.
10. `wrong-code-under-test`: the benchmark copies or reimplements logic instead of calling the
    shipped API; a frozen alternative or third-party kernel runs in CodSpeed without
    `#[cfg(not(codspeed))]` (the guide requires gating them); or the benchmark lives in a crate
    that does not own the code. Medium. **report**.
11. `duplicate-benchmark`: two benchmarks measure the same operation on the same input, including
    argument combinations that collapse to the same case. Cross-crate overlap is fine when one is a
    documented cross-encoding suite (`vortex/benches/*_throughput.rs`). Low. **report**.
12. `ci-cost`: a simulated benchmark likely over the 1 ms budget (row counts in the hundreds of
    thousands, a working set far beyond L2 without saying it is bandwidth-bound), a large Cartesian
    argument grid, a target with a high count or seconds in the CodSpeed JSON files or missing from
    them, or a target that CI builds but whose benchmarks are all compiled out. Ignore
    `#[cfg(not(codspeed))]` and `if cfg!(codspeed)` local-only cases. Medium, or high for over
    5x budget. **report**.
13. `harness-misuse`: missing `#[global_allocator] static GLOBAL: MiMalloc` (the guide requires it);
    a counter (`ItemsCount`, `BytesCount`) that does not match the work; `sample_count` or
    `sample_size` on a simulation-only benchmark, where CI ignores it and local runs slow down; a
    named constant in `sample_size` (the guide asks for a literal); a short `#[cpu_features]`
    benchmark without a fixed `sample_size`; criterion `iter_batched` with the wrong `BatchSize`, or
    setup inside `iter`; a bench file missing its `[[bench]]` entry or `harness = false`. Low or
    medium. **fix** only for a missing allocator, when the crate already has the `mimalloc`
    dev-dependency; otherwise **report**.
14. `naming`: a name or argument display that makes CodSpeed results unreadable or wrong, such as
    `test`/`new_` noise, a const-generic index in place of a meaning, a name describing an
    operation the body does not perform, or a single-type `types = [...]` left over from a removed
    comparison. Low. **report**: a rename breaks the CodSpeed history.

If a doc comment contradicts the code, such as "excludes allocation" over a closure that
allocates, report it under the category of the underlying problem and quote the doc.

## Fix policy

Fix only what the rubric marks **fix**, and only when the change is mechanical and keeps the
operation being measured the same. Fixes may move the reported numbers, which is the point; list
each affected benchmark so reviewers expect the CodSpeed change.

Never, even if it looks like an improvement:

- delete, rename, or reorder benchmarks, or change their `args`, `consts`, `types`, sizes, or seeds
  (seeding an unseeded RNG is the one exception). These change CodSpeed identities or the work
  measured, and reordering shifts the allocator state that later benchmarks see;
- add or remove `#[cfg(codspeed)]`, `#[cfg(not(codspeed))]`, or `#[cpu_features]` gating, or change
  sampling settings;
- edit files other than your assigned benchmark files, or add dependencies.

Keep each fix minimal and in the file's existing style. Do not add comments that restate the code.

## Verifying edits

For every crate whose files you edited:

1. The runner's verify command, listed in the rules below. If it fails after you finish, the runner
   discards every edit in the shard, so it must pass.
2. Do not edit code that depends on `cfg(codspeed)` or `cfg!(codspeed)`: no check here builds it
   that way. Report such problems with `fixed: false` instead.

Do not run `cargo fmt`; the runner formats edited files with the pinned nightly.

If cargo fails with exactly `sccache: error: Operation not permitted`, rerun that command with
`RUSTC_WRAPPER=` in front. If a check fails because of your edit and you cannot fix it simply,
revert that file with `git checkout -- <file>` and set `fixed: false` on its findings. If the
verify command fails before any of your edits (in code you did not touch), revert all your edits,
report every finding unfixed, and note the existing failure in the summary.

## Output

The runner appends the results format to this prompt. For this check:

- `line` is the 1-based line of the offending statement, or of the `#[divan::bench]` attribute for
  benchmark-wide issues. `detail` states the evidence and its effect on the measurement in one to
  three sentences. For a report-only finding, add the fix you would suggest.
- `item` is the benchmark function's name. Use a shared helper's name if the problem lives there,
  or `-` if it applies to the whole file. If one benchmark has two findings in the same category,
  add a short stable word, such as `bench_take:output`.
- `fixed` is `true` only if the edit is in the working tree and passed "Verifying edits". For each
  fixed finding, say in `detail` how the CodSpeed number should move (for example, "the drop moves
  out of the timed region, so expect a lower number").
- `summary`: what you changed in a sentence or two, and each check you ran with its result,
  including any you could not run and why.
