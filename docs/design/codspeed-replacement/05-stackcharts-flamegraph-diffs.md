# 05 — Stackcharts with per-run diffs (Polar Signals? or something else)

Slice owner bullet: *"Stackcharts with diffs of each run, Polar Signals? or something else."*

Goal: produce a per-run flamegraph / stack chart for each benchmark run and a **diff** between
two runs (PR head vs. develop base, or run-over-run on develop) so a reviewer can see *where* a
regression came from, not just *that* one happened. This complements the numeric comparison
(bullet 3 storage, bullet 4 PR comment).

---

## 0. TL;DR recommendation

1. **Keep Polar Signals Cloud** as the continuous, low-overhead profiler for the *macro* binary
   benchmarks (`random-access-bench`, `compress-bench`, SQL/TPC-H, the `polarsignals` dataset
   bench). It is already wired, it already does pprof + native flamegraph **diff** UI, and the diff
   between two `gh_run_id` labels is one query. Nothing to build here except a **deep link** from
   the PR comment.
2. **CodSpeed replacement gap is the micro-benchmarks** (criterion, shards 1-8 in
   `codspeed.yml`, run under Valgrind in `simulation` mode). These currently have **no profile
   artifact at all** — CodSpeed produces the diff UI we'd be losing. For these, add a *separate
   sampling pass* of the same criterion binary using **`pprof-rs` `PProfProfiler`** (or `samply`),
   emit **collapsed/folded stacks + pprof**, store per `(commit, arch, bench)` in S3, and render a
   **differential flamegraph with `inferno-diff-folded` + `inferno-flamegraph`** as a static SVG
   served by `benchmarks-website`.
3. Cheapest fully-owned diff visual = **static inferno red/blue differential SVG in S3**. Richest
   interactive diff = **Polar Signals Cloud** (already paid for) or self-hosted **Parca**. Use both:
   PS Cloud link for macro, static inferno SVG for micro.

---

## 1. The two senses of "Polar Signals" in this repo (important — they are unrelated)

There are two completely different things named "Polar Signals" in the tree. Do not conflate them.

### (a) Polar Signals **the continuous-profiling product** (Parca / Polar Signals Cloud)

Already wired and live. The GitHub Action `polarsignals/gh-actions-ps-profiling@68ae857…` (v0.8.1)
runs an eBPF profiler agent on the runner during the benchmark and ships pprof samples to **Polar
Signals Cloud** project `e5d846e1-b54c-46e7-9174-8bf055a3af56`. It appears in three workflows:

- `.github/workflows/bench.yml` — "Setup Polar Signals" step, before the macro bench run
  (random-access, compression). Labels: `branch`, `gh_run_id`, `benchmark`.
- `.github/workflows/bench-pr.yml` — same step, gated `if: …head.repo.fork == false`.
- `.github/workflows/sql-benchmarks.yml` (lines ~379-388) — same step for SQL/TPC-H and the
  `polarsignals` dataset query benchmark.

Config in all three: `profiling_frequency: 199`, `extra_args: "--off-cpu-threshold=0.03"`. Token in
`secrets.POLAR_SIGNALS_API_KEY`. This is **CPU+off-CPU sampling of the whole bench process**, stored
in PS Cloud, queryable/diffable in their UI by label. It is *not* tied to criterion.

### (b) The **`polarsignals` benchmark dataset** (a workload, not a profiler)

`vortex-bench/src/polarsignals/` (`mod.rs`, `benchmark.rs`, `data.rs`, `schema.rs`) +
`vortex-bench/polarsignals.sql`. This is a **synthetic dataset modeled on Parca/Polar Signals
profiling data** — a `stacktraces` table with sparse nullable label columns, deeply nested
`List<Struct<…, List<Struct<…>>>>` locations, low-cardinality strings, sorted by labels then
`time_nanos` (`data.rs` header comment). The `.sql` file is TPC-H-style query workload Q0..Qn
against that table. It exercises Vortex's nested/sparse encodings. It is driven by the SQL bench
harness (matrix entry `id: polarsignals` / `subcommand: polarsignals` in
`sql-benchmarks.yml:244`).

**Distinction for the user:** the profiler product (a) gives us flamegraph diffs; the dataset (b) is
just one more benchmark whose runs *also* get profiled by (a). When we say "use Polar Signals for
stackcharts" we mean (a).

---

## 2. What kind of benchmark are we profiling? (this drives everything)

| Benchmark | Harness | How it runs in CI | Profile available today? |
|---|---|---|---|
| `random-access-bench`, `compress-bench` | standalone bin (`vortex-bench`) | `bench.yml` / `bench-pr.yml`, `release_debug` profile, wall-clock | **Yes** — PS Cloud sampling |
| SQL / TPC-H, `polarsignals` dataset | `vx-bench` orchestrator bins | `sql-benchmarks.yml`, wall-clock | **Yes** — PS Cloud sampling |
| **micro-benchmarks** (`vortex-array`, encodings, buffer…) | **criterion** | `codspeed.yml`, `cargo codspeed run`, `mode: simulation` | **No** — CodSpeed owns it |
| CUDA micro-benchmarks | criterion `walltime` | `codspeed.yml` `bench-codspeed-cuda` | No |

Key facts established from the repo:

- Micro-benches use **criterion** (`Cargo.toml:127 criterion = "0.8"`). CodSpeed runs them in
  `mode: "simulation"`, which is CodSpeed's **Valgrind/cachegrind instruction-count** measurement
  (deterministic instruction counts, *not* wall time). CUDA shards use `mode: "walltime"`.
- There is already a `[profile.samply]` profile in `Cargo.toml:392` (`inherits=release`,
  `debug="full"`, `codegen-units=1`) and a full **`.agents/skills/samply`** skill — the team
  routinely records Firefox-Profiler `profile.json.gz` and knows pprof/Firefox Profiler analysis.
- A **`bench-performance`** skill documents the local timing→profile loop. Reuse its conventions.

**The Valgrind subtlety:** instruction-count benches under Valgrind are *simulated execution* — you
cannot derive a meaningful sampled flamegraph from the cachegrind run itself, and timestamps are
fictional. Two viable ways to get a *stack chart* for a micro-bench:

1. **Separate sampling pass** of the *same* criterion binary (preferred). Build the bench bin once,
   then (a) run it under CodSpeed-replacement instruction counting for the *number*, and (b) run it
   again natively under `pprof-rs`/`samply` for the *flamegraph*. Criterion's
   `--profile-time <secs>` flag makes it loop the bench body for sampling instead of measuring.
2. **Callgrind annotation as the "stack chart."** If we keep Valgrind for the metric, run the same
   bin under `valgrind --tool=callgrind` and produce a `callgrind.out` → `gprof2dot`/`inferno` call
   tree. This is per-instruction-attributed and diffable, but heavier and less intuitive than a
   sampled flamegraph. Treat as fallback.

---

## 3. Options for capturing + diffing per-run profiles

### Option A — Polar Signals Cloud / Parca (the incumbent; pprof + built-in diff UI)

- pprof-native, continuous, <1% eBPF overhead, **built-in differential flamegraph (icicle) UI** and
  a query API; diff = pick two label sets (e.g. two `gh_run_id`s or `branch=develop` vs PR branch).
- Already capturing the macro benches. **For those, we get diffs for free** — the only work is
  constructing a deep link to the comparison view keyed by the run's labels.
- Self-host alternative: **Parca** (open source, free, pprof, same diff UI, default `:7070`,
  unauthenticated query API). Costs ops (a server + object storage).
- Weakness for micro-benches: the eBPF agent profiles the *whole process over wall time*; criterion
  micro-benches are sub-millisecond loops measured under Valgrind, so a wall-clock sampler attributes
  almost nothing useful unless we add a dedicated long sampling pass (see §2.1). It works, but it's a
  separate run, which somewhat negates "continuous/zero-config."

### Option B — `samply` + Firefox Profiler (already used by the team)

- `samply record <bench-bin> -- --bench --profile-time N` → `profile.json.gz`. Skill + `[profile.samply]`
  already exist.
- Sharing: `samply` opens `profiler.firefox.com` and can **Upload Local Profile** to get a permalink
  on `share.firefox.dev/…`; this can be scripted by gzip-POSTing to
  `https://api.profiler.firefox.com/compressed-store` and reading the returned token → hosted URL.
  Profiles can also be self-hosted/loaded from a URL.
- **Diff:** Firefox Profiler has a **Compare** view (load two profiles, "+ Compare…"), so head-vs-base
  comparison is possible but is *not* a one-click shareable artifact and the compare UX is manual.
- Best fit: keep `samply` as the **local developer** loop (matches existing skill); not ideal as the
  automated CI diff artifact.

### Option C — `cargo-flamegraph` / **inferno** (native DIFFERENTIAL flamegraphs — best for static CI artifact)

- `inferno` (Rust port of Brendan Gregg's FlameGraph) ships **`inferno-diff-folded`** (port of
  `difffolded.pl`) and **`inferno-flamegraph`**. Pipeline:
  1. produce **folded/collapsed stacks** for base and head (from pprof via `pprof`'s
     `Output::Flamegraph`/collapsed, from `perf` via `inferno-collapse-perf`, or from a
     `profile.json.gz`).
  2. `inferno-diff-folded base.folded head.folded > diff.folded`
  3. `inferno-flamegraph --negate < diff.folded > diff.svg` → **red/blue differential SVG**
     (red = got hotter on head, blue = got colder). `flamegraph.pl`/inferno auto-detect 3-column
     input.
- Output is a single self-contained **SVG** — trivially committable to S3 and `<img>`/iframe-embeddable
  in `benchmarks-website`. No server, no account, fully owned. **This is the cheapest diff visual.**
- Downside: static (no zoom/search beyond SVG's built-in JS), and "differential" colorizing can be
  unintuitive when the two stacks diverge structurally. Mitigate by also linking a Firefox/PS view.

### Option D — `pprof-rs` + speedscope

- **`pprof-rs`** (`tikv/pprof-rs`) provides `pprof::criterion::PProfProfiler` that plugs straight into
  criterion: `Criterion::default().with_profiler(PProfProfiler::new(199, Output::Protobuf))`. With
  `--profile-time N`, criterion writes `target/criterion/<bench>/profile.pb` (pprof) and/or
  `flamegraph.svg`. This is the **lowest-friction way to get a per-micro-bench pprof** with no extra
  process — and pprof feeds both inferno (Option C) and Parca/PS (Option A).
- **speedscope** offers a "Left Heavy" view (great for single-profile reading) but **A/B comparison is
  an open, unimplemented feature request** (`jlfwong/speedscope#445`). So speedscope = nice single-run
  viewer, **not** a diff tool. Use inferno or Firefox/PS for the diff.

**Verdict on capture:** add `pprof-rs PProfProfiler` to the criterion harness → pprof per micro-bench
(reused for both inferno static diff and optional PS/Parca upload). Keep PS Cloud for macro. Keep
`samply` for local dev.

---

## 4. Recommended approach

### (a) Storage — one profile artifact per `(commit, arch, benchmark)`

Coordinate with bullet 3 (S3 results store, bucket `vortex-ci-benchmark-results`, role
`arn:aws:iam::245040174862:role/GitHubBenchmarkRole`, already used in every bench workflow). Store
**pprof** (and the derived **folded** text) as the canonical blobs:

```
s3://vortex-ci-benchmark-results/profiles/<arch>/<benchmark-id>/<commit-sha>.pb.gz      # pprof
s3://vortex-ci-benchmark-results/profiles/<arch>/<benchmark-id>/<commit-sha>.folded.gz  # collapsed stacks
```

pprof is the durable interchange format (re-renderable by inferno, Parca, PS, speedscope, pprof
toolchain). Folded is cached so diff rendering is cheap. Reuse `scripts/cat-s3.sh` /
`scripts/s3-download.py` patterns already in the workflows.

### (b) Generating the DIFF (base vs head)

The base SHA is already computed in `bench-pr.yml` ("Compare results" step) by querying the latest
successful `bench.yml` develop run and `grep`ing it out of `data.json.gz`. Reuse exactly that to pick
the base commit, then:

```bash
aws s3 cp s3://…/profiles/<arch>/<bench>/<base_sha>.folded.gz - | gunzip > base.folded
# head.folded was just produced this run
inferno-diff-folded base.folded head.folded > diff.folded
inferno-flamegraph --negate --title "<bench> head vs base" < diff.folded > diff.svg
```

For macro benches, *skip* local diffing and instead build a **PS Cloud comparison URL** (two
`gh_run_id` label sets) — PS renders the diff server-side.

### (c) Surfacing it

- **PR comment (bullet 4):** the existing `thollander/actions-comment-pull-request` comment already
  carries the numeric table. Append a "Profiles" line:
  - micro: link to the static `diff.svg` in S3 / on `benchmarks-website`.
  - macro: deep link to the Polar Signals Cloud differential flamegraph for this run's labels.
- **`benchmarks-website`:** it is a Docker-served Vite/React site (`benchmarks-website/`,
  `Dockerfile`, `server.js`, published by `publish-benchmarks-website.yml`). Add a per-benchmark
  "Flamegraph diff" panel that either `<iframe>`s the PS Cloud view or `<object>`/`<img>`-embeds the
  S3 SVG keyed by the commit the user is viewing. SVGs are static so they can be lazy-loaded.

---

## 5. Cost / ops tradeoffs

| Approach | $ | Ops burden | Diff UX | Ownership |
|---|---|---|---|---|
| **Static inferno red/blue SVG in S3** | ≈ S3 storage only | None (just CI + S3) | Static SVG, no zoom/search | Full |
| **Self-host Parca** | server + object store | Run/upgrade a service, auth, retention | Rich interactive icicle diff, query API | Full |
| **Polar Signals Cloud** | consumption pricing (data volume / instances), already paying | None (managed) | Best-in-class diff UI, VS Code ext, Grafana plugin | Vendor |
| **samply + Firefox Profiler share** | free (Mozilla-hosted) | Low | Manual Compare view, not 1-click | Mozilla-hosted blobs |

Recommendation: **inferno static SVG (micro, fully owned, cheapest) + Polar Signals Cloud (macro,
already paid, best UX).** Revisit self-hosting Parca only if PS Cloud cost becomes a concern *and*
we want interactive diffs for micro-benches too — Parca consumes the same pprof we'd already store,
so adopting it later is incremental.

---

## 6. Concrete pipeline sketch

### Micro-benchmarks (the CodSpeed replacement path)

```
codspeed.yml replacement job (per shard, per arch):
  1. cargo build the criterion benches once (existing `cargo codspeed build` → reuse the bin,
     or `cargo bench --no-run`).
  2. metric pass  : run under instruction-counting (CodSpeed-replacement / iai-callgrind) → numbers
  3. profile pass : run the SAME bin natively with pprof-rs PProfProfiler, e.g.
        target/.../<bench> --bench --profile-time 5
     → target/criterion/<bench>/profile.pb  (pprof)
  4. convert pprof → folded; gzip both; upload to
        s3://…/profiles/<arch>/<bench>/<sha>.{pb,folded}.gz   (reuse cat-s3.sh)
  5. (PR only) fetch base <sha>.folded from S3 (base SHA via existing bench-pr.yml logic)
        inferno-diff-folded base.folded head.folded | inferno-flamegraph --negate > diff.svg
        upload diff.svg → s3://…/profiles/diffs/<bench>/<pr>-<sha>.svg
  6. append link to the PR comment; benchmarks-website embeds the SVG.
```

### Macro / SQL benchmarks (keep PS Cloud)

```
bench.yml / bench-pr.yml / sql-benchmarks.yml (unchanged capture):
  - "Setup Polar Signals" already ships pprof to PS Cloud with labels
    branch / gh_run_id / benchmark.
  - NEW: emit a PS Cloud comparison deep link (head gh_run_id vs base develop gh_run_id)
    into the PR comment so reviewers jump straight to the differential icicle graph.
```

---

## 7. Open questions / decisions for the user

1. **Which CodSpeed-replacement metric** for micro-benches drives bullet 1/2 — iai-callgrind
   (Valgrind instruction counts, deterministic) vs criterion walltime on dedicated runners? This
   decides whether the *metric* pass and the *profile* pass are one run or two (§2).
2. **One extra profiling pass acceptable?** It roughly doubles micro-bench CI time for the profiled
   subset. Acceptable to profile only a curated subset, or only on regression, rather than every
   bench every run?
3. **Static SVG vs interactive** for micro-bench diffs — ship cheap inferno SVGs now, or invest in
   self-hosted Parca for interactive micro diffs?
4. **Keep paying for Polar Signals Cloud** for macro, or migrate macro to self-hosted Parca to
   consolidate on one fully-owned stack?
5. **Where do diff SVGs/links live in the PR comment** — one combined comment (bullet 4) or a
   separate "Profiles" comment? And retention policy for pprof blobs in S3 (they are large).
6. **Symbolication on `release_debug`/`bench` profiles** — confirm `debug="full"` (already set in
   `[profile.bench]` and `[profile.samply]`) is applied to the bins we profile so stacks symbolicate.

---

## References (files in repo)

- `.github/workflows/codspeed.yml` — micro-bench (criterion, `simulation`/Valgrind) — the gap.
- `.github/workflows/bench.yml`, `bench-pr.yml`, `sql-benchmarks.yml` — "Setup Polar Signals" steps
  (PS Cloud capture for macro/SQL), S3 upload, base-SHA comparison logic.
- `.github/workflows/publish-benchmarks-website.yml`, `benchmarks-website/` — embed target.
- `vortex-bench/src/polarsignals/` + `polarsignals.sql` — the *dataset* benchmark (sense (b)).
- `Cargo.toml:127` (criterion 0.8), `Cargo.toml:392` (`[profile.samply]`), `[profile.bench]`.
- `.agents/skills/samply/SKILL.md`, `.agents/skills/bench-performance/SKILL.md` — existing
  profiling know-how.
- `scripts/bench-taskset.sh`, `scripts/cat-s3.sh`, `scripts/s3-download.py` — reuse for capture/store.

## References (external)

- Brendan Gregg, Differential Flame Graphs — https://www.brendangregg.com/blog/2014-11-09/differential-flame-graphs.html
- FlameGraph `difffolded.pl` — https://github.com/brendangregg/FlameGraph/blob/master/difffolded.pl
- inferno (`inferno-diff-folded`, `inferno-flamegraph`) — https://github.com/jonhoo/inferno / https://docs.rs/inferno
- pprof-rs criterion `PProfProfiler` — https://github.com/tikv/pprof-rs/blob/master/examples/criterion.rs
- Parca (self-host, pprof, diff) — https://github.com/parca-dev/parca
- Polar Signals Cloud (managed, differential flamegraphs) — https://www.polarsignals.com/blog/posts/2023/10/10/polarsignals-cloud-ga
- samply (record, Firefox Profiler upload/share) — https://github.com/mstange/samply
- Firefox Profiler upload/compare docs — https://github.com/firefox-devtools/profiler/blob/main/docs-developer/loading-in-profiles.md
- speedscope (Left Heavy; A/B comparison open issue #445) — https://github.com/jlfwong/speedscope / https://github.com/jlfwong/speedscope/issues/445
