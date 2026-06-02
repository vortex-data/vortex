<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 04 — PR Comment + Diff View (CodSpeed replacement)

Bullet: **"A nice GitHub comment and diff view, impl here."**

This slice replaces two CodSpeed surfaces for the **microbenchmarks** currently
run by `.github/workflows/codspeed.yml`:

1. The auto-posted PR comment (per-benchmark % delta vs base, with regression
   flags).
2. The CodSpeed web report / diff view linked from that comment.

It is designed against the infrastructure this repo *already has* for **macro**
benchmarks: a sticky-comment workflow (`bench-pr.yml`), a Python markdown
generator (`scripts/compare-benchmark-jsons.py`), an S3 results store
(`vortex-ci-benchmark-results`), and the v3 Rust benchmarks-website
(`benchmarks-website/server/`, an axum + DuckDB SSR site). The plan reuses these
patterns rather than inventing new ones.

> Scope boundary: *getting the head + base microbench measurements* (running
> divan/criterion benches off-CodSpeed, storing them, picking the develop base)
> is **bullet 3's** job. This doc consumes those two result sets and owns (a) the
> comparison + markdown generation, (b) the workflow that posts the comment
> safely, and (c) the website diff/compare view + deep links. Where bullet 3's
> storage shape is load-bearing here, it is called out as a dependency.

---

## 1. How CodSpeed works today (so we match or beat it)

### What `codspeed.yml` posts today

`.github/workflows/codspeed.yml` runs `cargo codspeed run` across 8 CPU shards
(`vortex-buffer`, `vortex-array`, `vortex` …) plus 3 CUDA walltime shards, on
`push: develop`, every `pull_request`, and `workflow_dispatch`. It uploads to
CodSpeed via `CodSpeedHQ/action` with `secrets.CODSPEED_TOKEN`. The repo's only
job is to *produce* measurements; **all** comment/diff UX is CodSpeed-hosted —
there is no in-repo markdown for microbenches today. The harness is
`codspeed-divan-compat` (`Cargo.toml:152`, `divan = { package =
"codspeed-divan-compat" }`) plus `criterion = "0.8"`; the `cfg(codspeed)` cfg
(`Cargo.toml:327`) switches those benches into instrumentation mode only when
built under `cargo codspeed`. Built normally they are ordinary
divan/criterion binaries that emit wall-clock timings — which is exactly the
hook bullet 3 uses to get measurements without CodSpeed.

### CodSpeed's PR UX (from CodSpeed docs)

- A single GitHub-App comment per PR, **updated in place** (sticky), linking to
  a hosted report. "CodSpeed posts a comment … with the run report; clicking it
  redirects you to the CodSpeed report of the PR."
- **Base selection = the PR's merge-base on the default branch.** Head is
  compared against the most recent base run for that merge-base commit. We
  approximate this today for macro benches by taking the latest *successful*
  `bench.yml` run on `develop` (`bench-pr.yml` lines 101–110), not a true
  merge-base; see Open Questions.
- **Regression threshold** is configurable project-wide and per-benchmark; the
  check fails when a benchmark "overshoots your regression threshold," and can be
  wired as a **required status check** ("Require status checks to pass before
  merging"). Per-benchmark thresholds are the recommended noise tool rather than
  blanket ignores.
- **Ignored benchmarks** are excluded from the regression gate (CodSpeed
  "Ignoring a Benchmark").
- Status filter buckets each row as regression / improvement / unchanged.

Sources: [CodSpeed reporting](https://codspeed.io/docs/features/reporting),
[performance checks](https://docs.codspeed.io/features/performance-checks),
[ignoring benchmarks](https://codspeed.io/docs/features/ignoring-benchmarks),
[noise/macro runners](https://codspeed.io/blog/benchmarks-in-ci-without-noise).

### What we already do better than a naive copy

`scripts/compare-benchmark-jsons.py` (macro) already beats CodSpeed's flat table
on noise handling: log-ratio effects, a parquet **control** to subtract systemic
drift, a conservative **Z=2.576 (99%)** noise floor, median-polish, and a
collapsed verdict (`Likely regression / improvement / No clear signal` with
high/medium/low confidence). The microbench comment should inherit the *shape*
(sticky comment, `<details>` groups, emoji status, geomean summary) but use a
simpler model — microbenches have no parquet control row, so the noise floor
comes from per-iteration variance (divan/criterion already give us the sample
distribution) plus a min-effect threshold.

---

## 2. PR comment design

One **sticky** comment per PR (single, updated in place via a marker), grouped by
**arch** (the codspeed matrix is `amd64` CPU + `g5/CUDA`; group by the
`env_triple` already in the v3 schema, e.g. `x86_64-linux-gnu` vs the CUDA host).
Layout mirrors `compare-benchmark-jsons.py`'s output so the two comments feel
like siblings.

Structure:

- **Header + summary line**: `N regressed · M improved · K unchanged · J new`
  plus a geomean line and a one-word verdict.
- Per-arch `<details>` block (collapsed unless it has regressions) with a table:
  `Benchmark | base | head | Δ% | status`.
- Status emoji: ✅ improved (faster, past threshold), ⚠️ regressed within a
  "soft" band, 🔴 regressed past the hard gate, ➖ unchanged/noise, 🆕 new bench.
- Footer with deep links to the website compare view (per-bench + PR-level) and
  the workflow run.

### Example rendered comment

```markdown
<!-- vortex-microbench-comment -->
## Microbenchmarks

**Verdict**: 🔴 Regression likely (medium confidence) ·
**3 regressed · 5 improved · 412 unchanged · 1 new**
Geomean Δ (non-noise): **+1.8%** · base `a1b2c3d4` · head `e5f6a7b8`
[Full compare ↗](https://bench.vortex.dev/compare?base=a1b2c3d4&head=e5f6a7b8&pr=1234)

<details open>
<summary>x86_64-linux-gnu (3 regressed, 5 improved)</summary>

| Benchmark | base | head | Δ% | status |
|-----------|-----:|-----:|---:|:------:|
| [vortex-array/take_primitive/u32/65536](https://bench.vortex.dev/micro/take_primitive%2Fu32%2F65536?base=a1b2c3d4&head=e5f6a7b8) | 8.41 µs | 9.92 µs | +18.0% | 🔴 |
| [vortex-fastlanes/canonicalize/1024](https://bench.vortex.dev/micro/canonicalize%2F1024?base=a1b2c3d4&head=e5f6a7b8) | 1.12 ms | 1.19 ms | +6.3% | ⚠️ |
| [vortex/pipeline/compress/taxi](https://bench.vortex.dev/micro/pipeline%2Fcompress%2Ftaxi?base=a1b2c3d4&head=e5f6a7b8) | 44.2 ms | 39.1 ms | −11.5% | ✅ |
| vortex-buffer/bitbuffer/true_count/16384 | 612 ns | 610 ns | −0.3% | ➖ |
| vortex-zigzag/encode/i64/65536 *(new)* | — | 2.04 µs | — | 🆕 |

</details>

<details>
<summary>x86_64-linux-gnu+cuda (0 regressed, 0 improved)</summary>
…
</details>

<sub>Δ% = median head/base − 1. 🔴 ≥ +10% past noise floor · ⚠️ ≥ +5% ·
✅ ≤ −5% · ➖ within noise. Per-bench rows link to history + base↔head diff.
Posted by `microbench-comment.yml` · [run ↗](…)</sub>
```

Only the rows that link should link (regressions/improvements); unchanged rows
stay plain text to keep the comment scannable. Tables longer than ~40 rows
collapse the unchanged section into a nested `<details>`.

---

## 3. Mechanism: which workflow posts it, base acquisition, fork safety

### 3.1 Getting base + head without re-running base

The clean path reuses the **v3 ingest store** that bullet 3 will write microbench
results into (the DuckDB site already stores `value_ns` + `all_runtimes_ns` per
`(commit_sha, dim)` — see `benchmarks-website/server/src/schema.rs`). Then:

- **Head**: built + run in the PR workflow → a `--gh-json-v3`-style JSONL (the
  same shape `vortex-bench` emits today, `vortex-bench/src/v3.rs`). We do **not**
  need a microbench fact-table yet; a sixth `micro_measurements` family (or
  reusing a generic timing table) is bullet 3's call. The comparison script only
  needs `{name, value_ns, all_runtimes_ns, env_triple}`.
- **Base**: **fetched, not re-run.** Resolve the merge-base SHA, then pull that
  commit's stored microbench rows. Two acquisition options, pick per bullet 3:
  - **(A) Query the website** — add a read endpoint
    `GET /api/micro/commit/{sha}` (or reuse `/api/chart/{slug}` per bench) that
    returns the stored series value at a commit. Cheap, no S3 round trip.
  - **(B) S3 grep** — mirror the macro path exactly: `s3-download.py
    s3://vortex-ci-benchmark-results/<micro>.json.gz` then `grep <base_sha>`
    (`bench-pr.yml` lines 109–110). Zero new server code.
- **Fallback when base is missing** (merge-base never ran microbenches): emit a
  comment that says "no base data — informational only," never gate. This mirrors
  the macro `_No baseline … found_` branch in `compare-benchmark-jsons.py`.

Re-running base on the PR runner (two-run compare) is the costly fallback and
should be reserved for when neither store has the merge-base — flag it as an
explicit, opt-in mode (label `action/microbench-rerun-base`).

### 3.2 Fork safety (mirror existing repo patterns)

The repo has two established patterns:

- `bench-pr.yml` gates *all* privileged steps on
  `github.event.pull_request.head.repo.fork == false` (S3 OIDC, Polar Signals,
  `Comment PR`). The benchmark *runs* on forks but never gets AWS creds or posts.
- `claude-review.yml` refuses fork PRs outright at a `gate` job and uses the
  built-in `github.token` read-only, never the App key.

For microbenches we want fork PRs to still get a comment (unlike Claude review),
so adopt the **`workflow_run` split** — the safest pattern for "run untrusted
code, then post with write perms":

1. **`microbench.yml`** (`on: pull_request`, `permissions: contents: read`)
   builds + runs the divan/criterion benches on the PR head, generates
   `comment.md` + `summary.json` with the comparison script, and uploads them as
   an **artifact**. It has *no* write token and *no* AWS creds, so a malicious
   fork PR cannot post a comment or touch S3. Base data is fetched read-only
   (public S3 `--no-sign-request`, as `bench-pr.yml` already does, or the public
   website API).
2. **`microbench-comment.yml`** (`on: workflow_run`, `types: [completed]`,
   `permissions: pull-requests: write`) downloads the artifact from the
   triggering run and posts the sticky comment. Code here is from the **base
   branch** (trusted), not the PR, so even fork PRs are posted safely. This is
   the canonical GitHub-recommended pattern and avoids `pull_request_target`
   (which would check out untrusted code with a write token — the thing
   `claude-review.yml` explicitly refuses).

For *same-repo* PRs we could keep it single-workflow like `bench-pr.yml`
(`thollander/actions-comment-pull-request` with a `comment-tag`). But forks are
the whole reason CodSpeed's app exists; the `workflow_run` split gets fork
coverage for free, so prefer it uniformly.

### 3.3 Sticky-comment mechanism (already in the repo)

`bench-pr.yml` uses `thollander/actions-comment-pull-request@v3` with
`comment-tag: bench-pr-comment-<id>` and `file-path: comment.md`. Reuse the same
action with `comment-tag: microbench-comment` (one sticky comment, updated in
place). The HTML marker `<!-- vortex-microbench-comment -->` in the example is
belt-and-suspenders so the comment is also locatable via the GitHub API if the
action is ever swapped for `actions/github-script`.

### 3.4 Workflow sketches (yamllint-clean, pinned SHAs as `# TODO pin`)

```yaml
# .github/workflows/microbench.yml
# Runs microbenchmarks on the PR head and uploads the rendered comment as an
# artifact. Read-only: never posts, never touches AWS. Safe for fork PRs.
name: Microbenchmarks

concurrency:
  group: ${{ github.workflow }}-${{ github.head_ref || github.run_id }}
  cancel-in-progress: true

on:
  pull_request: { }
  workflow_dispatch: { }

permissions:
  contents: read

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: "1"

jobs:
  micro:
    timeout-minutes: 30
    runs-on: >-
      ${{ github.repository == 'vortex-data/vortex'
          && format('runs-on={0}/runner=amd64-medium/image=ubuntu24-full-x64-pre-v2/tag=microbench-{1}', github.run_id, matrix.shard)
          || 'ubuntu-latest' }}
    strategy:
      fail-fast: false
      matrix:
        include:
          - { shard: 1, packages: "vortex-buffer vortex-error vortex-mask" }
          - { shard: 2, packages: "vortex-array" }
          # ... same shard split as codspeed.yml ...
    steps:
      - uses: runs-on/action@v2  # TODO pin
        if: github.repository == 'vortex-data/vortex'
        with:
          sccache: s3
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      - name: Setup benchmark environment
        run: sudo bash scripts/setup-benchmark.sh
      - uses: ./.github/actions/setup-prebuild
      - uses: ./.github/actions/system-info
      - name: Run microbenchmarks (head)
        env:
          RUSTFLAGS: "-C target-feature=+avx2"
        run: |
          bash scripts/bench-taskset.sh \
            cargo run -p vortex-bench --bin run-microbench -- \
            --packages "${{ matrix.packages }}" --gh-json-v3 head.jsonl
      - name: Resolve merge-base
        id: base
        run: |
          set -Eeu -o pipefail
          base_sha=$(git merge-base \
            "origin/${{ github.event.pull_request.base.ref }}" \
            "${{ github.event.pull_request.head.sha }}")
          echo "sha=$base_sha" >> "$GITHUB_OUTPUT"
      - name: Fetch base measurements (read-only)
        run: |
          set -Eeu -o pipefail
          python3 scripts/s3-download.py \
            s3://vortex-ci-benchmark-results/microbench.json.gz \
            micro.json.gz --no-sign-request
          gzip -dc micro.json.gz | grep "${{ steps.base.outputs.sha }}" > base.jsonl || true
      - name: Generate comment
        run: |
          set -Eeu -o pipefail
          uv run --no-project scripts/compare-microbench-jsons.py \
            --base base.jsonl --head head.jsonl \
            --base-sha "${{ steps.base.outputs.sha }}" \
            --head-sha "${{ github.event.pull_request.head.sha }}" \
            --pr "${{ github.event.pull_request.number }}" \
            --shard "${{ matrix.shard }}" \
            --out-md "comment-${{ matrix.shard }}.md" \
            --out-json "summary-${{ matrix.shard }}.json"
      - uses: actions/upload-artifact@v4  # TODO pin
        with:
          name: microbench-comment-${{ matrix.shard }}
          path: |
            comment-${{ matrix.shard }}.md
            summary-${{ matrix.shard }}.json
            head.jsonl
          retention-days: 3
```

```yaml
# .github/workflows/microbench-comment.yml
# Posts/updates the sticky microbench comment after microbench.yml finishes.
# Runs from the base branch (trusted code) with write perms, so fork PRs are
# posted safely. Mirrors the workflow_run pattern, not pull_request_target.
name: Microbenchmark Comment

on:
  workflow_run:
    workflows: ["Microbenchmarks"]
    types: [completed]

permissions:
  contents: read
  pull-requests: write

jobs:
  comment:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    if: github.event.workflow_run.event == 'pull_request'
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6
      - name: Download artifacts from the triggering run
        uses: actions/download-artifact@v4  # TODO pin
        with:
          run-id: ${{ github.event.workflow_run.id }}
          github-token: ${{ github.token }}
          pattern: microbench-comment-*
          path: artifacts
          merge-multiple: true
      - name: Resolve PR number + merge shards
        id: merge
        run: |
          set -Eeu -o pipefail
          pr=$(cat artifacts/summary-1.json | jq -r '.pr_number')
          echo "pr=$pr" >> "$GITHUB_OUTPUT"
          cat artifacts/comment-*.md > final-comment.md
      - name: Post sticky comment
        uses: thollander/actions-comment-pull-request@24bffb9b452ba05a4f3f77933840a6a841d1b32b  # v3
        with:
          file-path: final-comment.md
          comment-tag: microbench-comment
          pr-number: ${{ steps.merge.outputs.pr }}
```

> yamllint notes (per `.github/CLAUDE.md` / `.yamllint.yaml`): empty mappings as
> `{ }`, two-space indents, quote the leading-`1` env (`RUST_BACKTRACE: "1"`),
> name every job step. The two sketches above follow `codspeed.yml` /
> `bench-pr.yml` conventions so `yamllint --strict` passes.

### 3.5 Comparison + markdown script (`scripts/compare-microbench-jsons.py`)

Language: **Python**, consistent with `scripts/compare-benchmark-jsons.py`
(macro) and the existing `uv run --no-project` invocation pattern. (A Rust
subcommand in `vortex-bench` is viable too, but the markdown/stats logic already
lives in Python and the comment is not perf-sensitive.) Sketch:

```python
# /// script  (uv inline metadata, like compare-benchmark-jsons.py)
# dependencies = ["numpy"]
# ///
# Inputs: --base/--head JSONL of {name, value_ns, all_runtimes_ns, env_triple}.
# Output: a sticky-comment .md grouped by env_triple, plus a summary.json the
# comment workflow uses to find the PR number + gate status.

MIN_EFFECT = 0.05          # 5% — below this everything is "unchanged"
HARD_REGRESSION = 0.10     # 10% — 🔴 gate band (matches CodSpeed default UX)
Z = 1.96                   # 95% per-bench band from all_runtimes_ns variance

def classify(base_ns, head_ns, base_runs, head_runs):
    if base_ns is None:
        return "new", None
    ratio = head_ns / base_ns
    delta = ratio - 1.0
    # noise floor: max(min-effect, sampling band from per-iteration logs)
    floor = max(MIN_EFFECT, Z * pooled_log_se(base_runs, head_runs))
    if abs(delta) < floor:
        return "unchanged", delta
    if delta >= HARD_REGRESSION:
        return "regressed_hard", delta   # 🔴
    if delta > 0:
        return "regressed_soft", delta   # ⚠️
    return "improved", delta             # ✅

# - group rows by env_triple (arch); within a group sort regressions first.
# - geomean of non-"unchanged" ratios -> headline.
# - per-row deep links: COMPARE_BASE = "https://bench.vortex.dev"
#   row link = f"{COMPARE_BASE}/micro/{quote(name)}?base={base_sha}&head={head_sha}"
#   header link = f"{COMPARE_BASE}/compare?base={base_sha}&head={head_sha}&pr={pr}"
# - emit summary.json: {pr_number, base_sha, head_sha, n_regressed_hard, ...}
#   so the gate step (section 4) reads one file.
```

Reuse from the macro script: `format_ratio_change`, the `<details>`/`<summary>`
table idiom, `to_markdown(tablefmt="github")`, and the geomean helper. Drop the
parquet-control / median-polish machinery — microbenches have no control series.

---

## 4. Threshold / gating policy

Tie the bands to section 3.5:

| Band | Condition | Emoji | Gates? |
|------|-----------|:-----:|--------|
| Improved | `Δ ≤ −5%` past noise floor | ✅ | no |
| Unchanged | `|Δ| < max(5%, 95% sampling band)` | ➖ | no |
| Soft regression | `+5% ≤ Δ < +10%` past floor | ⚠️ | no (warn only) |
| Hard regression | `Δ ≥ +10%` past floor | 🔴 | optional |
| New | no base data | 🆕 | no |

- **Noise handling**: every classification first clears a per-bench 95% sampling
  band derived from `all_runtimes_ns` (we have the distribution; CodSpeed only
  exposes a single threshold). This is strictly better than a flat percent and
  reuses the log-SE idea already in `compare-benchmark-jsons.py`
  (`log_runtime_stats`, `ratio_stats`). A blanket `MIN_EFFECT = 5%` floors out
  sub-noise microbenches that are inherently jittery.
- **Per-bench thresholds / ignores**: support an in-repo
  `benchmarks/microbench-thresholds.toml` (`[ignore]` list + per-bench override
  percent), mirroring CodSpeed's "ignoring a benchmark" + per-benchmark
  threshold. Ignored benches render as ➖ and never gate.
- **When to fail the check**: start **non-gating** — `microbench.yml` always
  succeeds, the comment is informational, like the macro `bench-pr.yml` today.
  Add an *optional* `microbench-gate` job in `microbench-comment.yml` that
  `core.setFailed`s when `summary.json.n_regressed_hard > 0` **and** the PR
  carries a `perf-sensitive` label, so authors opt in. This avoids the classic
  microbench-flake-blocks-merge problem while still giving a hard signal where it
  matters.
- **Required status check?** Not initially. Microbench noise on shared CI
  runners makes a hard required check a merge-blocker hazard. Recommendation:
  keep it advisory; only promote the gate job to *required* (branch protection)
  after we have several weeks of measured run-to-run variance on the dedicated
  `runs-on` bench runners and the 95% band proves stable (Open Question).

---

## 5. Web diff view (benchmarks-website)

The v3 site (`benchmarks-website/server/`, axum + DuckDB SSR, `html/` +
`api/`) has **no compare/PR view today** (grep for `compare`/`permalink` found
only chart permalinks, `html/chart.rs`). Two additions:

### 5.1 Per-benchmark history + sparkline (mostly exists)

Microbenches need a fact family + chart slug (bullet 3). Once present, the
existing per-chart page (`GET /chart/{slug}`, `html/chart.rs`) already renders a
Chart.js time series with the latest-100 materialized window and `?n=all`
zoom-out. The PR comment's per-row links target a thin wrapper:

- `GET /micro/{name}?base=<sha>&head=<sha>` → resolves `name` to its `ChartKey`,
  renders the existing chart card, and **overlays two vertical markers** (base,
  head) on the series. Implementation: extend `chart_body` (`html/chart.rs`) to
  accept optional `highlight_shas: &[String]` and pass them into the inlined
  `chart-data-0` JSON so `static/chart-init.js` draws annotation lines. No new
  data path — it reuses `/api/chart/{slug}`.

### 5.2 Base-vs-head compare page (new)

- `GET /compare?base=<sha>&head=<sha>&pr=<n>` → a page that for **every**
  microbench computes `head/base` at the two pinned commits and renders the same
  table as the PR comment, but interactive (sortable, filter to regressions).
- New read endpoint `GET /api/compare?base=<sha>&head=<sha>` returning
  `[{slug, name, env_triple, base_ns, head_ns, delta, status}]`. SQL is a
  self-join of the micro fact table on `commit_sha IN (base, head)` grouped by
  the dim tuple — directly analogous to how `api/summary.rs` self-joins
  `compression_sizes` for ratios (`schema.rs` doc, principle 6: "Ratios are not
  stored … computed at query time").
- New module `server/src/html/compare.rs` (peer of `chart.rs`) + a route in
  `app.rs::public_router` (`GET /compare`) and `api::compare` (`GET
  /api/compare`). Follow the `is_materialized_window` cache pattern in
  `api/mod.rs` and add a snapshot test under `server/tests/snapshots/`.
- **Deep-linkable by sha and PR**: the `?pr=` param is display-only (links back
  to the GitHub PR); `base`/`head` shas are the data keys, so the URL is stable
  and shareable — matching the site's "permalinks are the sharing mechanism"
  philosophy (`html/mod.rs` doc).

The PR comment's header link (`/compare?...`) and per-row links (`/micro/...`)
in section 2 point exactly at these two routes. Host is `bench.vortex.dev`
(the v3 site, per `vortex-bench/src/v3.rs` doc) — make it a single
`COMPARE_BASE` constant in the Python script + a workflow `vars.BENCH_SITE_URL`
so non-prod environments degrade to plain text.

---

## 6. Open questions / decisions for the user

1. **True merge-base vs latest-develop-success.** CodSpeed compares against the
   PR's merge-base. `bench-pr.yml` (macro) uses *latest successful develop run*,
   which drifts from merge-base on stale branches. Use real `git merge-base`
   (sketched in 3.4) — confirm we always have microbench data at the merge-base
   commit, or accept the "no base data" fallback. (Depends on bullet 3 running
   microbenches on **every** develop commit, like `codspeed.yml` does today.)
2. **Where base data lives** — website read endpoint (3.1-A) vs S3 grep
   (3.1-B). S3 is zero new server code and matches the macro path; the website
   API is cleaner but couples the PR comment to site uptime (note `bench.yml`
   ingest is "no longer best-effort", per `benchmarks-website/AGENTS.md`).
3. **Microbench storage shape (bullet 3 dependency).** Do microbenches get a
   sixth DuckDB fact family (`micro_measurements`) or reuse a generic timing
   table? The comment script only needs `{name, value_ns, all_runtimes_ns,
   env_triple}`, but the website compare view needs a `ChartKey` variant + slug
   prefix (`schema.rs`, `slug.rs`, `family.rs` — a coordinated wire change per
   `AGENTS.md`).
4. **Gating policy.** Confirm "advisory by default, opt-in hard gate via
   `perf-sensitive` label, never required initially." Promote to a required
   check only after variance is characterized.
5. **CUDA arch grouping.** The CUDA shards are walltime on a `g5` host. Group
   them under their own `env_triple` and apply a wider threshold (CUDA walltime
   is noisier) — decide the CUDA hard-regression band separately (e.g. 15–20%).
6. **Comment vs separate sub-comments.** Macro `bench-pr.yml` posts *one comment
   per benchmark id*. We propose **one merged** microbench comment (shards
   concatenated in `microbench-comment.yml`). Confirm a single comment is wanted
   over per-shard comments.
7. **Decommissioning `codspeed.yml`.** This slice can run **alongside** CodSpeed
   for a validation period (compare our verdicts vs CodSpeed's) before deleting
   `codspeed.yml` and the `CODSPEED_TOKEN` secret. Recommend a 2–4 week overlap.

---

## Files referenced

- `.github/workflows/codspeed.yml` — current microbench producer (to replace).
- `.github/workflows/bench-pr.yml` — macro sticky-comment pattern to mirror
  (token, `thollander/...@v3`, fork gate, base-from-develop).
- `.github/workflows/bench-dispatch.yml`, `sql-pr.yml` — label-dispatch +
  `workflow_call` patterns.
- `.github/workflows/claude-review.yml`, `approvals.yml` — fork-PR refusal /
  read-only-token patterns (basis for the `workflow_run` split).
- `scripts/compare-benchmark-jsons.py` — macro markdown generator to inherit
  shape/helpers from; `scripts/s3-download.py`, `scripts/bench-taskset.sh`.
- `benchmarks-website/server/src/{app.rs,api/mod.rs,schema.rs,slug.rs,html/chart.rs}`
  — site routes/schema to extend for `/compare` + `/micro`.
- `vortex-bench/src/v3.rs`, `scripts/post-ingest.py` — v3 wire shape + ingest.
- `Cargo.toml` (`codspeed-divan-compat`, `criterion`, `cfg(codspeed)`) — the
  off-CodSpeed harness this all relies on.
