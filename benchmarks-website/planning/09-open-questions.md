<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 09 - Open questions

These are decisions that need human input (or a follow-up agent session)
before implementation. They are ordered roughly by "how much does the answer
change everything else".

## Resolved (kept here as a log)

### ✓ Ingestion concurrency model

Rejected the earlier proposal of CAS against a DuckDB file on S3 and the
per-shard-plus-merger variant. Settled on: **the Leptos server owns the DB
on a local EBS volume; CI POSTs to an authenticated `/api/ingest` endpoint.**
See [`07-ingestion.md`](./07-ingestion.md) and [`04-architecture.md`](./04-architecture.md).

### ✓ Primary key on `measurements`

Settled on a **synthetic hash PK** (`measurement_id = xxhash64(dimensional
tuple)`). NULLs stay as NULLs in the columns; the ingester canonicalizes
NULL into the hash input. Rejected composite UNIQUE because of its
`NULL != NULL` gotcha. See [`05-schema.md`](./05-schema.md).

### ✓ `filter_sql` in group definitions

Dropped. Group definitions live in typed Rust code, not in data. Display
strings still live in DB lookup tables. See [`05-schema.md`](./05-schema.md).

### ✓ Extensibility for non-row-table benchmarks

Added `data_descriptor JSON` to `measurements`. Covers vector-search
(layout / threshold / dimensions / row counts), criterion-style
microbenchmarks (group / parameter), and any future benchmark with
parameters that aren't cross-cutting dimensions. See
[`05-schema.md`](./05-schema.md).

## Architecture

### Q1. Where does the ingester-classifier crate live?

The classifier is shared between the Leptos server's `/api/ingest` route and
the one-shot historical migrator. Candidate homes:

- New crate `benchmarks-website/classifier/` (colocated with the website).
- New crate under the workspace root, e.g. `vortex-bench-classifier`.
- A module inside the server crate, re-exported for the migrator.

The migrator can't live inside the server crate (it's a separate binary that
might not pull in axum/leptos). Lean toward: one `benchmarks-website-shared`
crate with the `RawMeasurement` types and the classifier function, depended
on by both `benchmarks-website-server` and `benchmarks-website-migrator`.

### Q2. Is Leptos the right framework *today*?

Check the state of Leptos at implementation time:

- Is SSR-with-hydration stable?
- Are breaking changes imminent?
- Alternatives: axum + `askama`/`maud` templates with vanilla JS for
  interactivity; dioxus SSR.

If Leptos is fine, use it. If it's in flux, fall back to axum + templates;
the DB and schema decisions are framework-agnostic.

## Authentication and ops

### Q3. OIDC vs. shared-secret for `/api/ingest` auth

Option 1: validate GitHub OIDC tokens on the server, check repo+ref claims.
No rotating shared secret.

Option 2: static bearer token in GitHub Actions secrets.

Lean toward Option 1 if the ergonomics are reasonable. See
[`07-ingestion.md`](./07-ingestion.md).

### Q4. EBS volume size + backup cadence

Guess: 20 GiB gp3, nightly snapshot → S3. Refine when we know actual DB size
under real workload. Not a blocker; defaults are fine for launch.

### Q5. Rollback strategy beyond "restore from snapshot"

If v3 is broken in prod, can we flip back to v2 in <10 minutes? Yes, as long
as v2's `cat-s3.sh` → `data.json.gz` pipeline is still running. Don't tear
that pipeline out until v3 has been live without incident for a week. See
[`06-migration.md`](./06-migration.md) cutover plan.

## Schema / data

### Q6. Do we preserve the raw JSON line per measurement?

`data_descriptor` gives us room for anticipated extra fields. A stronger
version is to also store the **entire** raw JSON line (cheap in DuckDB) so
we can re-derive anything without re-downloading `data.json.gz`.

Lean toward: yes, store it in a `raw_json JSON` column for the first six
months post-cutover, then drop it if unused.

### Q7. `all_runtimes_ns` in-row vs. sidecar table

Today Shape B records carry individual run times. `measurements` stores them
as a `LIST<BIGINT>`. For a chart that only shows the median, that's wasteful
but harmless. If we ever surface variance/error bars, we have them. Ok as-is
until proven otherwise.

### Q8. Run-level metadata

Do we need a separate `runs` table (one row per CI invocation) with
hardware class, start/end times, etc.? Not for launch - the hardware class
rarely varies and `env_triple` is already in the fact table. Revisit if
hardware churn becomes a story.

### Q9. Unit precision for very long runs

BIGINT ns gives >292 years of headroom. Fine.

## Website UX

### Q10. Per-group vs. global filters

v2 has per-group engine filters. A "only show me vortex across every chart"
global toggle is tempting but changes URL structure. Ship per-group first
(parity with v2); consider global later.

### Q11. Ad-hoc SQL page

Do we expose `SELECT`-only SQL against the DB from the UI? DuckDB makes it
safe-ish with a read-only handle + timeouts + row limits. Launch-blocker or
v3.1 feature? Lean: v3.1. Not worth expanding the attack surface before we
know it's useful.

### Q12. Commit-diff page shape

`/commit/:sha` shows one commit's state across every benchmark. What does
"diff vs. parent" look like in the UI? Table with colored deltas? Sparkline?
Defer; iterate once the page exists.

## Smaller things worth noting but not blocking

- `scripts/compare-benchmark-jsons.py` on `ct/vfvb` might have useful
  testing-time hooks; re-read before writing the migrator's verification
  step.
- LTTB downsampling in v2 uses the `downsample` npm package. For v3, either
  port LTTB to Rust or compute it in SQL with a window function. Pick
  whichever the implementer finds simpler.
- `chartjs-plugin-zoom` + `hammerjs` (touch gestures): keep them.
- "Regressions in last N days" page using percentile windows over
  `measurements` is nice but not launch-critical.
