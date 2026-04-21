<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 09 - Open questions

These are decisions that need human input (or at least a follow-up agent
session with more time than this planning pass had) before implementation
starts. They are ordered roughly by "how much does the answer change
everything else" - top items are the biggest deal.

## Architecture

### Q1. Option A vs B for ingestion

[`07-ingestion.md`](./07-ingestion.md) leans toward Option B (per-run shards +
separate merger). Option A (each CI job writes DuckDB with CAS) is simpler in
infra but worse under concurrent writers. Confirm B before starting.

### Q2. Where does the migrator/ingester crate live?

Options:

- New crate `benchmarks-website/ingester/` (colocated with the website).
- New crate at workspace root like `vortex-bench-ingester`.
- Binaries under an existing crate (e.g. `vortex-bench/src/bin/ingest.rs`).

`vortex-bench` already knows the measurement types, so there's a strong case
for putting the migrator there. But the migrator also needs DuckDB and a
bunch of S3 glue that `vortex-bench` shouldn't depend on. Probably a separate
small crate is cleanest.

### Q3. Is Leptos the right framework *today*?

Leptos is the Rust-ecosystem answer. Check the state of Leptos at
implementation time:

- Is SSR-with-hydration stable enough?
- Are there breaking changes imminent that would bite us?
- Alternatives worth a quick look: `axum` + `askama` or `maud` templates
  (server-rendered HTML, no hydration framework, use vanilla JS for the
  interactive bits); `dioxus` with SSR (similar-ish to Leptos).

If Leptos is fine, use it. If it's in flux, fall back to axum + templates +
a minimal client JS bundle; the DB and schema decisions are framework-agnostic.

## Schema

### Q4. Synthetic PK vs composite UNIQUE

[`05-schema.md`](./05-schema.md) proposes a `measurement_id` that's a hash of
the dimensional tuple. The alternative is a composite UNIQUE without a
synthetic id.

- Hashed id: easy upsert (`INSERT ... ON CONFLICT (measurement_id) DO UPDATE`).
- Composite unique: more honest about what's unique, no magic hash.

Either works. Pick one. Preference: composite UNIQUE with no synthetic PK if
DuckDB supports it well; hashed id otherwise.

### Q5. `filter_sql` in `benchmark_groups`?

Storing SQL strings in a data column means anyone with write access to the DB
can inject arbitrary query logic. That's OK for an internal benchmarks tool
but we should be explicit about it. The alternative is keeping the group
definitions in a Rust const table and regenerating at build time.

### Q6. How do we handle schema evolution?

Expect the schema to evolve. We need to decide upfront:

- Do we store a `schema_version` in a meta table in the DB itself?
- Do we use DuckDB migrations (via `refinery` or similar)?
- Or do we just re-run the full migrator from `data.json.gz` every time the
  schema changes (since we keep the raw data forever)?

Re-run-from-source is actually viable and simple as long as JSONL is kept
around. Recommend that as the default.

### Q7. Unit normalization at ingest or at query

Current plan: persist `value_ns` / `value_bytes` / `value_unitless` as raw
integers, convert to display units in views/frontend. This is right, but we
should double-check there's no precision loss in the BIGINT ns representation
for the longest-running benchmarks (TPC-H SF=1000 queries can run for
minutes; minutes-in-ns fit in i64 with many orders of magnitude of headroom,
so it's fine).

## Ingestion / data

### Q8. What happens to `all_runtimes_ns`?

Shape B records carry `all_runtimes` arrays (individual run times, not just
the median). v2 discards them; v3 can keep them as a DuckDB `LIST<BIGINT>`.

Decision needed: do we expose them in the UI (variance plots? error bars)?
If not, do we still store them (for future use)? I recommend "yes, store them;
UI can light them up later".

### Q9. Do we preserve the raw JSON line somewhere?

`extra_json` in the schema gives room for unanticipated fields. A stronger
version is to store the **entire** raw JSON line per row. Cheap in DuckDB,
lets us re-derive anything without going back to the JSONL blob.

Lean toward storing it for the first 6 months post-cutover while we find
edge cases, then drop it.

### Q10. `file-sizes-*.json.gz` or a single blob?

CI currently writes per-benchmark-id `file-sizes-<id>.json.gz` files. In v3
these all go into `measurements` anyway, so the sharding doesn't matter to
the DB - but the ingester needs to know where to look. Settle on either a
single `pending/<run_id>/<job_id>/results.json` path (unify with Shape A/B/C
results) or keep size results in a separate key and make the merger read
both.

Simpler: unify. One path per CI job.

## Website UX

### Q11. Do we keep engine/storage/SF filters per-group or global?

v2 has a per-group engine filter. Should a global "only show me vortex:parquet
rows across every chart" toggle exist? If yes, it changes URL structure.
Start per-group (parity with v2), consider global later.

### Q12. How do we expose the ad-hoc SQL page?

Do we want to let anyone with the URL execute SQL against the DB? DuckDB's
`SELECT`-only mode + a hardcoded row limit makes this safe-ish. Decide
whether this is a v3 launch feature or a v3.1 feature.

## Ops

### Q13. Where does the DB live in the short term?

Do we write to a *new* S3 path (`bench-v3.duckdb` in a new bucket) while v2
keeps running, then swap? Or do we write to the same bucket alongside
`data.json.gz`?

Same bucket, new key. Keeps IAM simple. The existing benchmark-writer IAM
role just needs an extra object permission.

### Q14. Auth / access

The v2 site is fully public. Do we keep it that way? (Assumption: yes.) The
DB file on S3 needs to be public-readable (same as `data.json.gz`). Writer
role stays OIDC-only.

### Q15. Rollback strategy

If v3 is broken in prod, can we flip back to v2 in <10 minutes? The answer is
"yes, as long as v2's data pipeline (`cat-s3.sh` → `data.json.gz`) is still
running". Don't tear that pipeline out until v3 has been live without
incident for at least a week.

## Smaller things worth noting but not blocking

- `scripts/compare-benchmark-jsons.py` on `ct/vfvb` might have useful
  testing-time hooks; re-read before writing the migrator's verification
  step.
- The `LTTB` downsampling in v2 uses the `downsample` npm package; for v3 we
  either port LTTB to Rust (small amount of code) or compute it in SQL with
  a window function.
- `chartjs-plugin-zoom` + `hammerjs` for touch gestures: keep them, they work.
- Consider adding a "regressions in last N days" page that uses percentile
  windows over the `measurements` table. Nice but not launch-critical.
