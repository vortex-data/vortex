<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# AGENTS.md - benchmarks-website v3 rewrite

Read this before starting any implementation work in this subtree.

## TL;DR

We are rebuilding the benchmarks website. Target: **Leptos SSR + DuckDB on S3**.
Read `planning/README.md` for the doc tour; read `planning/00-context.md` for
the why; read `planning/09-open-questions.md` to see what's still up in the
air before you write code.

**The v2 site at `benchmarks-website/` is still in production.** Do not delete
or rewrite it until v3 has a successful cutover. During v3 implementation,
**add new code alongside** (e.g. `benchmarks-website/server-v3/`,
`benchmarks-website/ingester/`). Do not modify the live `server.js` / React
app except to fix bugs.

## Essential context

### The data

- Benchmark results live in `s3://vortex-ci-benchmark-results/data.json.gz` as
  gzipped JSONL, appended on every merge to `develop`.
- Commit metadata lives in `s3://vortex-ci-benchmark-results/commits.json`.
- `vortex-bench/src/measurements.rs` is the emitter. There are ~4 distinct
  record shapes (see `planning/03-raw-data-schema.md`).
- The current v2 Node server at `benchmarks-website/server.js` does all the
  classification of raw records into groups/charts/series. This logic **must
  be ported into the v3 ingester**, bug-for-bug, before we can cut over.

### Prior art on `ct/vfvb`

A 2025 hackathon branch attempted to store benchmarks in Vortex files on S3
and render the site from WASM. It failed but left useful scaffolding. See
`planning/02-vfvb-salvage.md` for exactly what to copy forward and what to
ignore. Highlights:

- `vortex-wasm/src/website/{commit_id,commit_info,entry}.rs` - reusable structs.
- `vortex-wasm/src/bin/migrate_{data,commits}.rs` - migration binary templates.
- Everything else is hackathon-era and mostly not worth porting.

Verbatim snapshots of the useful files live in `planning/reference/`. These
are **non-compiling references** (`ct/vfvb` branched off a very old
`vortex-array` API). If you port code, **re-target it to current HEAD** -
the Array APIs changed substantially.

### Repository conventions

See the root `CLAUDE.md` / `AGENTS.md` for Rust style, test layout, and CI
norms. Callouts specific to this project:

- New crates: follow the existing workspace layout. Add to root `Cargo.toml`'s
  `[workspace] members = [...]`.
- All commits need `Signed-off-by:` trailers.
- Run `cargo +nightly fmt --all` + narrow clippy on what you changed. Public
  API changes need `./scripts/public-api.sh`.
- Add a doc comment to every new public item. Sample tests should return
  `VortexResult<()>` and use `?`, not `unwrap`.

## Working branches

| Branch | Purpose |
|--------|---------|
| `ct/vfvb` | Archived hackathon. Read-only reference. |
| `develop` | Current v2 site. **Do not break.** |
| `claude/review-vfvb-branch-lT2Pg` | This planning branch. |

Later agents will work on feature branches like:

- `claude/benchmarks-v3-ingester`
- `claude/benchmarks-v3-leptos`
- `claude/benchmarks-v3-cutover`

Each picks up from `develop`, not from the planning branch.

## Recommended implementation order

This is suggested, not prescriptive.

1. **Schema pilot**. Write a prototype migrator that reads a *copy* of
   `data.json.gz` + `commits.json` and emits a `bench.duckdb`. Port the v2
   classifier faithfully. Compare the resulting DB's distinct
   (group, chart, series, commit) tuples against v2's `/api/metadata`. Do not
   move forward until this diff is clean.

2. **Leptos scaffold**. Stand up a minimal Leptos SSR server that reads the
   prototype DuckDB and renders one chart. Prove the end-to-end loop works.

3. **Ingester for new runs**. Implement the per-shard + merger pattern
   (Option B in `planning/07-ingestion.md`). Wire it into CI behind a flag so
   it runs in parallel with the existing `cat-s3.sh` flow.

4. **Website feature parity**. Build out the page inventory from
   `planning/08-website.md`. Start with `/group/:slug` + `/chart/:slug`;
   landing and per-commit come later.

5. **Cutover**. Flip DNS, archive v2's pipeline after a quiet week.

## Things to avoid

- Don't touch `vortex-bench`'s emission format in this project. We are
  changing the *sink*, not the source. If the emitter produces garbage,
  fix that in a separate PR.
- Don't invent new dimension tables / new group categorizations beyond what
  v2 already has. **Parity first, improvements later.** It's much easier to
  argue about whether "Compression" should be split into "Write" and "Scan"
  *after* users can see both versions side by side.
- Don't delete `scripts/cat-s3.sh`, `commits-json.sh`, `data.json.gz`, or any
  v2 code until cutover is complete and verified. These are the rollback path.
- Don't add WASM to the critical path. If you find yourself reaching for
  `wasm-bindgen`, stop and reread `planning/00-context.md`.
- Don't over-engineer schema evolution. Our raw data is tiny; re-running the
  migrator is free. Start simple.

## Questions that keep coming up

> "Should we store ratios as rows?"

No. They're views. See `planning/05-schema.md`.

> "Should we precompute downsampled series in the DB?"

Probably not, but it's in open questions (Q1-related). DuckDB can do it fast.
If server-side memoization isn't enough, revisit.

> "Can we drop the v2 config.js mapping tables entirely?"

Yes - that's one of the point of v3. The presentation metadata moves into
`known_engines`, `known_formats`, `known_datasets`, `benchmark_groups` tables
in the DB. The frontend's only static config should be purely visual (e.g. a
fallback color palette).

> "What if Leptos isn't ready?"

See Q3 in `planning/09-open-questions.md`. Fallback is axum + templates
(askama/maud) with minimal client JS. The DB layer is framework-agnostic.

## How to update this file

If you learn something a future agent will need, add it here. Keep it short.
If you are tempted to write more than three paragraphs, that content probably
belongs in a numbered planning doc instead.
