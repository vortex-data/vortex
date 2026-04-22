<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# AGENTS.md - benchmarks-website v3 rewrite

Read this before starting any implementation work in this subtree.

## TL;DR

We are rebuilding the benchmarks website. Target: **axum + compile-time HTML
templates (maud / askama) + DuckDB on a local EBS volume**, with CI POSTing
new benchmark results to an authenticated `/api/ingest` endpoint on the
server.

**Start here**: `planning/README.md` for the doc tour → `planning/00-context.md`
for the why → `planning/11-implementation-kickoff.md` for the binding Rust
contracts, hash algorithm, error matrix, and directory layout. Doc 11 is
where the numbers-and-types answers live; docs 00-10 are the context for
*why* those answers are what they are. `planning/09-open-questions.md`
tracks the two remaining open questions (both post-launch).

**The v2 site at `benchmarks-website/` is still in production.** Do not delete
or rewrite it until v3 has a successful cutover. During v3 implementation,
**add new code alongside** (e.g. `benchmarks-website/server-v3/`,
`benchmarks-website/migrator/`). Do not modify the live `server.js` / React
app except to fix bugs.

## Essential context

### The data

- Benchmark results live in `s3://vortex-ci-benchmark-results/data.json.gz` as
  gzipped JSONL, appended on every merge to `develop`.
- Commit metadata lives in `s3://vortex-ci-benchmark-results/commits.json`.
- `vortex-bench/src/measurements.rs` is the emitter. There are ~4 distinct
  record shapes (see `planning/03-raw-data-schema.md`).
- The current v2 Node server at `benchmarks-website/server.js` parses those
  records into groups/charts/series via a hand-written regex stack
  (`getGroup`, `formatQuery`, `normalizeChartName`). **v3 deletes this
  problem** by extending `vortex-bench` to emit v3-shape JSON directly -
  see `planning/10-emitter-changes.md`. The v2 regex logic lives exactly
  once more, inside the one-shot historical migrator, which is deleted
  post-cutover.

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

- `claude/benchmarks-v3-emitters`   (vortex-bench `-d gh-json-v3` + vector-search-bench CI wiring)
- `claude/benchmarks-v3-migrator`   (one-shot historical migrator + classifier)
- `claude/benchmarks-v3-server`     (axum server, routes, templates)
- `claude/benchmarks-v3-cutover`    (CI dual-write wiring + DNS flip + v2 retire)

Each picks up from `develop`, not from the planning branch.

## Recommended implementation order

This is suggested, not prescriptive. The emitter work comes first because
everything downstream depends on the shape of its output.

1. **Extend `vortex-bench` emitters.** Add the `-d gh-json-v3` output
   format to every measurement type (see `planning/10-emitter-changes.md`).
   Existing `-d gh-json` path stays intact. Include vector-search-bench
   in this pass - it doesn't emit `gh-json` today and should gain v3
   emission + a CI workflow.

2. **Write the migrator.** Standalone binary under
   `benchmarks-website/migrator/` (or wherever - kept off `main` /
   `develop`). Carries its own one-shot v2→v3 classifier (port v2's
   `server.js::getGroup` bug-for-bug). Reads `data.json.gz` +
   `commits.json` + `file-sizes-*.json.gz`, emits a populated
   `bench.duckdb`. Verify against v2's `/api/metadata` before moving on.

3. **Server scaffold.** Stand up a minimal axum server that opens the
   migrator's DuckDB read-write and renders one chart with a template.
   Prove the end-to-end loop works.

4. **Ingest endpoint.** Add `POST /api/ingest` to the server. **No
   classifier** - the handler serde-parses v3-shape records and upserts.
   Wire bearer-token auth.

5. **CI integration (dual-write).** Add the new `-d gh-json-v3` + POST
   step to `.github/workflows/bench.yml` and `sql-benchmarks.yml`,
   alongside the existing `-d gh-json` + `cat-s3.sh` calls. Add the
   `drain-ingest-outbox.yml` cron workflow for the spool-to-S3 safety
   net.

6. **Website feature parity.** Build out the page inventory from
   `planning/08-website.md`. Start with `/group/:slug` + `/chart/:slug`;
   landing and per-commit come later.

7. **Cutover.** Flip DNS, then in a follow-up PR: remove the `-d gh-json`
   emission path, remove `cat-s3.sh` / `commit-json.sh`, **delete the
   migrator binary and its classifier**. The main repo is now classifier-
   free.

## Things to avoid

- **Don't write a string-parsing classifier in the server.** If you find
  yourself porting v2's `getGroup` regex stack into the axum app, stop.
  Extend `vortex-bench`'s emitter instead and emit the v3-shape directly.
  See `planning/10-emitter-changes.md`. The only place a classifier is
  allowed to exist is inside the one-shot migrator binary, and that
  binary is deleted post-cutover.
- Don't invent new dimension tables / new group categorizations beyond
  what v2 already has. **Parity first, improvements later.** It's much
  easier to argue about whether "Compression" should be split into
  "Write" and "Scan" *after* users can see both versions side by side.
- Don't remove v2's `-d gh-json` output path from `vortex-bench` until
  cutover is complete. Same for `scripts/cat-s3.sh` / `commits-json.sh` -
  they are the dual-write rollback path.
- Don't add WASM to the critical path. If you find yourself reaching for
  `wasm-bindgen`, stop and reread `planning/00-context.md`.
- Don't over-engineer schema evolution. Our raw data is tiny; the
  migrator can be re-run during development for free. Start simple.

## Questions that keep coming up

> "Should we store ratios as rows?"

No. They're views. See `planning/05-schema.md`.

> "Should we precompute downsampled series in the DB?"

Probably not. DuckDB can compute LTTB-style slices on demand quickly enough
at our data size. Revisit if charts ever render slowly. See
`planning/09-open-questions.md` Q2.

> "Can we drop the v2 config.js mapping tables entirely?"

Yes - that's one of the points of v3. Pure-data bits (engine/format/dataset
display names and colors) move into `known_engines` / `known_formats` /
`known_datasets` tables in the DB. Group-definition logic (which rows make
up "TPC-H (NVMe) (SF=10)"?) moves into typed Rust code, not SQL strings -
see `planning/05-schema.md` for why `filter_sql` was rejected. The frontend's
only static config should be purely visual (e.g. a fallback color palette).

> "Where does the classifier live?"

**In the one-shot migrator only.** Nowhere else. The server doesn't have
one; neither does any part of the main repo. `vortex-bench` emits v3-shape
records directly (see `planning/10-emitter-changes.md`), so the server's
`/api/ingest` is a serde-validated passthrough.

## How to update this file

If you learn something a future agent will need, add it here. Keep it short.
If you are tempted to write more than three paragraphs, that content probably
belongs in a numbered planning doc instead.
