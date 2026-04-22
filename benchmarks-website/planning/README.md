<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Benchmarks Website v3 - Planning

This directory contains the planning, architecture, and design docs for the next
iteration of the Vortex benchmarks website. The goal is to replace the current
Node + React site (which streams a compressed JSONL blob from S3 and parses it on
every refresh) with a **axum-based website backed by a DuckDB database** of
historical benchmark results.

These docs are **planning artifacts**, not implementation instructions. Numbers,
SQL, and Rust sketches are indicative; final shapes will be settled in follow-up
PRs that actually build the thing.

## How to read this directory

Read in order:

| # | File | Purpose |
|---|------|---------|
| 00 | [`00-context.md`](./00-context.md) | Why we're doing this. Prior art (`ct/vfvb`, v2). What failed and what we learned. |
| 01 | [`01-current-state.md`](./01-current-state.md) | Audit of the current (v2) `benchmarks-website/` + its data pipeline. |
| 02 | [`02-vfvb-salvage.md`](./02-vfvb-salvage.md) | What we can and should reuse from the `ct/vfvb` hackathon PR, and what we should drop. |
| 03 | [`03-raw-data-schema.md`](./03-raw-data-schema.md) | The messy, de-facto schema emitted by `vortex-bench` today. |
| 04 | [`04-architecture.md`](./04-architecture.md) | Target architecture: axum + DuckDB. Boxes-and-arrows, deploy story. |
| 05 | [`05-schema.md`](./05-schema.md) | Proposed DuckDB schema (tables, columns, indexes, views). |
| 06 | [`06-migration.md`](./06-migration.md) | One-time historical data migration plan (JSONL → DuckDB). |
| 07 | [`07-ingestion.md`](./07-ingestion.md) | Ongoing ingestion: how new runs get into the DB. Replaces `cat-s3.sh`. |
| 08 | [`08-website.md`](./08-website.md) | UX principles and page inventory. |
| 09 | [`09-open-questions.md`](./09-open-questions.md) | Unresolved decisions + a log of resolved ones. |
| 10 | [`10-emitter-changes.md`](./10-emitter-changes.md) | The `vortex-bench` extension that lets us emit v3-shape JSON directly, deleting the need for a classifier in the server. |
| 11 | [`11-implementation-kickoff.md`](./11-implementation-kickoff.md) | **Binding contracts** for a fresh implementer: concrete Rust types, pinned hash algorithm, DuckDB crate choice, HTTP error matrix, seed-SQL bootstrap. Read after 00-10; this is what closes the "TBD by implementer" gaps. |

## Memory files

- [`AGENTS.md`](./AGENTS.md) - Brief for future coding agents working on this
  project. Read this before doing implementation work.

## Reference snapshots

- [`reference/`](./reference/) - Verbatim copies of the directly useful files
  from the archived `ct/vfvb` branch (CommitId, CommitInfo, the two migration
  binaries, the ETag CAS helper, and the original hackathon plan). These are
  **non-compiling references** - see [`reference/README.md`](./reference/README.md)
  for what each is and what to do with it.

## Working branches

- `ct/vfvb` - the hackathon PR from 2025 that tried WASM+Vortex-on-S3. Archived
  reference; do not re-use its runtime, but the schema + migration binaries are
  useful scaffolding.
- `develop` - the current v2 (Node SSR + React) site under `benchmarks-website/`.
  This stays live until v3 is feature-complete.
- `claude/review-vfvb-branch-lT2Pg` - this planning branch.

## What this plan is NOT

- Not a line-by-line implementation plan. That comes later.
- Not a commitment to every design choice here. Open questions are tracked in
  `09-open-questions.md`.
- Not a rewrite of everything. The benchmark *runners* (`vortex-bench`,
  `random-access-bench`, `compress-bench`, etc.) don't change in this project.
  We change how their output is *stored, queried, and rendered*.
