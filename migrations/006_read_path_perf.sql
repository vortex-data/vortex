-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright the Vortex contributors

-- migrate-schema: requires-superuser
-- This migration ALTERs `query_measurements` and CREATEs indexes on tables owned
-- by the RDS master (`postgres`), both of which require the table owner (or a
-- superuser locally / the testcontainer suite). The marker makes
-- `migrate-schema.py` reject a non-master `apply` loudly and early, exactly like
-- 002/004/005. Apply it as the RDS master, the same operator path PR-5.0 used.

-- Read-path performance (PR-5.1.5). At the full prod seed (4.85M
-- `query_measurements`) two read paths were pathological even after the sargable
-- WHERE rewrite + the db.t4g.medium upsize:
--   (c) the per-group query summary's "latest value per (query_idx, engine,
--       format)" scanned the group's entire history and `row_number()`-windowed
--       it over a `commits` join (~6.2s warm for tpcds), because the sort key
--       (`commits.timestamp`) lived off-table; and
--   (d) `collectFilterUniverse`'s `SELECT DISTINCT engine/format` scanned the
--       whole fact tables on every page render (~4s warm).
-- Fix (c) by denormalizing the commit timestamp into `query_measurements` so the
-- latest-per-series lookup is a single index scan, and fix (d) with low-cardinality
-- indexes that back a loose-index (skip) scan. The other fact tables' summaries are
-- small-table scans (<300ms warm) and are left as-is.

-- (c) Denormalized commit timestamp. Nullable on purpose: adding it NOT NULL
-- would require every writer (`post-ingest.py --postgres`, the Rust migrate
-- loader) to populate it BEFORE this column exists, which is impossible to order.
-- The writers are updated to set it, the summary query orders `NULLS LAST`, and a
-- post-deploy re-backfill fills any rows a not-yet-deployed writer inserted, so a
-- transient NULL never silently wins "latest". `IF NOT EXISTS` keeps the migration
-- idempotent (the ledger already guards re-apply, but additive DDL stays safe).
ALTER TABLE query_measurements ADD COLUMN IF NOT EXISTS commit_timestamp timestamptz;

-- One-time backfill from `commits`. A no-op on a fresh (empty) schema, so the
-- testcontainer suite is unaffected; on prod it fills the 4.85M existing rows.
UPDATE query_measurements q
   SET commit_timestamp = c.timestamp
  FROM commits c
 WHERE c.commit_sha = q.commit_sha
   AND q.commit_timestamp IS NULL;

-- Supporting index for the latest-per-series summary: the group-filter columns,
-- then the series identity, then `commit_timestamp DESC` so
-- `DISTINCT ON (query_idx, engine, format) ORDER BY ..., commit_timestamp DESC`
-- resolves to an index scan that yields each series' latest row directly.
CREATE INDEX IF NOT EXISTS idx_query_measurements_summary
    ON query_measurements (dataset, dataset_variant, scale_factor, storage,
                           query_idx, engine, format, commit_timestamp DESC);

-- (d) Low-cardinality indexes backing the loose-index (skip) scan in
-- `collectFilterUniverse` (distinct engines + distinct formats), which runs on
-- every page render. `engine` exists only on `query_measurements`; `format` exists
-- on the four non-vector fact tables.
CREATE INDEX IF NOT EXISTS idx_query_measurements_engine
    ON query_measurements (engine);
CREATE INDEX IF NOT EXISTS idx_query_measurements_format
    ON query_measurements (format);
CREATE INDEX IF NOT EXISTS idx_compression_times_format
    ON compression_times (format);
CREATE INDEX IF NOT EXISTS idx_compression_sizes_format
    ON compression_sizes (format);
CREATE INDEX IF NOT EXISTS idx_random_access_times_format
    ON random_access_times (format);
