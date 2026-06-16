-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright the Vortex contributors

-- migrate-schema: requires-superuser
-- DROP/CREATE INDEX on `query_measurements` (owned by the RDS master), so this
-- carries the same requires-superuser marker as 005/006 and is applied by the
-- master, the operator path PR-5.0 used.

-- Read-path performance follow-up to 006 (PR-5.1.5 fix c). Measuring the 006
-- `idx_query_measurements_summary` against the prod seed showed the planner would
-- not use it for the per-group "latest value per series" summary: `value_ns`
-- (both the `value_ns > 0` filter and the projected value) was off the index, so
-- using it meant a heap fetch for every one of a group's ~870K rows -- more
-- expensive than the bitmap+sort the planner fell back to (~8.6s warm for tpcds).
-- Adding `value_ns` as an INCLUDE (non-key) payload makes the index COVER the
-- summary, so a `DISTINCT ON (query_idx, engine, format) ... ORDER BY
-- commit_timestamp DESC` resolves to an Index Only Scan (Heap Fetches: 0) at
-- ~1.95s warm. (A loose-index/recursive-CTE skip scan would reach ms but at a
-- large SQL-complexity cost; ~2s under the bounded-concurrency summary fan-out is
-- the pragmatic target for a CDN-fronted dashboard.)
DROP INDEX IF EXISTS idx_query_measurements_summary;
CREATE INDEX IF NOT EXISTS idx_query_measurements_summary
    ON query_measurements (dataset, dataset_variant, scale_factor, storage,
                           query_idx, engine, format, commit_timestamp DESC)
    INCLUDE (value_ns);

-- Cleanup: a one-off `idx_qm_summary_test` was created by hand on prod to measure
-- the covering-index design before formalizing it here. It does not exist in
-- testcontainers (this DROP is a no-op there) or any other environment; dropping
-- it keeps prod in sync with the migration set.
DROP INDEX IF EXISTS idx_qm_summary_test;
