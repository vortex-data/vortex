<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Deferred work (phase 2+)

Things we know we need eventually, but **not in alpha**. Each item
gets a one-paragraph stub here so we don't lose the thinking. None
of these is being designed in detail right now: the path forward is
clearer once the alpha loop (server + emitter + web-ui) is running
end-to-end.

The order below is roughly the expected pickup order, but nothing
is binding.

## Historical data migration

A one-shot binary that reads `s3://vortex-ci-benchmark-results/data.json.gz`
+ `commits.json` + `file-sizes-*.json.gz` and writes a fully
populated v3 DuckDB. Carries a bug-for-bug port of v2's
`server.js::getGroup` classifier (the only place a classifier
exists in the codebase). Verifies against `bench.vortex.dev`'s
`/api/metadata` before dual-write opens; the binary and its
classifier are deleted post-cutover. The v2 classifier and lookup
tables to port from live on `develop` at
`benchmarks-website/server.js` and `benchmarks-website/src/config.js`.

## Production deploy

Dockerfile + docker-compose + EC2 init + EBS mount + nightly DuckDB
`.backup` to S3 + watchtower polling ghcr.io. Single-EC2 deploy,
matching v2's footprint. RPO is bounded by snapshot cadence; if
that's too loose, streaming WAL backup is a follow-up.

## CI workflow integration

Update `bench-orchestrator/runner/executor.py` to pass the new
`--gh-json-v3` flag. Add a dual-write step in `bench.yml` and
`sql-benchmarks.yml` that POSTs to the v3 server alongside the
existing `cat-s3.sh` append. Add a `commit-metadata` step that POSTs
`records: []` for every push to `develop`. Old `--gh-json` emission
stays alive through cutover.

## Outbox safety net

When CI POSTs start landing in real volume, failed POSTs need
somewhere to go. Plan: `post-ingest.py` falls back to dumping the
payload to `s3://vortex-ci-benchmark-results/outbox/<run_id>/...`,
and a `drain-ingest-outbox.yml` cron re-POSTs every 10 minutes,
deleting on success. Not built until we observe a failure that needs
it.

## Schema migration framework

A `schema_meta` table + forward-only `migrations/NNN_*.sql` files
applied in lex order on boot. Tested by replaying against a recent
prod backup before merge. Not needed at alpha (the DB is rebuilt on
schema change); becomes essential once real data lives in the DB.

## Lookup tables and seed SQL

`known_engines` / `known_formats` / `known_datasets`: display names
and color hex per row, populated from a seed SQL file applied on
every boot. Replaces v2's `ENGINE_RENAMES` / `SERIES_COLOR_MAP`
constants in `config.js`. Until this lands, the web-ui falls back to
raw `engine:format` strings and a small palette.

## Derived views

`v_compression_ratios`, `v_latest_per_group`, etc. Replaces v2's
stored `vortex:parquet ratio compress time/...` rows with on-the-fly
SQL. Until the views land, handlers compute the same thing inline.

## Multi-page UI

The full v2 page inventory: per-group landing with engine/category
filters, full-screen modal, zoom/pan, deep links, per-commit
snapshot page, summary cards (geomean ratios, random-access
rankings), ad-hoc SQL page, mobile-friendly redesign.

## Admin tooling

A `benchmarks-admin` CLI that talks to the running server over a
unix-domain socket, file-permission-gated to the bench user. SSH
access to the host = admin access. First commands: `health`,
`reload-seed`, `backup-now`. Unix socket listener in the server is
the integration point. No HTTP `/admin/*` surface.

## Auth upgrades

The shared bearer token in a GH Actions secret + EC2 env var is
fine for alpha and likely for launch. Upgrade paths (post-launch):
AWS ALB + Cognito + GitHub OIDC, Cloudflare Tunnel + Access, or
server-side GitHub JWKS validation. Pick when an admin feature or
a multi-tenant scenario actually needs it.

## Cutover + cleanup

DNS flip from v2 to v3. Subsequent cleanup PR removes:
- The v2 React app + Node `server.js`.
- The legacy `vortex-bench --gh-json` emission path.
- `scripts/cat-s3.sh` and `scripts/commit-json.sh`.
- The migrator crate and its bug-for-bug classifier.

## CI PR comparison post-cutover

`sql-benchmarks.yml` PR mode currently downloads `data.json.gz`
from S3 to find a baseline. Post-cutover that file stops growing.
Plan: point the comparator at the v3 server's
`/api/chart/:slug?last=N`. Cleanup follows the cutover PR.

## Downsampling

LTTB-style downsampling at 1x/2x/4x/8x, like v2 does today. Not
built until charts visibly suffer at full resolution. Implementation
is a SQL window function at query time, memoized per
`(slug, range, level)`. Doesn't change the schema either way.
