<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Benchmarks website v3 - Planning

Planning docs for `bench.vortex.dev` v3: a single Rust binary
(axum + maud + duckdb-rs) replacing the v2 Node/React stack.

## Status

- **Alpha shipped** to `ct/benchmarks-v3`. Server, migrator, and
  inline-charts UI are merged.
- **In production-readiness phase.** v2 is still serving
  `bench.vortex.dev`. v3 has not been deployed publicly yet.
- **UI follow-ups** are owned by the user, not by agents (see
  "Deferred UI follow-ups" below).

A 10-bullet architecture summary lives at the top of
[`AGENTS.md`](./AGENTS.md). Use that for handoffs and external
sharing.

## Production readiness checklist

In rough order. Each item is a separate task; do not bundle.

### 1. Repo secrets

Two GitHub repository secrets on `vortex-data/vortex` (admins only):

- `INGEST_BEARER_TOKEN` — random 32+ byte token. Same value gets
  set as the `INGEST_BEARER_TOKEN` env var on whatever host runs
  the v3 server. Generate with `openssl rand -hex 32`.
- `V3_INGEST_URL` — full URL of the v3 ingest endpoint, e.g.
  `http://<host>:3000/api/ingest` for the test box, or
  `https://bench.vortex.dev/api/ingest` for prod.

Repo-level secrets are fine for the test phase. Move to an
Environment-scoped secret (gated to `ct/benchmarks-v3` /
protected branches) before prod.

### 2. CI ingestion wiring

Confirm whichever workflow runs the benchmark suites and pushes
results uses `secrets.INGEST_BEARER_TOKEN` and
`secrets.V3_INGEST_URL`, and POSTs the versioned envelope shape
defined in [`02-contracts.md`](./02-contracts.md). The current
workflow targets the v2 endpoint; needs to either dual-write or
flip.

### 3. Test deployment

Currently a manual EC2 box for smoke-testing. Latest test host:

- DNS: `ec2-18-116-241-0.us-east-2.compute.amazonaws.com`
- Port: `3000` (open to `0.0.0.0/0` in the security group)
- Bind: `VORTEX_BENCH_BIND=0.0.0.0:3000` (default `127.0.0.1` does
  not work for external access)
- HTTP only, no TLS. Public IP changes on stop/start unless an
  Elastic IP is associated. Throwaway token only — don't reuse for
  prod.

Smoke test from a laptop:

```bash
curl -i http://<host>:3000/
```

Should return HTTP 200 with the landing HTML.

### 4. Smoke test with migrated data

Run `vortex-bench-migrate` against the v2 source, copy the
resulting `bench.duckdb` to the deployed host, point
`VORTEX_BENCH_DB` at it, and walk every group's charts in a
browser.

### 5. Operational hygiene (not yet done)

- Health-check endpoint (`GET /health` returning 200).
- Structured logging review (we already use `tracing`; verify
  fields are useful for prod debugging).
- Rate limiting on `/api/ingest` — the bearer token is the only
  gate today.
- TLS termination strategy: front with a load balancer / nginx /
  Caddy, or terminate in-process? Decide before DNS flip.
- DB schema-version tracking, so future migrations are coordinated
  rather than ad-hoc.
- Backup story. Open question: is "copy the file" enough, or do we
  want a WAL-based / point-in-time approach? Investigate DuckDB
  options.

### 6. Deployment platform decision

v2 ran on EC2 (see top-level `ec2-init.txt`,
`docker-compose.yml`). v3 is a self-contained binary + DuckDB file
+ env var, so the v2 setup isn't reusable verbatim. Decide:

- Reuse the v2 EC2 host (cutover-style)?
- Stand up a new EC2 box?
- Containerize and run somewhere managed?

The simplest first cut is a new EC2 instance with a systemd unit
and an Elastic IP.

### 7. DNS flip

Point `bench.vortex.dev` at the v3 host. After this:

- v2 is no longer serving production traffic.
- The v2 cleanup PR (item 8) becomes safe to merge.
- Production secrets are now load-bearing — rotate
  `INGEST_BEARER_TOKEN` if the test value was ever shared.

### 8. v2 cleanup PR

A separate PR, opened post-flip. Deletes everything top-level under
`benchmarks-website/` that belongs to v2:

- `server.js`, `src/`, `index.html`, `vite.config.js`,
  `package.json`, `package-lock.json`, `public/`
- Top-level `Dockerfile`, `docker-compose.yml`, `ec2-init.txt`
- Any GitHub Actions workflows that only target the v2 deploy

The v3 tree under `benchmarks-website/server/` and
`benchmarks-website/migrate/` is untouched.

## Open product decisions

These are user/owner decisions, not agent decisions.

- **What migrated data do we keep vs drop?** The classifier
  currently silently drops every record routed to a `Skip` variant
  (e.g. `Skip::HistoricalMemory`, legacy random-access shapes).
  Some of those `Skip`s are real "we don't want this" cases; some
  are "we'd want this if we extended a fact table." Once v2 is
  gone the source records are gone with it, so this needs an
  explicit pass through `Skip` variants before flip.
- **Group naming.** Server emits names like `tpch sf=1 [nvme]`;
  v2's published names are `TPC-H (NVMe) (SF=1)`. Either rename the
  server-emitted names to v2 form, or add a sort-key + display-name
  map. Cosmetic but visible.
- **Deferred UI follow-ups.** The user is handling these directly;
  agents should not pre-empt them:
  - `collect_group_charts` N+1 refactor in `api.rs:583-613`.
  - Mobile legend resize handler.
  - Zoom-sync within a group.
  - LTTB downsampling for very long histories.
  - Swap the inline crosshair plugin for `chartjs-plugin-crosshair`.

## Reading order (alpha-era reference)

Still useful for context on why the schema and contracts look the
way they do. Not all of this is current.

| File | Read when |
|---|---|
| [`AGENTS.md`](./AGENTS.md) | Always. Status, architecture, code map, conventions, common mistakes. |
| [`00-overview.md`](./00-overview.md) | The original alpha pitch and dependency map. |
| [`01-schema.md`](./01-schema.md) | The five DuckDB fact tables + `commits` dim. Live contract. |
| [`02-contracts.md`](./02-contracts.md) | Wire shapes (one `kind` per fact table), HTTP error matrix, auth header. Live contract. |
| [`benchmark-mapping.md`](./benchmark-mapping.md) | Existing benchmarks → fact tables. Live reference, especially when extending the migrator. |
| [`decisions.md`](./decisions.md) | What was pinned for alpha. |
| [`deferred.md`](./deferred.md) | What was punted from alpha. Cross-reference with the "Deferred UI follow-ups" list above. |
| `components/<name>.md` | Original per-workstream plans. All three are merged; treat these as historical. |

## Components (merged)

Three workstreams shipped for alpha. All three are merged to
`ct/benchmarks-v3`. Plans kept for reference.

| Component | Plan | Status |
|---|---|---|
| Server | [components/server.md](./components/server.md) | Merged. |
| Emitter | [components/emitter.md](./components/emitter.md) | Merged. |
| Web UI | [components/web-ui.md](./components/web-ui.md) | Merged (plus per-chart UX rebuild). |

## Working branches

- `develop` — the v2 site, in production. **Do not touch** until
  after DNS flip.
- `ct/benchmarks-v3` — the integration branch. All v3 work lands
  here. Feature branches branch from it and PR back to it.
- `claude/benchmarks-v3-<topic>` — per-task feature branches.

PRs are reported by URL but **never auto-merged**. The user
reviews and merges.

## What this plan is not

- Not a parity-with-v2 plan. v3 ships the existing benchmark
  groups, not v2's exact UX.
- Not a phase-2 design doc. Phase 2 is the prod-readiness
  checklist above; further-out work lives in
  [`deferred.md`](./deferred.md) and the "Deferred UI follow-ups"
  list.

## Updating these docs

If you find a gap, prefer to:

1. Update [`02-contracts.md`](./02-contracts.md) when the gap is at
   a component boundary.
2. Update [`AGENTS.md`](./AGENTS.md) when the gap is a new agent
   norm or a new "thing to avoid."
3. Update this file when the gap is the prod-readiness punch list
   or an open product decision.
4. Update [`decisions.md`](./decisions.md) when the gap is "we
   just haven't decided yet, but we need to."
5. Update [`deferred.md`](./deferred.md) when the gap is "real
   work but not now."

Don't add a new top-level numbered doc.
