<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Benchmarks website v3 - Planning

Planning docs for `bench.vortex.dev` v3: a single Rust binary (axum + maud + duckdb-rs) replacing
the v2 Node/React stack.

## Status

- **Alpha shipped** to `ct/benchmarks-v3`. Server, migrator, full-history UI (client-side LTTB,
  range scrollbar, global filter chips, click-to-PR tooltip), response compression, and the
  `LANDING_INLINE_N` cold-load trim are all merged.
- **In production-readiness phase.** v2 is still serving `bench.vortex.dev`. v3 runs on a
  throwaway EC2 host for smoke-testing; not deployed publicly yet.
- **UI follow-ups** are owned by the user, not by agents (see "Deferred UI follow-ups" below).

A 10-bullet architecture summary lives at the top of [`AGENTS.md`](./AGENTS.md). Use that for
handoffs and external sharing.

## Production readiness checklist

In rough order. Each item is a separate task; do not bundle.

### 1. Repo secrets — done

`INGEST_BEARER_TOKEN` and `V3_INGEST_URL` are set as repo-level secrets on `vortex-data/vortex`.
They're fine at this scope for the test phase. Move to an Environment-scoped secret (gated to
`ct/benchmarks-v3` / protected branches) before prod. Rotate `INGEST_BEARER_TOKEN` if the test
value was ever shared in a comment / Slack / PR review.

### 2. CI ingestion wiring — partial

The dual-write step is wired into `bench.yml` and `sql-benchmarks.yml` via commit `f7fd270`. Still
to do: an end-to-end run that triggers the workflow on a feature branch, POSTs to the EC2 box, and
confirms the envelope lands in DuckDB intact. Outbox-style retry on failed POSTs is a follow-up;
not built until we observe a failure.

### 3. Test deployment

Currently a manual EC2 box for smoke-testing. Latest test host:

- DNS: `ec2-18-219-54-101.us-east-2.compute.amazonaws.com` (changes on stop/start unless an Elastic
  IP is associated)
- Port: `3000` (open to `0.0.0.0/0` in the security group)
- Bind: `VORTEX_BENCH_BIND=0.0.0.0:3000` (default `127.0.0.1` does not work for external access)
- HTTP only, no TLS. Throwaway bearer token only — don't reuse for prod.

Build path: build narrow on the box itself (it's a `c6a.4xlarge` to avoid local-vs-EC2 arch
mismatches). The v2 migration source is fetched directly from the public S3 bucket; no AWS creds
needed.

Smoke test from a laptop:

```bash
curl -i http://<host>:3000/
```

Should return HTTP 200 with the landing HTML.

### 4. Smoke test with migrated data — in progress

Run `vortex-bench-migrate` against the v2 source, point `VORTEX_BENCH_DB` at the result, walk every
group's charts in a browser. Done so far: Random Access (caught and fixed a missing-chart
regression — see `1228e530`); LTTB downsampling, range scrollbar, filter chips, and click-to-PR
all behave on real data. Still to walk: every other group at least once.

### 5. Operational hygiene (not yet done)

- Structured logging review (we already use `tracing`; verify fields are useful for prod
  debugging).
- Rate limiting on `/api/ingest` — the bearer token is the only gate today.
- TLS termination strategy: front with a load balancer / nginx / Caddy, or terminate in-process?
  Decide before DNS flip.
- DB schema-version tracking, so future migrations are coordinated rather than ad-hoc. The server
  already exposes the constant via `/health`; what's missing is on-disk persistence and a check on
  boot.
- Backup story. Open question: is "copy the file" enough, or do we want a WAL-based /
  point-in-time approach? Investigate DuckDB options.

### 6. Deployment platform decision

v2 ran on EC2 (see top-level `ec2-init.txt`, `docker-compose.yml`). v3 is a self-contained binary +
DuckDB file + env var, so the v2 setup isn't reusable verbatim. Decide:

- Reuse the v2 EC2 host (cutover-style)?
- Stand up a new EC2 box?
- Containerize and run somewhere managed?

The simplest first cut is a new EC2 instance with a systemd unit and an Elastic IP.

### 7. DNS flip

Point `bench.vortex.dev` at the v3 host. After this:

- v2 is no longer serving production traffic.
- The v2 cleanup PR (item 8) becomes safe to merge.
- Production secrets are now load-bearing — rotate `INGEST_BEARER_TOKEN` if the test value was
  ever shared.

### 8. v2 cleanup PR

A separate PR, opened post-flip. Deletes everything top-level under `benchmarks-website/` that
belongs to v2:

- `server.js`, `src/`, `index.html`, `vite.config.js`, `package.json`, `package-lock.json`,
  `public/`
- Top-level `Dockerfile`, `docker-compose.yml`, `ec2-init.txt`
- Any GitHub Actions workflows that only target the v2 deploy

The v3 tree under `benchmarks-website/server/` and `benchmarks-website/migrate/` is untouched.

## Open product decisions

These are user/owner decisions, not agent decisions.

- **What migrated data do we keep vs drop?** The classifier currently silently drops every record
  routed to a `Skip` variant (e.g. `Skip::HistoricalMemory`, legacy random-access shapes). Some of
  those `Skip`s are real "we don't want this" cases; some are "we'd want this if we extended a
  fact table." Once v2 is gone the source records are gone with it, so this needs an explicit
  pass through `Skip` variants before flip.
- **Group naming.** Server emits names like `tpch sf=1 [nvme]`; v2's published names are
  `TPC-H (NVMe) (SF=1)`. Either rename the server-emitted names to v2 form, or add a sort-key +
  display-name map. Cosmetic but visible.
- **Deferred UI follow-ups.** The user is handling these directly; agents should not pre-empt
  them:
  - `collect_group_charts` N+1 refactor in `api/charts.rs::collect_group_charts`.
  - Mobile legend resize handler. The position is picked once at chart construction via
    `matchMedia("(max-width: 768px)")`; it doesn't update if the viewport crosses the breakpoint.
  - Zoom-sync within a group.
  - Swap the inline crosshair plugin for `chartjs-plugin-crosshair`.

## Reading order (alpha-era reference)

Still useful for context on why the schema and contracts look the way they do. Not all of this is
current.

| File                                             | Read when                                                                                  |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| [`AGENTS.md`](./AGENTS.md)                       | Always. Status, architecture, code map, conventions, common mistakes.                      |
| [`00-overview.md`](./00-overview.md)             | The original alpha pitch and dependency map.                                               |
| [`01-schema.md`](./01-schema.md)                 | The five DuckDB fact tables + `commits` dim. Live contract.                                |
| [`02-contracts.md`](./02-contracts.md)           | Wire shapes (one `kind` per fact table), HTTP error matrix, auth header. Live contract.    |
| [`benchmark-mapping.md`](./benchmark-mapping.md) | Existing benchmarks → fact tables. Live reference, especially when extending the migrator. |
| [`decisions.md`](./decisions.md)                 | What was pinned for alpha.                                                                 |
| [`deferred.md`](./deferred.md)                   | What was punted from alpha. Cross-reference with the "Deferred UI follow-ups" list above.  |
| `components/<name>.md`                           | Original per-workstream plans. All three are merged; treat these as historical.            |

## Components (merged)

Three workstreams shipped for alpha. All three are merged to `ct/benchmarks-v3`. Plans kept for
reference.

| Component | Plan                                             | Status                              |
| --------- | ------------------------------------------------ | ----------------------------------- |
| Server    | [components/server.md](./components/server.md)   | Merged.                             |
| Emitter   | [components/emitter.md](./components/emitter.md) | Merged.                             |
| Web UI    | [components/web-ui.md](./components/web-ui.md)   | Merged (plus per-chart UX rebuild). |

## Working branches

- `develop` — the v2 site, in production. **Do not touch** until after DNS flip.
- `ct/benchmarks-v3` — the integration branch. All v3 work lands here. Feature branches branch
  from it and PR back to it.
- `claude/benchmarks-v3-<topic>` — per-task feature branches.

PRs are reported by URL but **never auto-merged**. The user reviews and merges.

## What this plan is not

- Not a parity-with-v2 plan. v3 ships the existing benchmark groups, not v2's exact UX.
- Not a phase-2 design doc. Phase 2 is the prod-readiness checklist above; further-out work lives
  in [`deferred.md`](./deferred.md) and the "Deferred UI follow-ups" list.

## Updating these docs

If you find a gap, prefer to:

1. Update [`02-contracts.md`](./02-contracts.md) when the gap is at a component boundary.
2. Update [`AGENTS.md`](./AGENTS.md) when the gap is a new agent norm or a new "thing to avoid."
3. Update this file when the gap is the prod-readiness punch list or an open product decision.
4. Update [`decisions.md`](./decisions.md) when the gap is "we just haven't decided yet, but we
   need to."
5. Update [`deferred.md`](./deferred.md) when the gap is "real work but not now."

Don't add a new top-level numbered doc.
