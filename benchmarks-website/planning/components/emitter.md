<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Component: Emitter (alpha)

## Required reading

- [`../00-overview.md`](../00-overview.md)
- [`../02-contracts.md`](../02-contracts.md)
- [`../benchmark-mapping.md`](../benchmark-mapping.md) - the
  source-type → target-table mapping.

## Goal

Extend `vortex-bench` so it emits v3-shape JSON. Plus a small POST
script that wraps the JSONL in an envelope and sends it to a
running alpha server.

This is **purely additive** to v2's emission path. Nothing in v2 is
touched. CI workflow integration, dual-write, the orchestrator
update, and the outbox safety net all wait until after the alpha
loop works end-to-end (see [`../deferred.md`](../deferred.md)).

## In scope

### Rust emitter

- Add a `--gh-json-v3 <path>` CLI flag that writes JSONL of bare
  v3 records (no envelope). The legacy `-d gh-json -o ...` form is
  untouched - both work at alpha.
- Emit a record with the appropriate `kind` for every measurement
  type produced today. The mapping from existing measurement
  structs to wire `kind`s is the table in
  [`../benchmark-mapping.md`](../benchmark-mapping.md).
- Two non-obvious points (everything else is mechanical):
  - `QueryMeasurement` and the paired `MemoryMeasurement` collapse
    into **one** `query_measurement` record with both `value_ns`
    and the four memory fields. If memory wasn't tracked, omit the
    memory fields.
  - Vector-search's `ScanTiming` doesn't carry its own dataset /
    layout / threshold (those live in the binary's `Args`). The
    emitter has to plumb them through to the record.
- `CustomUnitMeasurement` cross-format ratios are **not emitted** -
  ratios are computed in the read path.
- Snapshot tests per `kind` (any framework), scrubbing `commit_sha`
  and `env_triple`.

### Post-ingest script

A small Python script (path of the agent's choosing, e.g. under
`scripts/`) that:

- Reads JSONL of records.
- Fills the `commit` envelope fields by shelling out to `git show`
  (or equivalent) for the SHA passed as an argument.
- Wraps the records in the envelope from
  [`../02-contracts.md`](../02-contracts.md).
- POSTs to `<server>/api/ingest` with the bearer token.
- Exits non-zero on 4xx / 5xx. **No retries, no spool, no S3
  outbox at alpha** - those land when CI starts using this.

## Out of scope (deferred)

- Replacing the v2 `-d`/`-o` CLI form. Both forms coexist at alpha.
- Removing the v2 `gh-json` emission path.
- Updating `bench-orchestrator` or any GitHub Actions workflows.
  Alpha runs are manual.
- Retry / spool / outbox-drain on POST failures.

See [`../deferred.md`](../deferred.md) for the post-alpha plan.

## Acceptance criteria

- `cargo test -p vortex-bench` passes; one snapshot per `kind`.
- Running a benchmark with `--gh-json-v3 <path>` writes valid JSONL
  matching the wire shape from
  [`../02-contracts.md`](../02-contracts.md).
- The post-ingest script round-trips a fixture file through a
  running alpha server (200 with non-zero `inserted` on first run,
  200 with non-zero `updated` on second run).

## Branch

`claude/benchmarks-v3-emitter`
