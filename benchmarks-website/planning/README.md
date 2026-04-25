<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Benchmarks website v3 - Planning

Planning docs for rebuilding `bench.vortex.dev` as a single Rust
binary with DuckDB on local disk.

This plan is **alpha-only**. Everything beyond the smallest
end-to-end loop is deliberately punted to
[`deferred.md`](./deferred.md).

## Reading order

| File | Read when |
|---|---|
| [`00-overview.md`](./00-overview.md) | Always. The pitch, phases, and dependency map. |
| [`01-schema.md`](./01-schema.md) | Always. The five DuckDB fact tables + `commits` dim. |
| [`02-contracts.md`](./02-contracts.md) | Always. Wire shapes (one `kind` per fact table), HTTP error matrix, auth header. |
| [`benchmark-mapping.md`](./benchmark-mapping.md) | Always when working on the emitter or the historical migrator. Maps every existing benchmark to its target table. |
| [`decisions.md`](./decisions.md) | Skim once. What's pinned for alpha. |
| [`deferred.md`](./deferred.md) | Skim once. What we're not designing yet. |
| `components/<your-component>.md` | The plan for your specific workstream. |
| `components/<other>.md` | Avoid. If you're tempted, `02-contracts.md` probably needs an update. |

## Components

Three components for alpha. Each is one workstream, one branch, one
PR. After the schema and contracts are stable, **all three can be
worked on in parallel**.

| Component | Plan | Branch |
|---|---|---|
| Server | [components/server.md](./components/server.md) | `claude/benchmarks-v3-server` |
| Emitter | [components/emitter.md](./components/emitter.md) | `claude/benchmarks-v3-emitter` |
| Web UI | [components/web-ui.md](./components/web-ui.md) | `claude/benchmarks-v3-web-ui` |

## Working branches

- `develop` - the v2 site, in production. **Do not touch.**
- `claude/review-benchmarks-redesign-BO3la` - this planning branch.
- Component branches above - one per workstream, branched from
  `develop`.

## What this plan is not

- Not implementation instructions. Component plans are deliberately
  high-level.
- Not a phase-2 plan. Phase-2 work is one paragraph each in
  [`deferred.md`](./deferred.md). The path will be clearer once the
  alpha loop is running.
- Not a parity-with-v2 plan. v2 keeps running unchanged through
  alpha.

## Updating these docs

If you find a gap, prefer to:

1. Update [`02-contracts.md`](./02-contracts.md) when the gap is at
   a component boundary.
2. Update the relevant component plan when the gap is local.
3. Update [`decisions.md`](./decisions.md) when the gap is "we just
   haven't decided yet, but we need to."
4. Update [`deferred.md`](./deferred.md) when the gap is "this is
   real work but not for alpha."

Don't add a new top-level numbered doc.
