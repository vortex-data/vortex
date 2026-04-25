<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# AGENTS.md - benchmarks-website v3 (alpha)

Brief for coding agents working on this rewrite. Keep it short;
detail belongs in component plans.

## What you're working on

The **alpha** of v3 of `bench.vortex.dev`. Target: a single Rust
binary with **DuckDB on local disk**. The smallest end-to-end loop
that proves the design.

The v2 site at `benchmarks-website/` is in production and stays
running unchanged. v3 lives alongside in a new crate under
`benchmarks-website/` (path is the server agent's call).

Anything not listed in [`README.md`](./README.md) under
"Components" is **deferred**. See [`deferred.md`](./deferred.md).
Don't expand scope past your component plan.

## Where to start

1. [`README.md`](./README.md) - reading order.
2. [`00-overview.md`](./00-overview.md) - phases, components,
   dependency map.
3. [`01-schema.md`](./01-schema.md) - the DuckDB schema (column
   contracts; SQL is the server agent's call).
4. [`02-contracts.md`](./02-contracts.md) - wire shapes + HTTP
   matrix + auth header.
5. [`benchmark-mapping.md`](./benchmark-mapping.md) - existing
   benchmarks → fact tables (read this if you're working on the
   emitter or eventual migration).
6. Your component plan in [`components/`](./components/).

You **don't** need to read other components' plans.

## Repository conventions

See the root [`CLAUDE.md`](/CLAUDE.md) for Rust style, test layout,
and CI norms. Project-specific:

- New crates go under `benchmarks-website/`. Add to root
  `Cargo.toml` workspace members.
- All commits need a `Signed-off-by:` trailer.
- Run `cargo +nightly fmt --all` and narrow clippy on what you
  changed.
- Public-API changes need `./scripts/public-api.sh`.
- Every new public item needs a doc comment.
- Tests return `VortexResult<()>` and use `?`. No `unwrap`.

## Things to avoid

- **Don't widen scope past your component plan.** If a feature
  feels missing, check [`deferred.md`](./deferred.md) first - it
  is almost certainly already deferred there.
- **Don't write a server-side classifier.** The emitter is
  responsible for v3-shape records.
- **Don't drift from contracts.** Wire-shape changes are a
  coordinated PR across the affected components.
- **Don't touch the v2 React/Node app.** It stays in production
  unchanged through alpha and through phase 2 until cutover.
- **Don't reach for WASM.**

## Working branches

| Branch | Purpose |
|---|---|
| `develop` | Live v2 site. Don't break. |
| `claude/review-benchmarks-redesign-BO3la` | This planning branch. |
| `claude/benchmarks-v3-<component>` | Per-workstream feature branches. |

Component branches start from `develop`.

## How to update this file

Keep it short. If you've learned something a future agent will need:

- Cross-component contract → [`02-contracts.md`](./02-contracts.md)
- Local detail → your component plan
- Decided → [`decisions.md`](./decisions.md)
- Not designing yet → [`deferred.md`](./deferred.md)
- Cross-cutting agent norm → here
