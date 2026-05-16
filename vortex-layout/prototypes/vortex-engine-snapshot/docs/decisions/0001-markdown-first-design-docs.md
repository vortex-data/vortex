# 0001: Markdown-first design documentation

> **Status:** Accepted ADR.
> **Progress:** Still current.
> **Open questions:** none.

## Status

Accepted.

## Context

The engine design is expected to evolve quickly. The most important knowledge
is not just API usage, but rationale: why the engine has the lowering and
pipeline shape it does, why it avoids runtime-specific async traits, and how
Vortex shapes physical execution.

Embedding that narrative only in code comments would make it fragmented and
hard to review.

## Decision

Keep durable design documentation in `docs/` as plain Markdown. Structure it so
it can also be rendered by mdBook using `book.toml` and `docs/SUMMARY.md`.

Use code comments for local invariants and API documentation. Use design docs
for architecture and rationale. Use ADRs for decisions that constrain future
work.

## Consequences

Design changes should update docs in the same patch as code.

The docs can be reviewed without building a site.

The project can publish a docs site later without migrating content.

## Alternatives considered

Rustdoc-only documentation was rejected because it keeps rationale too close to
individual types and functions.

A full documentation site framework was rejected for now because it creates more
process before the engine architecture is stable.
