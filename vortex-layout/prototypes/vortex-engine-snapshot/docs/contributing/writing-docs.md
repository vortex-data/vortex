# Writing documentation

> **Status:** Accepted documentation process.
> **Progress:** Defines the status-banner format, voice, and the
> three-level doc-tree routing for new pages.
> **Open questions:** none.

Documentation is part of the implementation. If a change alters
the execution model, runtime contract, or public API, update the
docs in the same patch.

## What belongs where

A new page belongs to exactly one of three reading levels:

- **Concepts** (`docs/concepts/`) — what the engine is and how
  its pieces fit together. Approachable to an advanced developer.
  Short. Links down to architecture and reference.
- **Architecture** (`docs/architecture/`) — how the runtime is
  built. Ownership, layering, where things live.
- **Reference** (`docs/reference/`) — precise APIs, ABIs, and
  operator contracts. Long is fine.

If a page is trying to be two of these, split it.

Other locations:

- `docs/decisions/` — ADRs.
- `docs/implementation/` — code map, forward plan, open
  documentation tasks.
- `docs/contributing/` — process docs.

`SUMMARY.md` places source-of-truth pages into the reading path
by document intent.

Use Rustdoc comments for public API usage and local invariants.

## Status banners

Every documentation page must include a short banner immediately
below the top-level heading:

```markdown
> **Status:** Accepted.
> **Progress:** One concise sentence about what is stable.
> **Open questions:** One concise sentence about remaining work, or `none`.
```

Status values:

- `Accepted` — stable contract.
- `Draft` — implementer-facing surface that still has open API
  details.
- `Example` — walkthroughs.
- `Current` — implementation-state pages
  (`current-scaffold.md`).

## Voice

AI-authored docs must stay technical, concise, and precise:

- Prefer normative language for accepted contracts: `must`,
  `must not`, `is`.
- Use `should` only for intended policy with known implementation
  latitude.
- Avoid marketing tone, hype, jokes, and anthropomorphic
  explanations.
- Write from the accepted current model: state what the system is
  and does. Keep "not X" wording for explicit non-goals or
  contractual negatives.
- Define terms before examples use them.
- Separate semantic facts from runtime mechanisms.
- Keep examples concrete enough to test.
- Record open questions explicitly instead of hiding uncertainty
  in vague prose.

## Design-doc checklist

Every major design document should answer:

- What problem does this solve?
- What are the non-goals?
- What invariants must hold?
- What are the failure modes?
- What alternatives were considered?
- Which files implement this today?
- What questions remain open?

## Diagram checklist

Use diagrams where the reader needs to understand shape before
details:

- ownership hierarchy;
- pipeline DAG or state-machine flow;
- boundaries between core code and runtime adapters.

Keep the editable source as Mermaid in `docs/diagrams/*.mmd` (the
directory is created on demand). Check in an SVG beside it when
the diagram should render in mdBook without an additional
preprocessor.

Avoid diagrams that merely restate a bullet list.

## ADR checklist

Every ADR should include:

- status;
- context;
- decision;
- consequences;
- alternatives considered.

ADRs are append-only. If an accepted decision changes, add a new
ADR; the old one's body stays as written.
