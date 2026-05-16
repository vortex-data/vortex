# Architecture decision records

> **Status:** Accepted ADR index.
> **Progress:** ADRs that constrain the current engine. Add a new
> ADR when an accepted decision changes.
> **Open questions:** none.

Architecture decision records, or ADRs, capture consequential
decisions.

Use an ADR when a decision:

- changes the shape of the public execution model;
- creates or removes a subsystem;
- introduces a runtime requirement or dependency;
- chooses between plausible alternatives where future work needs
  to know which one was picked and why;
- creates a constraint future work must respect.

ADRs are append-only. If an accepted decision changes, add a new
ADR; the old one's body stays as written.

## Current decisions

- [0001 Markdown-first design documentation](0001-markdown-first-design-docs.md)
- [0003 Poll-based local operators](0003-poll-based-local-operators.md)
- [0006 Build a single-pass Scan API engine before shuffle](0006-scan-api-first.md)
- [0008 MPMC work-stealing channels and single-output operators](0008-mpmc-work-stealing-channels.md)
- [0009 Cargo workspace](0009-cargo-workspace.md)

## Template

Use the [ADR template](template.md) when adding a decision.
