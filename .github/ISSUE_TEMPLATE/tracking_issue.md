---
name: Tracking Issue
about: A tracking issue for a feature or initiative in Vortex.
title: "Tracking Issue: "
labels: tracking-issue
---

<!--
Thank you for opening a tracking issue. Tracking issues record a feature's
progress from design to landing and connect related bugs, PRs, and design
questions. They should stay readable a year from now.

Route discussion elsewhere:

- Bug reports surfaced while implementing: file a bug issue and reference this
  tracking issue.
- A design thread big enough to need its own page: spawn a tracking issue (as
  a sub-issue of this one if it is narrow) and discuss it there.

Short clarifying comments are fine. Long threads should be redirected.
-->

This is a tracking issue for ...

<!--
State what the feature is and where it lives in the codebase (crate, module,
file format, encoding). Link the parent Epic, design doc, prototype PR, or
external reference if any.
-->

## Motivation

<!--
Optional. Skip if this is a sub-issue of an Epic that already covers the why.

For standalone tracking issues, cover:

- The concrete problem and who hits it. Examples: a downstream integration
  (DataFusion, DuckDB, Python, Java), a benchmark gap (TPC-H, ClickBench,
  vortex-bench), a regression against Parquet or Arrow, or a file format or
  encoding limitation.
- What is currently blocked, slower, or larger than it should be. Link the
  benchmark numbers, profile output, or related issues.
-->

## Design

<!--
Optional but recommended. Size to fit the work:

- API additions: a stripped-down signature block. Rust traits or functions,
  file-format struct, or wire-format message. Drop doc comments and bodies.
- Behavioral changes: a short before / after description, ideally with a small
  example.
- Architectural changes: a paragraph explaining the new shape, or a link to a
  design doc.

Skip only if the change is fully obvious from the title. The Design section is
what lets a reviewer judge whether the Steps below are the right ones.
-->

## Steps

<!--
Major checkpoints required to call this feature done. A handful of milestones,
not every PR. Tick steps as they land; do not delete them.

Steps are milestones, not PRs (initial implementation, documentation,
stabilization). If a step is itself a separable shippable unit, promote it to
a sub-issue.
-->

- [ ] Initial implementation
- [ ] Documentation
- [ ] Public API stabilization

## Unresolved questions

<!--
Open design or implementation questions blocking progress. Link discussions
and conclusions so a future reader can see how each question was settled.

If a question is large enough to need its own thread, spawn a tracking issue
or Discussion and link it.
-->

- [ ] None yet.

## Implementation history

<!--
A running log of every PR that touched this feature: initial implementation,
follow-ups, fixes, reverts. Grows continuously and is never ticked off. The
archaeology trail for someone reading this issue a year later.

Add PRs as they merge, in chronological order. A one-line description per PR
is helpful but not required.
-->
