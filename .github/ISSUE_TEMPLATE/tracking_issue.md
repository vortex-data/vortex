---
name: Tracking Issue
about: Implementation context for work likely to span multiple PRs.
title: "Tracking Issue: "
labels: tracking-issue
---

<!--
Tracking issues are for work that needs shared implementation context and is
likely to span multiple PRs. Use this template when the issue should track
planning, implementation, stabilization, and follow-through.

Route discussion elsewhere:

- New ideas, fit questions, or early design feedback: start with a GitHub
  Discussion.
- Bug reports surfaced while implementing: file a bug issue and reference this
  tracking issue.
- A design thread big enough to need its own page: spawn a tracking issue (as
  a sub-issue of this one if it is narrow) and discuss it there.

Tracking issues should be understandable a year from creation. Short clarifying
comments are fine. Long threads should be redirected.
-->

This is a tracking issue for ...

<!--
One or two sentences. State what the feature is and where it lives in the
codebase (crate, module, file format, encoding). Link the parent Epic, design
doc, prototype PR, or external reference.
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
