# Summary

> **Status:** Accepted navigation.
> **Progress:** Table of contents follows the three doc levels:
> concepts, architecture, reference.
> **Open questions:** none.

# Start

- [Home](README.md)
- [Engine overview](concepts/overview.md)
- [Glossary](glossary.md)

# Concepts

- [Execution model](concepts/execution-model.md)
- [Runtime filters](concepts/runtime-filters.md)
- [I/O and spawn](concepts/io-and-spawn.md)

# Architecture

- [Runtime architecture](architecture/runtime.md)

# Reference

- [Lowering API](reference/lowering-api.md)
- [Runtime traits](reference/runtime-traits.md)
- [Spawn primitives](reference/spawn-primitives.md)
- [Runtime filters reference](reference/runtime-filters.md)

# Project

- [Implementation plan](implementation/roadmap.md)
- [Current scaffold](implementation/current-scaffold.md)
- [Documentation TODOs](implementation/todo.md)
- [Architecture decisions](decisions/README.md)
  - [0001 Markdown-first design documentation](decisions/0001-markdown-first-design-docs.md)
  - [0003 Poll-based local operators](decisions/0003-poll-based-local-operators.md)
  - [0006 Build a single-pass Scan API engine before shuffle](decisions/0006-scan-api-first.md)
  - [0008 MPMC work-stealing channels and single-output operators](decisions/0008-mpmc-work-stealing-channels.md)
  - [0009 Cargo workspace](decisions/0009-cargo-workspace.md)
  - [ADR template](decisions/template.md)
- [Writing documentation](contributing/writing-docs.md)
