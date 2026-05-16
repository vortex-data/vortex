<div class="vortex-hero">
  <img class="vortex-hero-logo vortex-hero-logo-light" src="assets/brand/vortex-wordmark-black.svg" alt="Vortex">
  <img class="vortex-hero-logo vortex-hero-logo-dark" src="assets/brand/vortex-wordmark-white.svg" alt="Vortex">
  <h1 class="vortex-sr-only">vortex-engine documentation</h1>
</div>

> **Status:** Accepted documentation entry point.
> **Progress:** Reading paths follow the engine's three documentation
> levels — concepts, architecture, reference.
> **Open questions:** none.

This directory is the source of truth for the engine's design.
Public API docs explain how to call code; these documents explain
the model, the architecture, and the contracts the implementation
must respect.

The docs are organised in three reading levels:

1. **Concepts** (`concepts/`) — what the engine is and how its
   pieces fit together. Start here.
2. **Architecture** (`architecture/`) — how the runtime is built.
3. **Reference** (`reference/`) — precise APIs, ABIs, and operator
   contracts.

Plus three project areas:

- **Implementation** (`implementation/`) — code map, forward plan,
  open documentation tasks.
- **Decisions** (`decisions/`) — ADRs that constrain future work.
- **Contributing** (`contributing/`) — process docs.

## Reading paths

Start with [Engine overview](concepts/overview.md). Then follow
the path that matches the work.

- Understand the engine, in order:
  [Engine overview](concepts/overview.md),
  [Execution model](concepts/execution-model.md),
  [Runtime filters](concepts/runtime-filters.md),
  [I/O and spawn](concepts/io-and-spawn.md),
  [Runtime architecture](architecture/runtime.md).
- Write a planner or a new operator:
  [Execution model](concepts/execution-model.md), then
  [Lowering API](reference/lowering-api.md) and
  [Runtime traits](reference/runtime-traits.md).
- Add an async source or a Vortex-backed scan:
  [I/O and spawn](concepts/io-and-spawn.md) and
  [Spawn primitives](reference/spawn-primitives.md).
- Implementation context:
  [Implementation plan](implementation/roadmap.md) and
  [Current scaffold](implementation/current-scaffold.md).
- Decisions that constrain future work:
  [Architecture decisions](decisions/README.md).

## Documentation standards

Every page carries a status banner immediately under its title:

```markdown
> **Status:** Accepted.
> **Progress:** One concise sentence about what is stable.
> **Open questions:** One concise sentence about remaining work, or `none`.
```

Status values: `Accepted` (stable contract), `Draft` (specified
but with open API details), `Example` (walkthroughs), `Current`
(implementation-state pages).

Voice rules, the docs checklist, and the diagram standard are in
[Writing documentation](contributing/writing-docs.md).
