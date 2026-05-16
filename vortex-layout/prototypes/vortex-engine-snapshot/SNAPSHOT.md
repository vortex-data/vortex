# Vortex Engine Snapshot

This is a working-tree snapshot copied from `/Users/ngates/git/vortex-engine`.

- Source branch: `ngates/pipeline-scheduler`
- Source HEAD: `0a8f95e`
- Source worktree state at copy time: dirty `Cargo.lock`
- Copied on: 2026-05-16

The snapshot is intentionally kept outside the Rust workspace and outside
`vortex-layout/src`, so it is available for reading and prototyping without
changing normal Vortex builds.

## First Files To Read

- `src/physical_plan/mod.rs`: current physical-plan v2 overview.
- `src/physical_plan/lowering.rs`: lowering API and pipeline builder.
- `src/physical_plan/abi.rs`: batch, source, transform, sink, and runtime ABI.
- `src/physical_plan/runtime.rs`: worker/pipeline execution model.
- `src/physical_plan/spawn.rs`: `SpawnRuntime`, async CPU/IO work, priorities.
- `src/operator/mod.rs`: older operator/resource/proposal model.
- `src/domain/requirement.rs`: requirement and row-demand vocabulary.
- `src/resources/pruning.rs`: side-information resource example.
- `src/scheduler/*.rs`: scheduler/worker/turn prototype.
- `src/layouts/*.rs`: layout binding examples.
- `docs/reference/lowering-api.md`: concise lowering design reference.

## Why This Exists

The V2 layout `execute -> Stream` API hides admission control and makes it hard
to model bounded out-of-order work, in-order retirement, and sideways
information propagation. This snapshot preserves the pipeline/runtime prototype
so the scan/filter/projection design can be explored without first committing to
a direct integration path.

Use this as source material for a new Vortex-side execution prototype, not as a
drop-in module. In particular, the engine physical-plan v2 currently lacks row
demand and cancellation semantics, and its `Priority` / `IoCost` hints are not
yet a complete information-prioritized scheduler.
