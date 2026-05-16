# Repository instructions

## Scope

These instructions apply to the whole `vortex-engine` repository.
Read them whenever you start a session in this repo.

## What the engine is

`vortex-engine` is a single-node, scheduler-driven query engine
for ordered Vortex-native data. The implementation lives under
`src/physical_plan/`. The shape:

- A planner submits a `PhysicalPlan` rooted at an `Operator`.
- `Operator::lower(ctx, tail)` walks the tree in CPS style and
  emits a DAG of single-input pipelines.
- Each pipeline is a flat linear chain `SourceNode → TransformNode → … → SinkNode`
  driven by a per-thread executor.
- Cross-pipeline coordination uses sticky one-shot
  `PipelineBarrier`s and typed `Resource`s captured by closures.
- Async I/O and other awaits offload through `ctx.spawn` /
  `ctx.spawn_io`, returning a `WorkHandle<T>` the operator polls
  on later ticks.
- Sideways information passing is expressed as **typed runtime
  filters** (`RangeFilter<K>`, `BloomFilter<K>`,
  `KeyListFilter<K>`) published as `Resource`s. The engine has
  no row demand.

The full picture is in [docs/concepts/overview.md](docs/concepts/overview.md).
The exact APIs are in `docs/reference/`.

## Working model

- The source of truth for the design is `docs/`. The mdBook
  output under `book/` is generated; never hand-edit it.
- Read and edit source Markdown directly. Prefer existing pages
  to new ones.
- Keep design and code changes in one patch when they touch
  execution model, plan/runtime APIs, pipelines, barriers,
  resources, or runtime filters.
- Use `rg` / `rg --files` for searches.
- Never revert user changes in the worktree.

## Doc tree (three reading levels)

A new page must belong to exactly one of:

- **Concepts** — `docs/concepts/` — what the engine is, why it's
  shaped this way. Approachable. Short.
- **Architecture** — `docs/architecture/` — how the runtime is
  built. Ownership, layering.
- **Reference** — `docs/reference/` — precise APIs, ABIs,
  operator contracts.

If a page is trying to be two of these, split it. Concepts must
not plunge into ABI detail.

Entry points: [docs/README.md](docs/README.md),
[docs/SUMMARY.md](docs/SUMMARY.md),
[docs/glossary.md](docs/glossary.md),
[docs/concepts/overview.md](docs/concepts/overview.md).

## Status banners

Every page has this immediately under its top-level heading:

```markdown
> **Status:** Accepted.
> **Progress:** One concise sentence about what is stable.
> **Open questions:** One concise sentence about remaining work, or `none`.
```

Status values: `Accepted`, `Draft`, `Example`, `Current`. Do not
invent new statuses without an ADR.

## Voice

Docs are written for an advanced developer who already knows Rust
and query engines.

- Normative for accepted contracts: `must`, `must not`, `is`.
- `should` only for policy with intentional latitude.
- No marketing tone, hype, anthropomorphism, jokes, or
  reassurance.
- Write from the accepted current model. State what the system
  is and does; keep "not X" wording for explicit non-goals or
  contractual negatives.
- Define terms before using them.
- Separate semantic facts from runtime mechanisms.
- Examples must be concrete enough to test.
- Record open questions explicitly.

## Vocabulary

The accepted vocabulary is enumerated in
[docs/glossary.md](docs/glossary.md). The terms that show up most
often:

- `Operator`, `lower(ctx, tail)`, `PipelineTail`, `LoweringCtx`.
- `PhysicalPlan`, `LoweredPlan`, `PipelineBuilder`.
- `Pipeline`, `PipelineBarrier`, `Resource`, `BarrierRegistry`,
  `LatchBarrier`.
- `SourceNode`, `TransformNode`, `SinkNode`, plus the `Dyn*` /
  `*Driver` type-erasure layer.
- `OperatorPoll`, `Context`, `Waker`, `TransformOutput`,
  `PendingSend`.
- `SpawnRuntime`, `WorkHandle<T>`, `IoCost`, `DriverIo`.
- `RangeFilter<K>`, `BloomFilter<K>`, `KeyListFilter<K>`.
- `Parallelism::Serial` / `LaneSafe { max_lanes }`.
- `Domain`, `DomainSpan`, `OutputContract`, `Batch`.

## Engine invariants

These are the contracts code and docs must respect:

- An operator's `lower` either prepends a transform onto the
  tail, completes the tail at a source via `ctx.emit_pipeline`,
  or splits it (multi-input or pipeline-breaker by emitting more
  pipelines linked by barriers).
- Inside a pipeline there are no channels. Operators talk
  through direct method calls. Channel exchanges appear only
  between pipelines.
- A pipeline is a flat linear chain: one source, zero+
  transforms, one sink.
- Multi-input operators emit multiple pipelines from one `lower`
  call. The runtime never sees multi-input.
- Cross-pipeline coordination uses sticky one-shot
  `PipelineBarrier`s. A sink that `publishes(barrier)` fires it
  when `poll_finish` returns Ready. A pipeline that
  `depends_on(barrier)` does not start until the barrier fires.
- Cross-pipeline data handoff uses typed `Resource`s captured
  by closures in the sink and a downstream source or transform.
- Operators must not block inside `poll_*`. They offload via
  `ctx.spawn` / `ctx.spawn_io` and poll the returned
  `WorkHandle<T>` on later ticks.
- Dropping a `WorkHandle` abandons the result — the engine has
  no cancellation.
- Operator state must be `Send`. Spawned futures must be `Send`.
- The engine has no row demand. Anything that wants finer
  feedback than typed runtime filters either adds a new filter
  type with explicit semantics or comes back with concrete
  numbers and an ADR proposal.

## Rust style

`STYLE.md` is the canonical Rust guide. Highlights:

- One import per `use` line; three groups (`std`, external,
  `crate`/`super`) separated by blank lines; sorted within each
  group.
- Import the type, then use the short name. No fully-qualified
  paths in bodies.
- `pub` is the crate's external surface; `pub(crate)` is for
  cross-module use; `pub(super)` is for siblings.
- One operator per file in `src/operators/` and
  `src/physical_plan/<op>.rs`; helpers in `*_common.rs`.
- `mod.rs` only declares `mod` and re-exports.
- Every `pub` item has a doc comment; the first sentence is one
  line and a self-contained summary.
- Comments explain non-obvious *why*. Don't paraphrase the code.
- `EngineError::message` + `?` for engine errors. No `unwrap` /
  `expect` outside tests except for genuine invariants with a
  descriptive message.
- `unsafe` blocks have `// SAFETY:` immediately above them.

## Repo-local skills

Skills under `.claude/skills/` codify recurring patterns. Use
them when the trigger conditions in their descriptions match.

- [docs-discipline](.claude/skills/docs-discipline/SKILL.md) —
  status banners, voice, doc-tree routing. Use on every docs
  edit.
- [adr-writing](.claude/skills/adr-writing/SKILL.md) — when
  recording or revising a decision under `docs/decisions/`.
- [self-improve](.claude/skills/self-improve/SKILL.md) — where
  corrections, validated approaches, and project state land so
  the same feedback isn't needed twice.

## Self-improvement

The user expects me to compound corrections. When the user
corrects me, validates an unusual choice, or surfaces an
invariant that wasn't written down:

1. Decide the destination using the
   [self-improve](.claude/skills/self-improve/SKILL.md) skill's
   routing table. Repo-wide invariants land in this `AGENTS.md`;
   docs/process rules go into a skill; personal preferences go
   into memory.
2. Write the rule, then a one-line **Why:** that captures the
   reason or incident, and a one-line **How to apply:** that
   says when it kicks in. A rule without a reason rots.
3. Edit, don't append. If a similar rule already exists, update
   it instead of adding a near-duplicate.
4. Tell the user briefly that the lesson was recorded.

Watch for quiet validations too ("yes exactly", silently
accepted unusual choices). They are as valuable as corrections
and easier to miss.

## What does not belong here

- Long design rationale — that lives in `docs/concepts/` and the
  ADRs.
- Code-shape conventions covered by `STYLE.md`.
- Skill bodies — they live under `.claude/skills/<name>/SKILL.md`.

If something in this file grows past one screen, consider
splitting it into a skill.
