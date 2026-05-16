# vortex-engine

> **Status:** Early design and scaffold.
> **Progress:** Documentation now uses the V1 relation model; production
> operators are not implemented yet.
> **Open questions:** physical plan API, Vortex scan binding, scheduler policy,
> and concrete operator conformance tests.

<p align="center">
  <img src="docs/assets/brand/vortex-wordmark-black.svg" alt="Vortex" width="340">
</p>

`vortex-engine` is the physical execution layer for Vortex-backed query plans.
It is intended to sit below a higher-level planner, which can own logical
planning, row-domain and relation reasoning, and expression rewriting.

## Documentation

Design documentation lives in [`docs/`](docs/README.md). The docs are
Markdown-first and are structured so they can be read directly in GitHub or
rendered with mdBook later.

The initial crate focuses on a small set of execution contracts:

- batches carry Vortex arrays plus row-domain metadata;
- operators are poll-based and runtime-neutral;
- futures are local by default and do not require `Send` or `Sync`;
- cross-domain structure is represented with task-local `Relation`s and
  streamed relation witnesses;
- dynamic filtering is represented as shared domain-keyed selection state;
- the crate starts as a monocrate until the natural boundaries are clearer.

The first likely split, if needed, is:

- core operator and batch traits;
- runtime adapters such as Tokio or thread-per-core execution;
- concrete scan/join/filter operators.
