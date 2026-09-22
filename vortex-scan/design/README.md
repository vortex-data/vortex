# Composable scan planning

Design proposal for driving scans through planners, morsels, and optional
speculative planners. This directory contains design documents only. Its location
does not settle the implementation's crate boundaries.

Start with:

- [Traits and execution protocol](TRAITS.md): explicit IO/CPU states, planner and
  morsel outputs, per-successor speculative tickets, ownership, and a worker loop.
- [Pipeline and composition](PIPELINE.md): stage inputs and outputs, shared plan
  views, constructor composition, warming, file traversal, and implementation steps.
- [The `Next<Input>` composition API](PIPELINE.md#composing-planners-with-next):
  direct filter-to-projection and insertion of byte-based projection splitting.

## Design boundaries

- A planner can discover IO, perform CPU work, and produce more planners or morsels.
  A morsel can also perform IO before returning arrays. The public protocol uses
  explicit state machines, without futures or Rust iterators.
- Constructors compose stages through `Next<Input>`. Inputs are stage-specific;
  evaluation commonly passes views over shared plans with row selections.
  Constructing the next planner does not execute it.
- IO exposes the whole currently discovered batch. Layouts and constructor policy
  decide row splitting; the caller does not pass splits or budgets to `state()`.
- The default composition covers footer opening, file statistics, index/zone
  pruning, filtering, projection, and morsels. External-index/partition pruning is
  optional. Byte-based projection splitting is a candidate for the default.
- Warming optimises and walks a plan's filter and projection dependencies, announces
  their segment reads, and passes prepared plan views onward. It does not depend on
  the filter's speculative cursor.
- Each speculative successor has its own ticket. A parent can complete tickets as
  individual regions finish. The scheduler may discard preparation while retaining
  the information needed to construct required work cold.
- Portable descriptions, constructor policy, and ticket payloads are `Send + Sync`.
  Started objects belong to their worker, including while waiting. The initial
  scheduler uses local queues; moving unstarted descriptions is distinct from
  moving live execution.
- Morsels preserve their local row order. The driver controls global ordering,
  reordering, admission, and backpressure. Physical IO sharing and coalescing span
  all stages of a source.
- One reusable composition can open and scan successive files without constructing
  all file plans up front.

## Status

The documents distinguish the selected design direction from remaining API choices.
In particular, cold reconstruction, IO completion bindings, portable start
descriptions, speculative CPU outcomes, and ordering progress still need concrete
contracts before implementation. Examples are proposed Rust API shapes.

The intended rollout is a default driver and one complete real-file scan first,
then pruning/layout coverage, ordering and limits, optional warming/speculation,
and migration of all public scan entry points. See the
[implementation sequence](PIPELINE.md#steps-to-realise-the-system) and
[acceptance cases](PIPELINE.md#acceptance-cases-before-making-this-the-default).
