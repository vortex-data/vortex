# 0003: Poll-based local operators

> **Status:** Accepted ADR.
> **Progress:** The poll model is realised by the
> `SourceNode::poll_next`, `TransformNode::poll_next_output`, and
> `SinkNode::poll_send` / `poll_finish` traits.
> **Open questions:** none.

## Status

Accepted.

## Context

The engine needs to coordinate vectorized operators, async external
work, downstream resources, barriers, and back-pressure.

An `async fn next()` trait hides much of this state. It tends to
introduce boxed futures, object-safety workarounds, and broad
`Send` requirements that propagate through every layer above.

## Decision

The core runtime abstraction is **poll-based operator state
machines**. The three runtime traits — `SourceNode`,
`TransformNode`, `SinkNode` — expose `poll_*` methods that return
`Poll<EngineResult<T>>` and may return `Pending` after registering
a waker on the ctx's `Context`.

Operators that need to wait for async work spawn it via
`ctx.spawn` or `ctx.spawn_io` and poll the returned `WorkHandle<T>`
on later ticks. The pipeline driver suspends on `Pending` and is
re-polled when the relevant waker fires.

`Send + 'static` is required at the trait level because the work
pool migrates pipelines across threads, but no runtime-specific
bounds (e.g. Tokio) are imposed on the engine.

## Consequences

Operator implementations are explicit state machines rather than
opaque streams. This is acceptable because execution engines need
visible state machines for diagnostics and back-pressure.

The pipeline driver can see when an operator is waiting and on
what — the `WorkHandle` it polled returned `Pending`, the channel
it was sending to was full, the barrier it depends on hasn't
fired. None of that is hidden inside an `async fn`.

Future runtime adapters can wrap the poll surface in a
`futures::Stream` for embedding without forcing every operator to
become async.

## Alternatives considered

An `async_trait`-based operator interface was rejected because it
would obscure operator readiness and push the engine toward
global runtime and thread-safety constraints.

A pure `futures::Stream` operator interface was rejected because
it only models output polling and does not model input readiness,
output capacity, or resource readiness — all of which the engine
needs to express explicitly.
