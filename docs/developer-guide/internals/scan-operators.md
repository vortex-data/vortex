# Scan operators

This page is for implementers who want to add a pipeline operator, or understand how a compiled
plan runs. For plans, see [Scan plans](scan-planning.md). For the planners that run them, see
[Scan execution](scan-execution.md).

A plan says *what* to compute. When a planner runs a plan over some rows, the plan is compiled into
**pipelines** of **operators**, and those pipelines do the computing. Plans are the unit of
optimization; operators are the unit of execution.

## Why pipelines

- **Nothing blocks.** Operators compute one batch per call and say what they wait for, so a
  pipeline never holds a thread while it waits for IO. Whoever runs the scan, usually a planner,
  owns the thread and the IO.
- **Work follows the rows.** Each operator does work proportional to the rows it is handed. It never
  looks at a batch twice, never holds more than the batches it has to join, and never reads inlets
  it does not need.
- **Masks are applied at the read.** Every mask is known when the pipelines are compiled, so the
  stage that reads the rows applies it. Chunks a mask selects nothing of are never built and never
  read.
- **Decoded data is shared.** A segment that several readers need is decoded once and fanned out,
  and it is freed as soon as no reader can still ask for it.
- **Plans and operators are separate.** A plan operator such as `Pack` can compile to different
  operator chains depending on its rows and mask, and the optimizer never has to know how an
  operator runs.

## The model

```text
               pipeline                                 pipeline
  ┌──────────────────────────────────┐       ┌───────────────────────────────┐
  │ Source → Operator → … → Operator │─port─►│ Source (inlet 0) → Operator … │─port─► …
  └──────────────────────────────────┘       └───────────────────────────────┘
```

- A **pipeline** is one `Source`, zero or more `Operator` stages, and the **ports** its last stage
  pushes into, called its outlets.
- A **port** is a queue of batches with one sink (the pipeline that pushes into it) and one source
  (the pipeline that reads it as an **inlet**). The plan that joins pipelines sets how many batches
  each inlet can hold: `DEFAULT_CAPACITY` (8), or unbounded for an inlet the source needs whole.
- A **split** is a row range of the plan and a mask over it. `pipeline::Scan` compiles each split
  into pipelines and runs them.

The scheduler asks a source for a batch, pushes it through the stages, and writes the result into
the outlets. A pipeline is **blocked** when its source waits for a segment read or an inlet, or when
one of its outlets is full. It is re-run when that one condition clears.

## The contracts

Both traits live in `vortex_layout::plan::physical::operator`.

### Operator

```rust
pub trait Operator: Send {
    fn compute(&mut self, input: Input, cx: &mut Cx<'_>) -> VortexResult<Step>;
}
```

| `Input` | Meaning |
| --- | --- |
| `Batch(array)` | The next batch from the stage below. |
| `None` | No new input: continue after `Step::More`. Sources always get `None`. |
| `End` | The stage below has finished. Handed over once. |

| `Step` | Meaning |
| --- | --- |
| `More(array)` | A batch, with more to come for this input. Called again with `None` before new input. |
| `Final(array)` | The last batch for this input. |
| `End` | Nothing more for this input. After `Input::End`, the stage has finished. |

A stage that maps each batch is the simplest operator. This is the stage `Eval` compiles to:

```rust
impl Operator for EvalStage {
    fn compute(&mut self, input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        let Input::Batch(batch) = input else {
            return Ok(Step::End);
        };
        Ok(Step::Final(batch.apply_bound(&self.expression)?))
    }
}
```

A stage that splits a batch returns `More` for each piece but the last, and `Final` for the last.
A stage that buffers returns `End` for each batch it holds, then flushes when it gets `Input::End`.

### Source

```rust
pub trait Source: Operator {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        Ok(SourceReady::Ready)
    }
}
```

`ready` runs before every `compute` of a source, after the scheduler has checked its outlets. It
returns one of:

| `SourceReady` | Meaning |
| --- | --- |
| `Ready` | The next `compute` can produce output or finish without blocking. |
| `Blocked(Io)` | Wait for segment bytes. Queue the reads with `cx.request_read(segment)` first. |
| `Blocked(Inlet(i))` | Wait until inlet `i` gets a batch or is closed. |

A source then computes with `Input::None`. `Step::Final` means "check my readiness again", and
`Step::End` finishes the source.

### What `Cx` gives an operator

| Method | Use |
| --- | --- |
| `request_read(segment)` | Queue a segment read, from `ready`. |
| `has_bytes()`, `take_segment()` | Arrived segments, oldest first. |
| `inlet(i)` | The reading side of inlet `i`: `peek_mut`, `take`, `closed`, `is_empty`, `len`, `rows`, `full`. |
| `changed_inlet()` | An inlet whose sink pushed or closed since the last readiness check. Each is reported once. |
| `session()`, `exec()` | The session and execution context, for decoding and evaluation. |

Taking a batch from an inlet is what frees its slot, so a source that only peeks keeps its
producer blocked once the inlet is full.

## How plans compile to operators

`PlanVTable::compile(plan, rows, mask, compiler)` builds a `Chain`: a source and the stages after
it, before its outlets are attached. A plan does one of two things:

- **Extend a child's chain.** Compile the child with `compiler.compile(child, rows, mask)` and add a
  stage with `chain.with(stage)`. `Eval` does this.
- **Join children behind a new source.** Compile each child, then call
  `compiler.join(chains, source)`. Each child chain becomes its own pipeline writing a new port, and
  inlet `i` of `source` reads chain `i`. `join_with` sets a capacity per inlet. `Pack`, `Concat` and
  `Take` do this.

Compiling does no IO. Return `None` when the mask selects no rows, and the parent leaves that child
out.

## Built-in operators

| Operator | Kind | Work | Compiled from |
| --- | --- | --- | --- |
| segment scan | Source | Read one segment, decode it, slice it to the split, apply the mask | `SegmentScan` |
| `EvalStage` | Stage | Apply an expression to each batch | `Eval` |
| filter | Stage | Keep the selected rows of each dense batch | `Filter` |
| concat | Source | Emit its inlets one after another | `Concat` |
| pack | Source | Zip validity and field inlets into structs | `Pack` |
| take | Source | Wrap code batches as dictionaries over the values inlet, once it has ended | `Take` |
| whole | Source | Collect an unbounded inlet whole, then emit one batch made from it | `Share` (shared values), `Zoned` (the zone table) |
| once | Source | Emit one batch it was built with | `Share` and `Zoned` when the value is already cached, `Pack` with no fields to read |

## Sharing

Two kinds of data are shared, so they are decoded once.

- **Segments.** Before a scan starts, one pass over the plans of every split records which rows
  each segment is read for. The first reader of a segment that is read more than once builds a
  single pipeline that decodes it and fans the result out, one single-use port per reader. Ports
  for readers that never run, because their rows were pruned, are freed when that pipeline
  finishes. A decoded segment lives only as long as a split that may still read it.
- **Whole values.** Values a plan always produces whole, such as dictionary values under a `Share`
  plan, are kept in a `ValueCache` and reused across filters, projections and splits.

## Writing an operator

1. Decide whether it is a **stage** (one input, transform each batch) or a **source** (reads
   segments or joins inlets).
2. Do work proportional to the rows you are handed. Do not look at a batch twice or buffer more than
   you must.
3. In a source, do all waiting in `ready`: queue reads or name the inlet you wait for. `compute`
   must not block.
4. Take batches from inlets as soon as you have used them, so producers are not held up.
5. Compile to it from a plan's `compile`, with `Chain::with` for a stage or `Compiler::join` for a
   source with inlets.
