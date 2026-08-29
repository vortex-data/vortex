# vortex-morsel-push

An experimental morsel-driven scan executor for Vortex layouts — the P1 spine of the design in
`docs/developer-guide/internals/scan-execution-models/morsel-based-plan-execution.md`.

A scan is cut into *morsels* (contiguous root row ranges). Each morsel is driven by a tree of
stateful `ExecNode` state machines, inline and depth-first, by one affinity-owning worker.
`next_plan` *names* reads by registering keyed uses against the IO plane. `execute` can try a
source-provided non-blocking inline read for a required ticket; on a miss it suspends on that exact
ticket while workers service the background IO queues.

The crate is a prototype and is not part of the public API. It supports flat, chunked and
struct layouts only; anything else is a build error rather than a fallback.

The source commit and the scoped update procedure are recorded in [UPSTREAM.md](UPSTREAM.md).

Cross-morsel decode reuse comes from **leased shared cells**, not a cache: lease counts are
computed from the morsel cut before the scan starts, the first morsel to decode a unit publishes
it, every retiring morsel releases its lease, and the last release drops the array. No budget, no
eviction policy, nothing outliving the scan; the ledger is asserted to drain to zero. Sharing can
be disabled (`with_share_decodes(false)`), leaving no state across morsels at all — the
state-for-state fairness row against V1.

## Measured

Against the V1 `LayoutReader` on shape-matched workloads (see
`docs/.../morsel-prototype-p1-findings.md` for the full contract and caveats): geomean 0.539 at
equal thread count (0.644 with sharing disabled), 0.249 at four threads with coalesced morsels,
with every configuration validated against V1's output before timing.

## Running the evaluation

```bash
cargo run --release -p vortex-morsel-push --features _test-harness --bin morsel-push-eval
```
