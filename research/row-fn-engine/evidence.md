<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Evidence and scope

[Overview](README.md)

This tree combines source findings, historical design decisions, experiments, and proposed APIs.
The synthesis changes research Markdown only. It does not change the RowFn implementation.

## Source snapshots

The main Vortex reference is
[`96bd521eb0565555def2af7b8e97e96891728da6`](https://github.com/vortex-data/vortex/commit/96bd521eb0565555def2af7b8e97e96891728da6).
The x86 measurements and some source inventories use
[`f5b3b26cf895e73a7db4b138337d9388b98570ce`](https://github.com/vortex-data/vortex/commit/f5b3b26cf895e73a7db4b138337d9388b98570ce).
The RowFn module has no changes between those revisions. The experiments still use different
workloads, machines, and build configurations.

The [integration pages](integrations/README.md#source-snapshots) pin Arrow Rust 59.3.0,
DataFusion 55.1.0, DuckDB 1.5.5, and Arrow C++ 25.0.1. Broader survey notes identify upstream source
paths inspected on 2026-09-21. Unpinned links and historical issue states describe that snapshot.
They are not assertions about subsequent releases or current issue status.

During synthesis, source inspection confirmed the Velox adapter's existing `Status` path and
Substrait's built-in type coverage. Those corrections appear in the detailed prior-art pages.

The [design history](current-system/design-history.md) records PR and issue findings. Pinned source
takes precedence over tracker text when the two differ.

## How to read claims

| Kind | Meaning |
| --- | --- |
| Source finding. | The inspected implementation or contract supports the claim. |
| Measurement. | An experiment ran with the stated workload and environment. |
| Inference. | The evidence suggests a conclusion without isolating it experimentally. |
| Proposal. | A possible API or implementation that does not exist in this branch. |
| Open question. | The evidence does not yet select or establish a design. |

Trait sketches are uncompiled unless their page explicitly records a compilation. A feasible
generic interface does not establish equal performance or semantic compatibility across hosts.

## Retained experiments

| Experiment | Completed work | Retained evidence | Limits |
| --- | --- | --- | --- |
| [Generic type proof](type-system/compiled-proof.md). | Separate core and adapter crates compiled. Runtime assertions passed. A borrow escape failed with `E0515`. | Exact source, commands, compiler version, and reported output. | Small safe-indexed example. No extracted executor, unsafe sink, null handling, or real host adapter. |
| [ARM overhead](performance/local-measurements.md). | Five timing processes, separate allocation counters, optimized IR, and assembly inspection. | Harness, all 5,600 observations, allocation records, compiler excerpts, and artifact hashes. | Unary canonical non-nullable `i64`. One compiler configuration. No controlled CPU affinity or frequency. |
| [x86 overhead](performance/x86-measurements.md). | Existing Divan benchmark and a temporary sweep over seven row counts. | Result tables, environment, scenario definitions, and commands. | Sweep source, raw output, and fitting script were not retained. No IR or assembly inspection ran. |

These experiments predate the synthesis. They were not rerun while creating the combined branch.
The earlier ARM record reports reproduction of its table from retained observations. This synthesis
does not claim a new reproduction of those timings.

For x86, the documented protocol permits a new experiment with similar scenarios. It cannot
reproduce the deleted sweep byte for byte. The stock benchmark remains available in the repository.
No missing experiment artifact is reconstructed and presented as original evidence.

## Reconciled conclusions

The main pages use the following distinctions throughout:

- A host type parameter removes coupling. Shared function semantics require an additional contract.
- A small built-in vocabulary is useful. Requiring every host to support every kind is unnecessary.
- Outer nullability can move to a binding envelope. Nested nullability remains part of type meaning.
- Demand requests rows. Completion records results, including nulls. Validity identifies non-null results.
- Safe null placeholders are an output policy. They do not replace completion metadata for partial reuse.
- Semantic infallibility and decode infallibility are already separate in current RowFn.
- Other engines have binding phases. RowFn's plan-reproduction check is the narrower distinctive mechanism.
- Timings suggest optimization targets. They do not prove a particular compiler cause without compiler evidence.

The [design](architecture.md) gives the recommendation. The detailed files preserve alternatives
without treating them as accepted APIs.

## Synthesis checks

The work reviewed source contracts, research text, local links, and branch contents. The repository
commit hook also ran `cargo +nightly fmt --all -- --check` and `taplo fmt --check`. Both passed.
No Rust or TOML files changed.

## Work not performed

No production adapter, extracted crate, demand-aware evaluator, or optimization was implemented.
No test suite, Clippy, build, benchmark, or experimental harness ran during the synthesis.

The x86 study supplies native x86 timings. It does not supply x86 compiler evidence. The ARM study
supplies both timings and compiler evidence for its fixture. Neither supplies host-conversion or
end-to-end query measurements.
