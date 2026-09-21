<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Evidence and scope

[Research overview](README.md)

This investigation uses Vortex revision
[`96bd521eb0565555def2af7b8e97e96891728da6`](https://github.com/vortex-data/vortex/commit/96bd521eb0565555def2af7b8e97e96891728da6),
dated 2026-09-21. The research branch starts from that revision.

## How to read the claims

| Label | Meaning |
| --- | --- |
| Observed behavior | Inspected implementation or documented contract at the cited revision. |
| Measurement | An executed experiment with its environment and reproduction recorded. |
| Inference | A conclusion from stated contracts or observations, without an implementation experiment. |
| Proposal | A possible design, not a current API or a verified implementation. |
| Open question | Evidence remains insufficient to select or validate a design. |

Code sketches describe proposed boundaries. They are not compiled prototypes unless a page
explicitly records a compilation. A feasibility argument does not establish zero overhead.

## Sources

Vortex source links use the fixed revision above. Tracking issues provide historical motivation.
Current code takes precedence when issue descriptions differ.

The [integration research](integrations/README.md) records the inspected Arrow, DataFusion, and
DuckDB versions. Dependency versions in the lockfile take precedence over manifest minimums.
Unversioned external documentation includes an access date. Those pages can change after this
investigation.

The [prior-art notes](prior-art.md) cite the Velox SFI paper and Substrait specifications. The
[definedness notes](definedness/README.md) cite the relevant Velox evaluator contracts separately.

## Measurement boundaries

The [type abstraction experiment](type-system/compiled-proof.md) compiled and ran as separate Rust
crates without Vortex dependencies. Its expected borrow-escape compilation failed with `E0515`.
The retained source and commands establish the experiment's limited scope.

The [performance report](performance/README.md) owns the exact measurement status, reproduction,
and interpretation. No figure in this research represents an unimplemented portable library.

Host integration costs require real adapter measurements. Local Vortex measurements cannot
establish Arrow conversion costs, DuckDB callback costs, or end-to-end DataFusion query overhead.
An architecture diagram also cannot establish allocation counts or vectorization.

## Work included

- Source tracing of RowFn, its input/output capabilities, consumers, and evaluator dependencies.
- Primary-source research for the target hosts and relevant prior art.
- A comparison of generic type designs and a proposed extraction boundary.
- A compiled generic type experiment with independent adapter crates and borrowed input views.
- A demand/definedness contract with semantic and storage distinctions.
- A performance cost model, measurement evidence, and follow-up experiments.

This branch contains research Markdown. It does not change the production RowFn implementation or
claim that the proposed host adapters exist.

## Checks performed

The type experiment compiled its core, two adapters, and executable with warnings denied. Its
runtime assertions passed. The borrow-escape example produced the expected compiler rejection.

The performance experiment compiled an isolated downstream package and ran five timing processes.
A separate build counted allocation requests. Independent review reproduced the main table from
all 5,600 retained observations and matched the retained harness to its source hash.

Document review covered current contracts, cross-page consistency, relative links, local anchors,
pinned Vortex source paths and line ranges, and prose. The source audit and proposals do not require
a workspace build.

The repository test suite, Clippy, and workspace formatting did not run. Neither did native x86
timings or benchmarks of real host adapters. These are coverage limits, not failed checks.
