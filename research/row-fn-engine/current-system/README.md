<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# What RowFn currently owns

[Research overview](../README.md).

RowFn can become a portable execution library, but a generic type descriptor is only one part of
that work. Input decoding, output construction, validity, allocation, and optimizer metadata also
cross the Vortex boundary.

The useful separation is between typed row computation and host column operations. A row kernel
already receives Rust values, references, or row writers. The Vortex-specific work surrounds that
kernel. The current contracts also restrict which functions fit. RowFn preserves input nulls and
cannot produce a null from valid inputs.
[Source: RowFn contract][rowfn].

The detailed notes follow that boundary:

- [Execution](execution.md) follows one call through planning, constants, null handling, and output.
- [Dependency inventory](dependencies.md) identifies each Vortex dependency and its proposed owner.
- [Safety and errors](contracts.md) records the contracts that extraction must preserve.
- [Consumers and limits](consumers.md) shows actual users and function families outside this API.
- [Extraction boundary](extraction.md) proposes a small core, host adapters, and ownership rules.

Two details differ from older designs. Filtered execution now writes directly into the original
output positions. Every sink must support initialized placeholders for skipped rows. There is no
optional sink initializer or compact-output scatter fallback in this revision.
[Sources: filtered execution][filtered], [sink initialization][sink].

These notes describe commit `96bd521eb0565555def2af7b8e97e96891728da6`. Source inspection supports the
current-behavior claims. Proposed interfaces are design conclusions, not an implemented extraction.
This source audit did not run Rust checks. Local measurements are recorded in the
[performance notes](../performance/README.md).

[rowfn]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/row_fn.rs#L22-L45
[filtered]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/execute/filtered.rs#L4-L20
[sink]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/sink/mod.rs#L90-L123
