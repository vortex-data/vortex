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

- [Recent changes](recent-changes.md) records the implemented improvements since September 21.
- [Execution](execution.md) follows one call through planning, constants, null handling, and output.
- [Dependency inventory](dependencies.md) identifies each Vortex dependency and its proposed owner.
- [Safety and errors](contracts.md) records the contracts that extraction must preserve.
- [Consumers and limits](consumers.md) shows actual users and function families outside this API.
- [Extraction boundary](extraction.md) proposes a small core, host adapters, and ownership rules.
- [Crate and trait choices](engine-boundary.md) compares an `Engine` interface with smaller capabilities.
- [Design history](design-history.md) retains the PR evidence and records abandoned proposals.

Two details differ from older designs. Filtered execution now writes directly into the original
output positions. Every sink must support initialized placeholders for skipped rows. There is no
optional sink initializer or compact-output scatter fallback in this revision.
[Sources: filtered execution][filtered], [sink initialization][sink].

Output collection now has an associated storage type through `OutputElement::Buffer` and
`OutputBuffer`. The executor uses the execution allocator for output payloads. Dense Boolean retry
packs directly, and initialization tokens require an unsafe operation tied to the callback's exact
row. These are implemented contracts to preserve during extraction.

The execution, dependency, safety, and extraction notes reflect commit
`d8e45e0898e02efed0822a6c74bf0d515a5b3b74` as inspected on September 25. The consumer inventory and
design history retain their September 21 scope. Proposed interfaces remain design conclusions,
not an implemented extraction. This source update ran no Rust checks. The
[performance notes](../performance/README.md) distinguish historical measurements from current work.

[rowfn]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/row_fn.rs
[filtered]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/batch/execute/filtered.rs
[sink]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/sink/mod.rs
