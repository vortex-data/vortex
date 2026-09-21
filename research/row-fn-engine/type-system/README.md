<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Can the type system be generic?

[Research overview](../README.md).

Yes, the execution core can avoid Vortex `DType`. Sharing a function across hosts also requires an
explicit agreement about value semantics, accepted inputs, and output construction. A generic
`type DType` does not establish that agreement.

The current framework already keeps `DType` outside its typed row loops. It uses `DType` to select
input and output capabilities, validate them, and derive output metadata. This is a useful extraction
boundary. [Current source trace](current-boundary.md).

An isolated Rust experiment compiled one shared binder and row loop with two independently compiled
host adapters. It also exercised borrowed views and rejected unsupported semantic types.
[Compiled proof and limits](compiled-proof.md).

The strongest candidate is a combination of three mechanisms:

1. Host adapters retain native types, schema metadata, and physical layout choices.
2. Functions use small semantic descriptions for the domains they understand.
3. Binding selects typed input views and output builders before row execution.

This is a proposal, not an implemented API. It permits one function implementation across supported
hosts without requiring every host to represent every type. Unsupported semantic mappings fail at
binding. [Proposed contract](portable-contract.md).

The practical answer depends on what “fully generic” means:

| Goal | Assessment |
| --- | --- |
| Execute typed row kernels without Vortex dependencies. | Feasible by separating host decoding and output construction. |
| Define one function for Vortex, Arrow, DataFusion, and DuckDB. | Feasible for a declared semantic contract and supported adapters. |
| Support new domains without editing a central enum. | Feasible through registered semantic capabilities and extension identity. |
| Accept every host type with no adapter or semantic agreement. | Not feasible. Matching storage does not establish matching values or operations. |
| Preserve every host's native function semantics under one name. | Not generally possible. Decimal division and timestamp behavior already differ. |
| Publish one Rust generic definition as a runtime plugin for arbitrary hosts. | Requires a separate binary interface and precompiled bindings. Rust generics do not provide this. |

The next useful artifact is a small extraction prototype with `i64`, UTF-8, and one metadata-bearing
domain. A timestamp or tensor example tests more than primitive arithmetic alone. The prototype must
preserve existing null and failure contracts before any broader type catalog is added.
[Migration boundary](portable-contract.md#first-prototype).

- [Where current code needs `DType`](current-boundary.md).
- [Three designs and their tradeoffs](alternatives.md).
- [The associated-type and built-in vocabulary option](associated-types.md).
- [Detailed inventory of type operations](dtype-inventory.md).
- [A proposed binding contract and API sketches](portable-contract.md).
- [The isolated Rust compilation experiment](compiled-proof.md).
- [Mappings that succeed, lose information, or fail](mappings.md).
- [Caller demand and definedness](../definedness/README.md).

Local observations refer to Vortex `96bd521eb0565555def2af7b8e97e96891728da6`. The design sketches are
abridged and uncompiled. The separate compilation experiment contains its exact source and results.
This section contains no benchmark evidence.
