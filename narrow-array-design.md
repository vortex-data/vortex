<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# NarrowArray prototype

`NarrowArray` is an ordinary array encoding that keeps a wide logical integer dtype while
storing its values in a narrower child. This draft is stacked on comparison optimization
PR #9948, which specializes bitmap construction for primitive i8/u8 comparisons.

```text
NarrowArray(dtype = i64)
└── PrimitiveArray(dtype = i8)
```

The child can use any encoding with the required integer dtype, including bitpacking or a
dictionary. The wrapper stores no value buffers. Signedness and nullability must match;
the child must be strictly narrower. Nested wrappers are flattened.

## Construction

- `NarrowArray::try_new(child, dtype)` wraps an existing narrow child without scanning it.
- `NarrowArray::encode(primitive, ctx)` scans non-null bounds, selects the smallest fitting
  integer type of the same signedness, and converts the buffer once. It returns the original
  array if no smaller type fits. Existing exact bounds can avoid the scan.

This preserves a useful distinction from a general cast: every child value is representable
in the logical dtype, and widening preserves ordering and equality. The invariant allows
comparisons to use the child's dtype without inspecting individual values.

## Execution

| Operation | Behavior |
| --- | --- |
| Compare to a fitting constant | Cast the constant once and compare the narrow child. |
| Compare to an out-of-range constant | Produce the determined boolean values, preserving nulls. |
| Compare two NarrowArrays with the same child dtype | Compare their children directly. |
| Slice, filter, take, mask | Push into the child and retain the logical dtype. |
| Cast to the child's exact dtype | Unwrap the child. |
| Cast to another safely wider dtype | Rewrap the same child. |
| Scalar access | Return a scalar at the logical dtype. |
| Canonical execution | Execute a widening cast, allowing the child's cast kernel to participate. |
| Other operations, including mixed child widths | Use the existing canonical fallback. |

Selection may retain a lazy child. Calling `execute::<PrimitiveArray>` on the result still
requests a wide canonical buffer. The selection/comparison pipeline benchmarks execute through
the public array operations; they do not count construction of an unevaluated child as work done.

## Serialization

The prototype uses encoding ID `vortex.narrow` and one metadata byte for the child's PType.
Deserialization reconstructs the child at its stored width. Widening happens when a consumer
requests canonical execution, not while reading the NarrowArray.

The encoding is registered in the array session. No released file edition has been modified
to admit it, and the compressor does not automatically produce it.

## Scope and open work

This draft supports signed and unsigned primitive integers. It does not change
`PrimitiveArray::narrow`, decimal storage, or DecimalByteParts. DecimalArray already permits
coefficient storage widths independent of its logical decimal dtype; decimal kernels and
DecimalByteParts integration need a separate design and measurements.

There are no specialized arithmetic or aggregate kernels yet. Arithmetic must retain logical
overflow behavior, so operations such as `i64(120) + i64(120)` cannot simply execute in i8.
General cast expressions are not rewritten into NarrowArray, avoiding a loop with canonical
widening. Mixed-width comparison dispatch and automatic compressor selection remain open.
Canonical execution after take also needs attention: the existing dictionary cast rule
widens the full values child before gathering, which regresses the take-to-i64 benchmark.

The measured benefits and costs are recorded in [the benchmark report](narrow-array-benchmarks.md).
The report distinguishes buffer savings from whole-process memory usage and prebuilt-array
compute from encoding and materialization costs.
