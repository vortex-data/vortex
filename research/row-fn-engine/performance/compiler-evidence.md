<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Compiler evidence for the local probe

This is compiler evidence for the September 21 ARM artifact. It predates the
[RowFn updates](../current-system/recent-changes.md), including the separate Boolean retry work.

The direct iterator and RowFn both vectorize in the measured binary. Their main traversal shapes
differ. The standard iterator processes eight `i64` values per vector-loop iteration. The RowFn
collector processes 64 values per full chunk, followed by a remainder path.

This evidence concerns Apple M4 Max, rustc 1.98.0, LLVM 22.1.8, one codegen unit, and `lto = "off"`.
It does not describe x86 code generation. It supports the distinction between the
[two measured baselines](local-measurements.md), not a claim that the chunk length alone explains
their timing difference.

## Source mechanism

The direct-collector baseline calls `i64::build_from` over a borrowed native slice. The ordinary RowFn
path reaches the same output method through a decoded argument source. `build_from` delegates to
`IndexedSourceExt::map_into`, which divides the input into full `CHUNK_LEN` chunks and a remainder.
The measured `CHUNK_LEN` is 64.

The direct-iterator baseline uses standard slice iteration and `collect::<Vec<i64>>()` instead.
Both implementations construct the same output array from their vector. See [the measured source](reproduction.md),
[output collection][output], [lane mapping][mapping], and [the chunk constant][chunk].

## Optimized LLVM IR

The final artifact contains concrete implementations in the downstream binary, where the `AddOne`
row function is instantiated. A library-only artifact cannot establish this result.

| Function | Final IR location | Observation |
| --- | --- | --- |
| `row_fn_overhead_probe::direct`. | Line 19970. | A vector loop loads, adds, and stores four `<2 x i64>` vectors before advancing by eight rows. |
| `row_fn_overhead_probe::direct_collector`. | Line 17072. | A full chunk contains 32 `<2 x i64>` additions, followed by a separate remainder path. |
| `execute_rows::<AddOne>`. | Line 10521. | Its dense non-constant route contains the same 64-row chunk shape and vector remainder. |

The compiler also emits paths for input representations and validity states that the fixture does
not use. Their presence is not evidence that the timed calls execute them. The fixture has primitive,
non-nullable, non-constant input, which selects the dense non-constant route.

The final IR is
`/tmp/row-fn-overhead-96bd521e/target/release/deps/row_fn_overhead_probe-5032bd499a2e3b16.ll`.
Its SHA-256 is `2fcb23e3e6b5457a1f7c77d5e26e1360cff12e17210d31149bedf2a2b4da9299`.

## Assembly confirmation

These are full loop bodies from the final emitted assembly. The `add.2d` instructions operate on two
64-bit lanes. Neither displayed loop calls a row callback indirectly.

The direct iterator's loop, assembly lines 19097 through 19107, advances by 64 bytes:

```asm
LBB63_9:
	ldp	q1, q2, [x10, #-32]
	ldp	q3, q4, [x10], #64
	add.2d	v1, v1, v0
	add.2d	v2, v2, v0
	add.2d	v3, v3, v0
	add.2d	v4, v4, v0
	stp	q1, q2, [x9, #-32]
	stp	q3, q4, [x9], #64
	subs	x11, x11, #8
	b.ne	LBB63_9
```

The RowFn dense loop, assembly lines 13199 through 13267, advances by 512 bytes:

```asm
LBB49_344:
	ldp	q1, q2, [x11, #-256]
	ldp	q3, q4, [x11, #-224]
	add.2d	v1, v1, v0
	add.2d	v2, v2, v0
	add.2d	v3, v3, v0
	add.2d	v4, v4, v0
	stp	q1, q2, [x12, #-256]
	stp	q3, q4, [x12, #-224]
	ldp	q1, q2, [x11, #-192]
	ldp	q3, q4, [x11, #-160]
	add.2d	v1, v1, v0
	add.2d	v2, v2, v0
	add.2d	v3, v3, v0
	add.2d	v4, v4, v0
	stp	q1, q2, [x12, #-192]
	stp	q3, q4, [x12, #-160]
	ldp	q1, q2, [x11, #-128]
	ldp	q3, q4, [x11, #-96]
	add.2d	v1, v1, v0
	add.2d	v2, v2, v0
	add.2d	v3, v3, v0
	add.2d	v4, v4, v0
	stp	q1, q2, [x12, #-128]
	stp	q3, q4, [x12, #-96]
	ldp	q1, q2, [x11, #-64]
	ldp	q3, q4, [x11, #-32]
	add.2d	v1, v1, v0
	add.2d	v2, v2, v0
	add.2d	v3, v3, v0
	add.2d	v4, v4, v0
	stp	q1, q2, [x12, #-64]
	stp	q3, q4, [x12, #-32]
	ldp	q1, q2, [x11]
	ldp	q3, q4, [x11, #32]
	add.2d	v1, v1, v0
	add.2d	v2, v2, v0
	add.2d	v3, v3, v0
	add.2d	v4, v4, v0
	stp	q1, q2, [x12]
	stp	q3, q4, [x12, #32]
	ldp	q1, q2, [x11, #64]
	ldp	q3, q4, [x11, #96]
	add.2d	v1, v1, v0
	add.2d	v2, v2, v0
	add.2d	v3, v3, v0
	add.2d	v4, v4, v0
	stp	q1, q2, [x12, #64]
	stp	q3, q4, [x12, #96]
	ldp	q1, q2, [x11, #128]
	ldp	q3, q4, [x11, #160]
	add.2d	v1, v1, v0
	add.2d	v2, v2, v0
	add.2d	v3, v3, v0
	add.2d	v4, v4, v0
	stp	q1, q2, [x12, #128]
	stp	q3, q4, [x12, #160]
	ldp	q1, q2, [x11, #192]
	ldp	q3, q4, [x11, #224]
	add.2d	v1, v1, v0
	add.2d	v2, v2, v0
	add.2d	v3, v3, v0
	add.2d	v4, v4, v0
	stp	q1, q2, [x12, #192]
	stp	q3, q4, [x12, #224]
	add	x11, x11, #512
	add	x12, x12, #512
	subs	x13, x13, #1
	b.ne	LBB49_344
```

The direct collector has the same eight groups of vector loads, additions, and stores at assembly
lines 16212 through 16280. Its registers differ, but it also advances by 512 bytes and has a
separate remainder path. The [same-collector timing control](local-measurements.md) retains this
collector while removing the RowFn batch wrapper.

The final assembly is
`/tmp/row-fn-overhead-96bd521e/target/release/deps/row_fn_overhead_probe-5032bd499a2e3b16.s`.
Its SHA-256 is `92618354e30366200991bf0e598b5fb75037fe0d7f73edc24e49396d5e6d4e0c`.

## Attribution limits

At 16,384 rows, switching the baseline from the ordinary iterator to the shared collector removes
most of the measured difference from RowFn. The remaining paired median is about 99 ns.
This establishes the collector as a necessary comparison boundary.

The experiment does not change `CHUNK_LEN`, rewrite the executor, or compare compiler versions.
It therefore does not establish an optimal chunk length or identify one instruction as the cause.
Instruction layout, cache behavior, and traversal shape can all affect the measured cost.

Static specialization permits the row callback to inline in this example. It does not guarantee
identical code for every element, output, preparation, or error policy. No code-size or compile-time
regression is claimed. Those require separate controlled measurements.

[output]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/output.rs
[mapping]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-compute/src/lane_kernels/map_into.rs
[chunk]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-compute/src/lane_kernels/mod.rs
