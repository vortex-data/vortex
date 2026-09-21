<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Performance findings

[Research overview](../README.md)

RowFn overhead is measurable, but one percentage hides different costs. Type dispatch and batch
validation add work outside the row loop. Decoding, null handling, and output collection can change
the algorithm that the loop executes.

The local experiment compares the current framework with two equivalent canonical `i64` kernels.
It includes decode, new output allocation, array construction, and output destruction.

- Against the same output collector, RowFn adds about 101 to 104 ns for batches of 1 to 1,024 rows.
- At 16,384 rows, RowFn takes about 2.53 us versus 1.44 us for a direct iterator kernel.
  A direct kernel with the shared RowFn collector takes about 2.43 us.
- Reused-argument RowFn and both direct kernels issue the same allocation requests in this fixture.
  Constructing fresh arguments adds one 16-byte allocation and associated ownership operations.

The shared-collector comparison measures the additional batch wrapper while holding one framework
component fixed. It does not establish that total RowFn overhead is about 100 ns. Both comparisons
matter. These results apply to Apple M4 Max, rustc 1.98.0, and the measured release configuration.

Read the findings in this order:

1. [Local measurements](local-measurements.md) gives the results, baseline contracts, and limits.
2. [Cost model](cost-model.md) separates setup, decoding, row work, validity, output, and host costs.
3. [Measurement plan](measurement-plan.md) specifies the remaining targets and demand-mask cases.
4. [Compiler evidence](compiler-evidence.md) explains the observed collector distinction.
5. [Reproduction](reproduction.md) retains the exact harness and commands.
6. [Raw observations](raw-observations.md) contains every final timing and allocation observation.

A separate library needs this separation in its API and benchmark suite. A generic type abstraction
can preserve static row dispatch, but that alone does not preserve output code generation or host
representation. No Arrow, DataFusion, DuckDB, x86 runtime, or query benchmark was run in this study.
