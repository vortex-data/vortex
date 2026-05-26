# vortex-iter

Batch-first iterators that autovectorize across data layouts and operations.

The unit of work is a fixed-width lane block `[T; N]` rather than a single
element. A zero-copy source reinterprets a slice as `[T; N]` blocks via
`slice::as_chunks` and yields each block by value, so the inner loop over `N`
lanes stays a tight, countable, branch-free loop that LLVM turns into SIMD.

## Verifying SIMD

`verify-simd.sh` emits the assembly for the kernels in `examples/simd_subjects.rs`
and asserts that each one contains wide vector instructions, performs no
`memcpy` (proving the by-value blocks stay register-resident), and falls back to
no scalar loop.

```bash
./vortex-iter/verify-simd.sh
```
