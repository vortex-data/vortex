# vortex-simd

Runtime CPU feature detection and SIMD kernels for Vortex integer compute.

The detection layer is zero-cost after the first call (a single relaxed byte
load on the fast path). Dispatch is exposed in two flavors:

- **Function-pointer table** (`i32::ops().add(...)`) — the primary API. One
  indirect call, no branch, easy to extend without macro changes.
- **Tier-match macro** (`vortex_simd::dispatch!`) — keeps the per-tier code
  inlined at the call site for callers that care.

Direct per-tier kernels are also `pub`, for callers that have already
committed to a tier on a hot path.

Proof scope: `i32` add and `i32` equality on `Scalar`, `SSE2`, `AVX2`,
`AVX-512` (x86_64) and `Scalar`, `NEON` (aarch64). Other primitives follow
the same shape.
