//! F: Cranelift-at-build-time. The bytes emitted by `build.rs` (via
//! `cranelift-codegen`) are embedded via `include_bytes!` and fed through
//! the same `materialize()` path as D-spec.
//!
//! The kernel takes the effective compare constant as a SystemV argument,
//! so the runtime can supply it per-`compile_eq(c, r)` call without
//! Cranelift relocations. The broadcast cost is paid once per kernel call
//! and amortized across the many blocks each call processes — the
//! realistic shape for runtime-supplied query constants.
//!
//! Note: Cranelift 0.118's x64 backend only handles vector types up to
//! 128 bits (xmm), so this kernel uses i8x16/SSE2 rather than i8x32/AVX2.
//! Newer Cranelift (≥ 0.119) has wider ymm/zmm support but isn't available
//! in this sandbox's crates.io index. The mechanism is identical either
//! way; F's job is to demonstrate that the runtime is unchanged.

use core::ptr::NonNull;
use std::io;

use crate::{materialize, page_size};

/// Bytes produced by `build.rs` from a Cranelift IR function. No
/// relocations; self-contained x86-64 function with SystemV ABI.
const CRANELIFT_EQ_KERNEL: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cranelift_eq_kernel.bin"));

/// A kernel compiled at *build time* by Cranelift, materialised at runtime
/// by the same mmap+memcpy+mprotect path D-spec uses. This is F.
pub struct CraneliftKernel {
    page: NonNull<u8>,
    page_len: usize,
    entry: unsafe extern "sysv64" fn(*const u8, *mut u32, u64, u8),
    effective_const: u8,
}

unsafe impl Send for CraneliftKernel {}
unsafe impl Sync for CraneliftKernel {}

impl CraneliftKernel {
    /// Materialise the build-time-compiled kernel for the eq chain
    /// `(x + ffor_ref) == constant`. The constants are *not* baked into
    /// the code; they are folded by the call site (`effective_const =
    /// constant - ffor_ref`) and passed through the function ABI on every
    /// call, where the kernel broadcasts them once into an xmm register.
    pub fn compile_eq(constant: u8, ffor_ref: u8) -> io::Result<Self> {
        let effective_const = constant.wrapping_sub(ffor_ref);
        let page_len = page_size();
        // SAFETY: no relocations in the emitted bytes; nothing to patch.
        let page = unsafe { materialize(CRANELIFT_EQ_KERNEL, &[], page_len)? };
        // SAFETY: the page holds a valid sysv64 function emitted by Cranelift
        // for the signature `fn(*const u8, *mut u32, u64, u8)`.
        let entry: unsafe extern "sysv64" fn(*const u8, *mut u32, u64, u8) =
            unsafe { core::mem::transmute(page.as_ptr()) };
        Ok(Self {
            page,
            page_len,
            entry,
            effective_const,
        })
    }

    /// The folded effective constant the kernel was compiled with.
    pub fn effective_const(&self) -> u8 {
        self.effective_const
    }

    /// Process `n_blocks` 32-byte AVX2-equivalent blocks. The build-time
    /// Cranelift kernel internally iterates in 16-byte (xmm) increments —
    /// Cranelift 0.118's x64 backend doesn't yet emit ymm/zmm — so we pass
    /// `n_blocks * 2` 16-byte units to it. Total output (`n_blocks * 4`
    /// bytes) is identical to D-spec; the granularity just differs.
    ///
    /// # Safety
    /// `packed` must point to at least `n_blocks * 32` readable bytes;
    /// `out` to at least `n_blocks * 4` writable bytes.
    pub unsafe fn call(&self, packed: *const u8, out: *mut u32, n_blocks: usize) {
        let n_halves = (n_blocks * 2) as u64;
        // SAFETY: caller upholds buffer windows; `out` is written as a
        // sequence of u16 masks (alignment satisfied since u32 ≥ u16).
        unsafe { (self.entry)(packed, out, n_halves, self.effective_const) }
    }

    /// The raw kernel bytes, for inspection.
    pub fn bytes() -> &'static [u8] {
        CRANELIFT_EQ_KERNEL
    }
}

impl Drop for CraneliftKernel {
    fn drop(&mut self) {
        // SAFETY: page + page_len from materialize.
        unsafe {
            libc::munmap(self.page.as_ptr().cast(), self.page_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cranelift_kernel_matches_scalar_reference_for_many_constants() {
        const N: usize = 16;
        let mut packed = vec![0u8; N * 32];
        for (i, b) in packed.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(13).wrapping_add(5);
        }
        // Exercise both the algebraic-folding API and a variety of (c, r)
        // pairs to verify the kernel is genuinely parameterised, not baking
        // the constant at build time.
        for &(c, r) in &[(42u8, 7u8), (0, 0), (255, 1), (128, 200), (1, 255)] {
            let kernel = CraneliftKernel::compile_eq(c, r).expect("materialize");
            assert_eq!(kernel.effective_const(), c.wrapping_sub(r));
            let mut out = vec![0u32; N];
            // SAFETY: N*32 readable, N*4 writable.
            unsafe { kernel.call(packed.as_ptr(), out.as_mut_ptr(), N) };

            for i in 0..N {
                let block: &[u8; 32] = (&packed[i * 32..(i + 1) * 32]).try_into().unwrap();
                let mut want_lo: u16 = 0;
                let mut want_hi: u16 = 0;
                for (j, &b) in block[..16].iter().enumerate() {
                    if b.wrapping_add(r) == c {
                        want_lo |= 1u16 << j;
                    }
                }
                for (j, &b) in block[16..].iter().enumerate() {
                    if b.wrapping_add(r) == c {
                        want_hi |= 1u16 << j;
                    }
                }
                let want = u32::from(want_lo) | (u32::from(want_hi) << 16);
                assert_eq!(out[i], want, "block {i} (c={c} r={r}): cranelift mismatch");
            }
        }
    }

    #[test]
    fn cranelift_kernel_bytes_are_nonempty() {
        let bytes = CraneliftKernel::bytes();
        assert!(!bytes.is_empty(), "build.rs produced empty kernel");
        assert!(bytes.len() < 4096, "kernel surprisingly large: {} B", bytes.len());
    }
}
