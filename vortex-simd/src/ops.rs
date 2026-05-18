//! Function-pointer dispatch tables, the primary public API.
//!
//! Each integer type implements [`IntOps`], exposing a `'static` reference to
//! its kernel table. The table is built once on first access from the tier
//! reported by [`crate::cpu::tier`].
//!
//! ```
//! # use vortex_simd::ops::IntOps;
//! let a = [1_i32, 2, 3, 4, 5, 6, 7, 8];
//! let b = [1_i32, 0, 3, 0, 5, 0, 7, 0];
//! let mut out_add = [0_i32; 8];
//! let mut out_eq = [0_u8; 1];
//! (i32::ops().add)(&a, &b, &mut out_add);
//! (i32::ops().eq)(&a, &b, &mut out_eq);
//! assert_eq!(out_eq[0], 0b0101_0101);
//! ```

use core::sync::atomic::{AtomicPtr, Ordering};

use crate::cpu::{Tier, tier};
use crate::kernels::scalar;

/// Kernel table for one integer element type.
///
/// Fields are `fn` pointers, not `unsafe fn`: the constructor has already
/// verified the runtime tier supports them, so callers can invoke without
/// `unsafe`.
#[derive(Copy, Clone)]
pub struct IntKernels<T: 'static> {
    /// Element-wise wrapping add: `out[i] = a[i].wrapping_add(b[i])`.
    pub add: fn(&[T], &[T], &mut [T]),
    /// Element-wise equality. `out` is a packed bitmap, LSB-first, length
    /// `(a.len() + 7) / 8`.
    pub eq: fn(&[T], &[T], &mut [u8]),
    /// The tier these kernels were selected for. Exposed for diagnostics.
    pub tier: Tier,
}

/// Marker trait wiring an integer type to its kernel table.
pub trait IntOps: Sized + 'static {
    /// Returns the `'static` kernel table for this type.
    ///
    /// First call resolves the tier and atomically publishes a pointer to a
    /// `'static` table. Every subsequent call is a single relaxed pointer
    /// load and a sentinel-null compare with cold init.
    fn ops() -> &'static IntKernels<Self>;
}

// ---------- i32 ----------

impl IntOps for i32 {
    #[inline(always)]
    fn ops() -> &'static IntKernels<i32> {
        static CACHE: AtomicPtr<IntKernels<i32>> = AtomicPtr::new(core::ptr::null_mut());
        let p = CACHE.load(Ordering::Relaxed);
        if !p.is_null() {
            // SAFETY: only `init_i32_kernels` ever stores into CACHE, and it
            // stores a pointer to a `'static`.
            return unsafe { &*p };
        }
        init_i32_kernels(&CACHE)
    }
}

#[cold]
#[inline(never)]
fn init_i32_kernels(cache: &'static AtomicPtr<IntKernels<i32>>) -> &'static IntKernels<i32> {
    let table: &'static IntKernels<i32> = pick_i32_table();
    cache.store(
        table as *const IntKernels<i32> as *mut IntKernels<i32>,
        Ordering::Relaxed,
    );
    table
}

fn pick_i32_table() -> &'static IntKernels<i32> {
    let t = tier();
    #[cfg(target_arch = "x86_64")]
    {
        if t >= Tier::AVX512 {
            return &I32_AVX512;
        }
        if t >= Tier::AVX2 {
            return &I32_AVX2;
        }
        if t >= Tier::SSE42 {
            return &I32_SSE2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if t >= Tier::NEON {
            return &I32_NEON;
        }
    }
    let _ = t;
    &I32_SCALAR
}

static I32_SCALAR: IntKernels<i32> = IntKernels {
    add: scalar::add_i32,
    eq: scalar::eq_i32,
    tier: Tier::SCALAR,
};

#[cfg(target_arch = "x86_64")]
mod x86_tables {
    use super::IntKernels;
    use crate::arch::x86_64 as x;
    use crate::cpu::Tier;

    // Trampolines lift `unsafe fn` (target-feature gated) into a normal
    // `fn` pointer. They are only ever installed into a table when the tier
    // check has confirmed the feature is present.
    fn add_sse2(a: &[i32], b: &[i32], out: &mut [i32]) {
        // SAFETY: SSE2 is the x86_64 baseline; we additionally only install
        // this entry when tier >= SSE42.
        unsafe { x::add_i32_sse2(a, b, out) }
    }
    fn eq_sse2(a: &[i32], b: &[i32], out: &mut [u8]) {
        // SAFETY: see add_sse2.
        unsafe { x::eq_i32_sse2(a, b, out) }
    }
    fn add_avx2(a: &[i32], b: &[i32], out: &mut [i32]) {
        // SAFETY: only installed when tier >= AVX2.
        unsafe { x::add_i32_avx2(a, b, out) }
    }
    fn eq_avx2(a: &[i32], b: &[i32], out: &mut [u8]) {
        // SAFETY: only installed when tier >= AVX2.
        unsafe { x::eq_i32_avx2(a, b, out) }
    }
    fn add_avx512(a: &[i32], b: &[i32], out: &mut [i32]) {
        // SAFETY: only installed when tier >= AVX512.
        unsafe { x::add_i32_avx512(a, b, out) }
    }
    fn eq_avx512(a: &[i32], b: &[i32], out: &mut [u8]) {
        // SAFETY: only installed when tier >= AVX512.
        unsafe { x::eq_i32_avx512(a, b, out) }
    }

    pub(super) static I32_SSE2: IntKernels<i32> = IntKernels {
        add: add_sse2,
        eq: eq_sse2,
        tier: Tier::SSE42,
    };
    pub(super) static I32_AVX2: IntKernels<i32> = IntKernels {
        add: add_avx2,
        eq: eq_avx2,
        tier: Tier::AVX2,
    };
    pub(super) static I32_AVX512: IntKernels<i32> = IntKernels {
        add: add_avx512,
        eq: eq_avx512,
        tier: Tier::AVX512,
    };
}

#[cfg(target_arch = "x86_64")]
use x86_tables::{I32_AVX2, I32_AVX512, I32_SSE2};

#[cfg(target_arch = "aarch64")]
mod arm_tables {
    use super::IntKernels;
    use crate::arch::aarch64 as a;
    use crate::cpu::Tier;

    fn add_neon(a_: &[i32], b: &[i32], out: &mut [i32]) {
        // SAFETY: only installed when tier >= NEON.
        unsafe { a::add_i32_neon(a_, b, out) }
    }
    fn eq_neon(a_: &[i32], b: &[i32], out: &mut [u8]) {
        // SAFETY: only installed when tier >= NEON.
        unsafe { a::eq_i32_neon(a_, b, out) }
    }

    pub(super) static I32_NEON: IntKernels<i32> = IntKernels {
        add: add_neon,
        eq: eq_neon,
        tier: Tier::NEON,
    };
}

#[cfg(target_arch = "aarch64")]
use arm_tables::I32_NEON;
