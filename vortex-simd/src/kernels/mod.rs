//! Kernel registry: one `Kernels` struct holding every dispatched function
//! as a `fn` pointer, one static per supported tier, and one `AtomicPtr`
//! that picks the active table at first call.
//!
//! This is the only dispatch surface for the crate. Adding a kernel is one
//! field on [`Kernels`] plus one line per tier table; adding a tier is one
//! new static table. The call site is always `(kernels().foo)(args..)` —
//! one indirect call, no branch.

pub mod generic;
pub mod scalar;

use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use crate::cpu::{Tier, tier};

/// Registry of every dispatched kernel, one `fn` pointer per slot.
///
/// Each field is either the best specialized kernel available for the active
/// tier or the scalar fallback. The slot is never `None` — every kernel is
/// always callable.
#[derive(Copy, Clone)]
pub struct Kernels {
    /// Element-wise wrapping `i32` add: `out[i] = a[i].wrapping_add(b[i])`.
    pub i32_add: fn(&[i32], &[i32], &mut [i32]),
    /// Element-wise `i32` equality. `out` is a packed bitmap (LSB-first),
    /// length `(a.len() + 7) / 8`.
    pub i32_eq: fn(&[i32], &[i32], &mut [u8]),
    /// Tier this table was built for. Exposed for diagnostics; the call
    /// site does not need it.
    pub tier: Tier,
}

/// The default table. Always safe to fall back to.
static SCALAR_TABLE: Kernels = Kernels {
    i32_add: scalar::add_i32,
    i32_eq: scalar::eq_i32,
    tier: Tier::SCALAR,
};

#[cfg(target_arch = "x86_64")]
mod x86_tables {
    use super::Kernels;
    use crate::arch::x86_64 as x;
    use crate::cpu::Tier;
    use crate::kernels::scalar;

    // Trampolines lift `unsafe fn` (target-feature gated) into a normal
    // `fn` pointer. They are only installed into a tier table when the
    // tier check has confirmed the feature is present.
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

    // Slots are either the per-tier specialization or the scalar fallback.
    // SSE2 does not have a specialized eq beyond the hand-tuned one; AVX2
    // and AVX-512 have specialized eq. If a future op has no SSE2 variant,
    // fill its slot with `scalar::*` and the table stays complete.
    pub(super) static SSE2: Kernels = Kernels {
        i32_add: add_sse2,
        i32_eq: eq_sse2,
        tier: Tier::SSE42,
    };
    pub(super) static AVX2: Kernels = Kernels {
        i32_add: add_avx2,
        i32_eq: eq_avx2,
        tier: Tier::AVX2,
    };
    pub(super) static AVX512: Kernels = Kernels {
        i32_add: add_avx512,
        i32_eq: eq_avx512,
        tier: Tier::AVX512,
    };

    // Reference the fallback so the unused-import lint stays quiet when
    // future tables fall back here.
    #[allow(dead_code)]
    fn _fallback() -> fn(&[i32], &[i32], &mut [i32]) {
        scalar::add_i32
    }
}

#[cfg(target_arch = "aarch64")]
mod arm_tables {
    use super::Kernels;
    use crate::arch::aarch64 as neon;
    use crate::cpu::Tier;

    fn add_neon(a: &[i32], b: &[i32], out: &mut [i32]) {
        // SAFETY: only installed when tier >= NEON.
        unsafe { neon::add_i32_neon(a, b, out) }
    }
    fn eq_neon(a: &[i32], b: &[i32], out: &mut [u8]) {
        // SAFETY: only installed when tier >= NEON.
        unsafe { neon::eq_i32_neon(a, b, out) }
    }

    pub(super) static NEON: Kernels = Kernels {
        i32_add: add_neon,
        i32_eq: eq_neon,
        tier: Tier::NEON,
    };
}

/// Cache of the active table. Initialized lazily on the first call to
/// [`kernels`]; subsequent calls are one relaxed pointer load.
static ACTIVE: AtomicPtr<Kernels> = AtomicPtr::new(ptr::null_mut());

/// Returns the active kernel table for this host.
///
/// First call resolves the tier and atomically publishes the chosen table.
/// Every subsequent call is one relaxed load and a null-sentinel check.
/// Hoist this out of hot loops; the compiler will, but the explicit `let
/// kernels = kernels();` is the convention.
#[inline(always)]
pub fn kernels() -> &'static Kernels {
    let raw = ACTIVE.load(Ordering::Relaxed);
    if !raw.is_null() {
        // SAFETY: only `init` writes into ACTIVE, and it only stores
        // pointers to `'static` tables in this module.
        return unsafe { &*raw };
    }
    init()
}

#[cold]
#[inline(never)]
fn init() -> &'static Kernels {
    let table: &'static Kernels = pick();
    ACTIVE.store(ptr::from_ref(table).cast_mut(), Ordering::Relaxed);
    table
}

fn pick() -> &'static Kernels {
    let active_tier = tier();
    #[cfg(target_arch = "x86_64")]
    {
        if active_tier >= Tier::AVX512 {
            return &x86_tables::AVX512;
        }
        if active_tier >= Tier::AVX2 {
            return &x86_tables::AVX2;
        }
        if active_tier >= Tier::SSE42 {
            return &x86_tables::SSE2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if active_tier >= Tier::NEON {
            return &arm_tables::NEON;
        }
    }
    let _ = active_tier;
    &SCALAR_TABLE
}
