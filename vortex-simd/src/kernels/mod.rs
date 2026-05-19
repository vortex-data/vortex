//! Kernel registry: one [`Kernels`] struct holding every dispatched function
//! as a `fn` pointer, one table per supported tier, and one [`AtomicPtr`]
//! that picks the active table on first call.
//!
//! Each per-tier table is built with struct-update fallback against
//! [`SCALAR_TABLE`]: a tier only spells the slots it specializes, and every
//! other slot inherits the scalar implementation. Adding kernel #1001 is one
//! field on [`Kernels`] and one line per tier that has a real
//! specialization for it — every other tier falls back automatically.

pub mod generic;
pub mod scalar;

use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use crate::cpu::{Tier, tier};

/// Registry of every dispatched kernel, one `fn` pointer per slot.
///
/// Each slot is either the best specialized kernel for the active tier or
/// the scalar fallback. The slot is never `None` — every kernel is always
/// callable.
#[derive(Copy, Clone)]
pub struct Kernels {
    /// Element-wise wrapping `i32` add: `out[i] = a[i].wrapping_add(b[i])`.
    pub i32_add: fn(&[i32], &[i32], &mut [i32]),
    /// Element-wise `i32` equality. `out` is a packed bitmap (LSB-first),
    /// length `(a.len() + 7) / 8`.
    pub i32_eq: fn(&[i32], &[i32], &mut [u8]),
    /// Tier this table was built for. Exposed for diagnostics.
    pub tier: Tier,
}

/// The fallback table. Every other tier inherits from this via struct-update
/// (`..SCALAR_TABLE`); a tier only declares the slots it actually
/// specializes.
pub const SCALAR_TABLE: Kernels = Kernels {
    i32_add: scalar::add_i32,
    i32_eq: scalar::eq_i32,
    tier: Tier::SCALAR,
};

// `..SCALAR_TABLE` is the whole design — a tier only spells the slots it
// specializes and every other slot inherits scalar. Clippy correctly
// observes that with today's two-field `Kernels` the update is redundant;
// it stops being redundant the moment we add kernel #3.
#[allow(clippy::needless_update)]
#[cfg(target_arch = "x86_64")]
mod x86_tables {
    use super::{Kernels, SCALAR_TABLE};
    use crate::arch::x86_64 as x;
    use crate::cpu::Tier;

    // Trampolines turn `unsafe fn` (target-feature gated) into a plain
    // `fn` pointer. Installed only when the tier check has confirmed the
    // feature is present.
    fn add_sse2(a: &[i32], b: &[i32], out: &mut [i32]) {
        // SAFETY: only installed when tier >= SSE42.
        unsafe { x::add_i32_sse2(a, b, out) }
    }
    fn eq_sse2(a: &[i32], b: &[i32], out: &mut [u8]) {
        // SAFETY: only installed when tier >= SSE42.
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

    pub(super) static SSE2: Kernels = Kernels {
        tier: Tier::SSE42,
        i32_add: add_sse2,
        i32_eq: eq_sse2,
        ..SCALAR_TABLE
    };
    pub(super) static AVX2: Kernels = Kernels {
        tier: Tier::AVX2,
        i32_add: add_avx2,
        i32_eq: eq_avx2,
        ..SCALAR_TABLE
    };
    pub(super) static AVX512: Kernels = Kernels {
        tier: Tier::AVX512,
        i32_add: add_avx512,
        i32_eq: eq_avx512,
        ..SCALAR_TABLE
    };
}

#[allow(clippy::needless_update)] // see x86_tables for rationale
#[cfg(target_arch = "aarch64")]
mod arm_tables {
    use super::{Kernels, SCALAR_TABLE};
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
        tier: Tier::NEON,
        i32_add: add_neon,
        i32_eq: eq_neon,
        ..SCALAR_TABLE
    };
}

static SCALAR: Kernels = SCALAR_TABLE;

/// Cache of the active table. Initialized lazily on the first call to
/// [`kernels`]; subsequent calls are one relaxed pointer load.
static ACTIVE: AtomicPtr<Kernels> = AtomicPtr::new(ptr::null_mut());

/// Returns the active kernel table for this host.
///
/// First call resolves the tier and atomically publishes the chosen table.
/// Every subsequent call is one relaxed load and a null-sentinel check.
/// Hoist it out of hot loops; the compiler will, but the explicit
/// `let kernels = kernels();` is the convention.
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
    let table = pick();
    ACTIVE.store(ptr::from_ref(table).cast_mut(), Ordering::Relaxed);
    table
}

fn pick() -> &'static Kernels {
    let active = tier();
    #[cfg(target_arch = "x86_64")]
    {
        if active >= Tier::AVX512 {
            return &x86_tables::AVX512;
        }
        if active >= Tier::AVX2 {
            return &x86_tables::AVX2;
        }
        if active >= Tier::SSE42 {
            return &x86_tables::SSE2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if active >= Tier::NEON {
            return &arm_tables::NEON;
        }
    }
    let _ = active;
    &SCALAR
}
