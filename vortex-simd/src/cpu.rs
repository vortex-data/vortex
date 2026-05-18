//! Runtime CPU feature detection.
//!
//! The first call to [`tier`] (or any of the convenience accessors) runs
//! detection. Every subsequent call is a single relaxed byte load plus a
//! statically-cold branch on a sentinel.

use core::sync::atomic::{AtomicU8, Ordering};

/// A SIMD capability tier.
///
/// Tiers within the same architecture are ordered: a higher value implies
/// strictly more capability than every lower value on that architecture.
/// Comparing tiers across architectures is meaningless (the constants only
/// exist on the architecture they apply to).
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct Tier(u8);

impl Tier {
    /// Scalar fallback. Always available.
    pub const SCALAR: Self = Self(0);

    /// SSE2 + SSE4.2 (x86_64 baseline). 128-bit vectors.
    #[cfg(target_arch = "x86_64")]
    pub const SSE42: Self = Self(1);

    /// AVX2. 256-bit integer vectors.
    #[cfg(target_arch = "x86_64")]
    pub const AVX2: Self = Self(2);

    /// AVX-512 F + BW + DQ + VL. 512-bit vectors with mask registers.
    #[cfg(target_arch = "x86_64")]
    pub const AVX512: Self = Self(3);

    /// ARMv8 Advanced SIMD (NEON). 128-bit vectors. Baseline on aarch64.
    #[cfg(target_arch = "aarch64")]
    pub const NEON: Self = Self(1);

    /// Raw tier value. Stable across releases for a given variant.
    #[inline(always)]
    pub const fn as_u8(self) -> u8 {
        self.0
    }
}

impl core::fmt::Debug for Tier {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self.0 {
            0 => "SCALAR",
            #[cfg(target_arch = "x86_64")]
            1 => "SSE42",
            #[cfg(target_arch = "x86_64")]
            2 => "AVX2",
            #[cfg(target_arch = "x86_64")]
            3 => "AVX512",
            #[cfg(target_arch = "aarch64")]
            1 => "NEON",
            _ => "UNKNOWN",
        };
        write!(f, "Tier::{}", name)
    }
}

/// Sentinel meaning "not yet detected". Chosen so any valid tier byte differs.
const UNINIT: u8 = u8::MAX;

static CACHED: AtomicU8 = AtomicU8::new(UNINIT);

/// Returns the detected SIMD tier for this binary on this CPU.
///
/// Cost: one relaxed byte load and a sentinel compare on every call after
/// the first. The compare is `#[cold]`-biased so the predictor learns it
/// after one iteration.
#[inline(always)]
pub fn tier() -> Tier {
    let v = CACHED.load(Ordering::Relaxed);
    if v == UNINIT {
        return detect_and_cache();
    }
    Tier(v)
}

#[cold]
#[inline(never)]
fn detect_and_cache() -> Tier {
    let t = detect();
    CACHED.store(t.0, Ordering::Relaxed);
    t
}

#[cfg(target_arch = "x86_64")]
fn detect() -> Tier {
    // AVX-512 F + BW + DQ + VL is the floor we treat as "AVX-512" — it covers
    // every integer width we want to vectorize over with masked stores.
    if std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512vl")
    {
        return Tier::AVX512;
    }
    if std::is_x86_feature_detected!("avx2") {
        return Tier::AVX2;
    }
    if std::is_x86_feature_detected!("sse4.2") {
        return Tier::SSE42;
    }
    Tier::SCALAR
}

#[cfg(target_arch = "aarch64")]
fn detect() -> Tier {
    // NEON is mandatory on aarch64. We still check via the std feature macro
    // so cross-compiled builds without it fall back cleanly.
    if std::arch::is_aarch64_feature_detected!("neon") {
        return Tier::NEON;
    }
    Tier::SCALAR
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect() -> Tier {
    Tier::SCALAR
}

/// `true` iff [`tier`] is at least AVX2. Always `false` off x86_64.
#[inline(always)]
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        tier() >= Tier::AVX2
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `true` iff [`tier`] is AVX-512. Always `false` off x86_64.
#[inline(always)]
pub fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        tier() >= Tier::AVX512
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `true` iff NEON is available. Always `false` off aarch64.
#[inline(always)]
pub fn has_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        tier() >= Tier::NEON
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_returns_a_tier() {
        let t = tier();
        assert!(t >= Tier::SCALAR);
    }

    #[test]
    fn cached_after_first_call() {
        let a = tier();
        let b = tier();
        assert_eq!(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn tier_ordering_x86() {
        assert!(Tier::SSE42 < Tier::AVX2);
        assert!(Tier::AVX2 < Tier::AVX512);
        assert!(Tier::SCALAR < Tier::SSE42);
    }
}
