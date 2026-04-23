// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Platform-portable cardinality estimator.
//!
//! On 64-bit targets this delegates to Cloudflare's
//! [`cardinality_estimator`](https://crates.io/crates/cardinality-estimator), which is exact up
//! to ~128 distinct values and then transitions to a HyperLogLog++ representation. That crate is
//! gated by `#[cfg(target_pointer_width = "64")]` because it stores its state in a tagged
//! `usize`, so it cannot compile on 32-bit platforms. To keep the compressor portable we fall
//! back to the pure-Rust [`hyperloglogplus`](https://crates.io/crates/hyperloglogplus) crate on
//! non-64-bit targets, using a HyperLogLog++ with sparse representation that gives comparable
//! quality within the standard HLL++ error bound (~1.6% at the default precision).

use std::hash::Hash;

/// Approximate distinct-count estimator with a tiny, stable surface area.
///
/// The estimator is exact for small cardinalities on 64-bit targets and approximate beyond. On
/// non-64-bit targets it is a HyperLogLog++ sketch throughout; collisions in the sparse
/// representation make small-cardinality estimates approximate as well, though the error stays
/// well within the standard HLL++ bound.
pub(crate) struct CardinalityEstimator<T: Hash + ?Sized> {
    /// Platform-selected backend (Cloudflare on 64-bit, HLL++ elsewhere).
    inner: inner::Estimator<T>,
}

impl<T: Hash + ?Sized> CardinalityEstimator<T> {
    /// Create a new estimator with the default precision.
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            inner: inner::Estimator::new(),
        }
    }

    /// Insert a hashable item into the estimator.
    #[inline]
    pub(crate) fn insert(&mut self, item: &T) {
        self.inner.insert(item);
    }

    /// Return the current cardinality estimate.
    ///
    /// Takes `&mut self` because the 32-bit fallback's `count` implementation mutates internal
    /// caches; the 64-bit implementation is logically `&self` but is wrapped uniformly.
    #[inline]
    pub(crate) fn estimate(&mut self) -> usize {
        self.inner.estimate()
    }
}

impl<T: Hash + ?Sized> Default for CardinalityEstimator<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Backend implementations selected at compile time by pointer width.
#[cfg(target_pointer_width = "64")]
mod inner {
    use std::hash::Hash;

    /// Thin wrapper around Cloudflare's tagged-pointer cardinality estimator.
    pub(super) struct Estimator<T: Hash + ?Sized> {
        /// The Cloudflare estimator using its default `P=12, W=6` parameters.
        inner: cardinality_estimator::CardinalityEstimator<T>,
    }

    impl<T: Hash + ?Sized> Estimator<T> {
        /// Construct a fresh estimator at default precision.
        #[inline]
        pub(super) fn new() -> Self {
            Self {
                inner: cardinality_estimator::CardinalityEstimator::new(),
            }
        }

        /// Forward an insertion to the underlying estimator.
        #[inline]
        pub(super) fn insert(&mut self, item: &T) {
            self.inner.insert(item);
        }

        /// Forward to the underlying constant-time estimate.
        #[inline]
        pub(super) fn estimate(&mut self) -> usize {
            self.inner.estimate()
        }
    }
}

/// 32-bit fallback using the `hyperloglogplus` crate.
#[cfg(not(target_pointer_width = "64"))]
mod inner {
    use std::hash::BuildHasherDefault;
    use std::hash::Hash;

    use hyperloglogplus::HyperLogLog as _;
    use hyperloglogplus::HyperLogLogPlus;
    use rustc_hash::FxHasher;
    use vortex_error::VortexExpect;

    /// HLL++ precision exponent: 2^12 = 4096 registers, matching Cloudflare's default (~1.6% error).
    const PRECISION: u8 = 12;

    /// HyperLogLog++ sketch over `T`, hashed with a deterministic `FxHasher`.
    pub(super) struct Estimator<T: Hash + ?Sized> {
        /// Backing HLL++ sketch.
        inner: HyperLogLogPlus<T, BuildHasherDefault<FxHasher>>,
    }

    impl<T: Hash + ?Sized> Estimator<T> {
        /// Construct a fresh estimator at the configured `PRECISION`.
        #[inline]
        pub(super) fn new() -> Self {
            // `HyperLogLogPlus::new` only fails if precision is outside `[4, 18]`; 12 is a
            // compile-time constant inside that range.
            let inner = HyperLogLogPlus::new(PRECISION, BuildHasherDefault::default())
                .ok()
                .vortex_expect("HyperLogLogPlus precision constant is in [4, 18]");
            Self { inner }
        }

        /// Hash and absorb the item into the sketch.
        #[inline]
        pub(super) fn insert(&mut self, item: &T) {
            self.inner.insert(item);
        }

        /// Compute the current cardinality estimate.
        #[inline]
        pub(super) fn estimate(&mut self) -> usize {
            let count = self.inner.count();
            // `count` returns a non-negative `f64`; round to the nearest integer and clamp to
            // `usize` to match the 64-bit backend's return type.
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "HLL++ count is non-negative; truncation matches the 64-bit backend's usize semantics"
            )]
            let rounded = count.max(0.0).round() as usize;
            rounded
        }
    }
}
