// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex array implementing pco's `FloatMult` mode: an `f64` input stream is
//! decomposed into a `(primary, secondary)` pair of signed integers related
//! by a fixed `base: f64`.
//!
//! For every input `x`, encode produces:
//!
//! ```text
//! primary[i]    = round(x[i] / base) as i64
//! approx_bits   = (base * primary[i] as f64).to_bits() as i64
//! secondary[i]  = (x[i].to_bits() as i64).wrapping_sub(approx_bits)
//! ```
//!
//! and decode reconstructs:
//!
//! ```text
//! approx_bits = (base * primary[i] as f64).to_bits() as i64
//! out[i]      = f64::from_bits(approx_bits.wrapping_add(secondary[i]) as u64)
//! ```
//!
//! The round-trip is bit-exact for *every* `f64` input including NaN and
//! both infinities. For data with a natural decimal scale (currency at
//! `base = 0.01`, time at `base = 1.0`, …) `primary` has small magnitude
//! and `secondary` is tiny (often zero), both compressing well in the
//! entropy-coded children downstream.
//!
//! See [`FloatMultArray`] for the public type.

pub use array::*;

mod array;

/// Prost-encoded metadata for a [`FloatMultArray`].
///
/// `base` is the multiplier used to relate primary and secondary. It must be
/// a finite, strictly positive, non-subnormal `f64`.
#[derive(Clone, prost::Message)]
pub struct FloatMultMetadata {
    /// The multiplier used to decompose each `f64` value into
    /// `(primary, secondary)`.
    #[prost(double, tag = "1")]
    pub base: f64,
}
