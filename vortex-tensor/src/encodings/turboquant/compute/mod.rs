// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compute pushdown implementations for TurboQuant.

pub(crate) mod cosine_similarity;

mod ops;
mod slice;
mod take;

pub(crate) mod rules;

use num_traits::Float;
use num_traits::FromPrimitive;
use vortex_error::VortexExpect;

/// Convert an f32 value to a float type `T`.
///
/// `FromPrimitive::from_f32` is infallible for all Vortex float types: f16 saturates via the
/// inherent `f16::from_f32()`, f32 is identity, f64 is lossless widening.
pub(crate) fn float_from_f32<T: Float + FromPrimitive>(v: f32) -> T {
    FromPrimitive::from_f32(v).vortex_expect("f32-to-float conversion is infallible")
}
