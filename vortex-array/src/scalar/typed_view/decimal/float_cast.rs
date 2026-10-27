// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared decimal-to-float policy for scalar and array casts.

use crate::dtype::NativeDecimalType;

/// Convert the unscaled value to `f64`, then apply the scale with a single rounding step.
///
/// A non-negative scale divides by `10^scale` instead of multiplying by `10^-scale`. Up to scale
/// 22 the factor is exact, so `12345 / 100` gives the closest `f64` to `123.45`. A multiplication by
/// the inexact `0.01` can be one ulp off.
pub(crate) struct DecimalToFloatCast {
    factor: f64,
    scale_up: bool,
}

impl DecimalToFloatCast {
    pub(crate) fn new(scale: i8) -> Self {
        let scale = i32::from(scale);

        // The widest factor in the `i8` scale domain is `10^128`, well inside `f64` range.
        Self {
            factor: 10f64.powi(scale.abs()),
            scale_up: scale < 0,
        }
    }

    #[inline]
    pub(crate) fn cast<F: NativeDecimalType>(&self, value: F) -> f64 {
        // Every native decimal type implements `to_f64` for its whole range.
        let value = value.to_f64().unwrap_or(f64::NAN);

        if self.scale_up {
            value * self.factor
        } else {
            value / self.factor
        }
    }
}
