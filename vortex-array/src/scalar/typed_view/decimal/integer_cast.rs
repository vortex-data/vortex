// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared decimal-to-integer policy for scalar and array casts.

use num_traits::CheckedMul;
use vortex_error::VortexError;
use vortex_error::vortex_err;

use crate::dtype::BigCast;
use crate::dtype::IntegerPType;
use crate::dtype::i256;

/// Truncate toward zero, rejecting values outside the target range before truncation.
/// Integer arithmetic preserves exact values even beyond floating-point precision.
pub(crate) struct DecimalToIntegerCast<T> {
    scale: i8,
    factor: Option<i256>,
    minimum: T,
    maximum: T,
}

impl<T: IntegerPType + BigCast> DecimalToIntegerCast<T> {
    pub(crate) fn new(scale: i8) -> Self {
        Self {
            scale,
            factor: i256::from_i128(10).checked_pow(scale.unsigned_abs().into()),
            minimum: T::min_value(),
            maximum: T::max_value(),
        }
    }

    pub(crate) fn cast(&self, value: i256) -> Option<T> {
        if value == i256::ZERO {
            return Some(T::default());
        }
        // A negative scale can require a factor larger than i256. Any nonzero
        // value then exceeds every primitive integer's range.
        let factor = self.factor?;
        let integer = if self.scale > 0 {
            let integer = value / factor;
            if value % factor != i256::ZERO
                && ((value < i256::ZERO && integer == self.minimum.to_i256()?)
                    || (value > i256::ZERO && integer == self.maximum.to_i256()?))
            {
                return None;
            }
            integer
        } else {
            value.checked_mul(&factor)?
        };
        <T as BigCast>::from(integer)
    }

    pub(crate) fn error(&self, value: i256) -> VortexError {
        vortex_err!(
            "Decimal value {} at scale {} out of range for {}",
            value,
            self.scale,
            T::PTYPE
        )
    }
}
