// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use vortex_dtype::NativeDecimalType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

/// A struct representing the precision and scale of a decimal type, to be represented
/// by the native type `D`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionScale<D> {
    precision: u8,
    scale: i8,
    phantom: PhantomData<D>,
}

impl<D: NativeDecimalType> PrecisionScale<D> {
    /// Create a new [`PrecisionScale`] with the given precision and scale.
    ///
    /// # Panics
    ///
    /// Panics if the precision/scale are invalid.
    pub fn new(precision: u8, scale: i8) -> Self {
        Self::try_new(precision, scale).vortex_expect("Failed to create `PrecisionScale`")
    }

    /// Try to create a new [`PrecisionScale`] with the given precision and scale.
    pub fn try_new(precision: u8, scale: i8) -> VortexResult<Self> {
        if precision == 0 {
            vortex_bail!(
                "precision cannot be 0, has to be between [1, {}]",
                D::MAX_PRECISION
            );
        }
        if precision > D::MAX_PRECISION {
            vortex_bail!(
                "Precision {} is greater than max {}",
                precision,
                D::MAX_PRECISION
            );
        }
        if scale > D::MAX_SCALE {
            vortex_bail!("Scale {} is greater than max {}", scale, D::MAX_SCALE);
        }
        if scale > 0 && scale as u8 > precision {
            vortex_bail!("Scale {} is greater than precision {}", scale, precision);
        }
        Ok(Self {
            precision,
            scale,
            phantom: Default::default(),
        })
    }

    /// Create a new [`PrecisionScale`] with the given precision and scale without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the precision and scale are valid.
    pub unsafe fn new_unchecked(precision: u8, scale: i8) -> Self {
        if cfg!(debug_assertions) {
            Self::new(precision, scale)
        } else {
            Self {
                precision,
                scale,
                phantom: Default::default(),
            }
        }
    }

    /// Validate that the given value fits within the precision and scale.
    pub fn validate(&self, value: &D) -> bool {
        let (int_digits, frac_digits) = D::count_digits(value);
        let allowed_int_digits = (self.precision as i8 - self.scale).max(0) as usize;
        let allowed_frac_digits = self.scale.max(0) as usize;

        int_digits <= allowed_int_digits && frac_digits <= allowed_frac_digits
    }
}
