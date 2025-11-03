// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::fmt::Display;
use std::marker::PhantomData;

use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::{DecimalDType, NativeDecimalType};

/// A struct representing the precision and scale of a decimal type, to be represented
/// by the native type `D`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionScale<D> {
    precision: u8,
    scale: i8,
    phantom: PhantomData<D>,
}

impl<D: NativeDecimalType> Display for PrecisionScale<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "decimal({}, p={}, s={})",
            type_name::<D>(),
            self.precision,
            self.scale
        )
    }
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

    /// The precision is the number of significant figures that the decimal tracks.
    #[inline(always)]
    pub fn precision(&self) -> u8 {
        self.precision
    }

    /// The scale is the maximum number of digits relative to the decimal point.
    #[inline(always)]
    pub fn scale(&self) -> i8 {
        self.scale
    }

    /// Validate whether a given value of type `D` fits within the precision and scale.
    #[inline]
    pub fn is_valid(&self, value: D) -> bool {
        self.precision <= D::MAX_PRECISION
            && value >= D::MIN_BY_PRECISION[self.precision as usize]
            && value <= D::MAX_BY_PRECISION[self.precision as usize]
    }
}

impl<D: NativeDecimalType> From<PrecisionScale<D>> for DecimalDType {
    fn from(value: PrecisionScale<D>) -> Self {
        DecimalDType {
            precision: value.precision,
            scale: value.scale,
        }
    }
}

impl<D: NativeDecimalType> TryFrom<&DecimalDType> for PrecisionScale<D> {
    type Error = vortex_error::VortexError;

    fn try_from(value: &DecimalDType) -> VortexResult<Self> {
        PrecisionScale::try_new(value.precision, value.scale)
    }
}

#[cfg(test)]
mod tests {
    use crate::PrecisionScale;

    #[test]
    fn max_precision() {
        let prec = PrecisionScale::<i8>::new(2, 1);
        assert!(prec.is_valid(8));
        assert!(prec.is_valid(99));
        assert!(prec.is_valid(-9));
        assert!(prec.is_valid(0));
        assert!(prec.is_valid(-99))
    }
}
