// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{DecimalDType, DecimalType};

/// Maps a decimal precision into the smallest type that can represent it.
pub fn smallest_decimal_value_type(decimal_dtype: &DecimalDType) -> DecimalType {
    match decimal_dtype.precision() {
        1..=2 => DecimalType::I8,
        3..=4 => DecimalType::I16,
        5..=9 => DecimalType::I32,
        10..=18 => DecimalType::I64,
        19..=38 => DecimalType::I128,
        39..=76 => DecimalType::I256,
        0 => unreachable!("precision must be greater than 0"),
        p => unreachable!("precision larger than 76 is invalid found precision {p}"),
    }
}

/// True if `value_type` can represent every value of the type `dtype`.
pub fn is_compatible_decimal_value_type(value_type: DecimalType, dtype: DecimalDType) -> bool {
    value_type >= smallest_decimal_value_type(&dtype)
}
