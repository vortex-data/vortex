// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use itertools::MinMaxResult;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::arrays::DecimalArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;

/// Return the array's unscaled values widened to `W`, which must be at least as wide as the
/// array's storage type.
pub(crate) fn widened_buffer<W: NativeDecimalType>(array: &DecimalArray) -> Buffer<W> {
    if array.values_type() == W::DECIMAL_TYPE {
        return array.buffer::<W>();
    }
    match_each_decimal_value_type!(array.values_type(), |T| {
        array
            .buffer::<T>()
            .iter()
            .map(|v| W::from(*v).vortex_expect("widening decimal cast must succeed"))
            .collect()
    })
}

/// Return the array's unscaled values converted to exactly `W`, whatever the array's
/// storage type. Zero-copy when the array is already stored at `W`.
///
/// Widening is lossless. Narrowing fails for any *valid* value that does not fit `W`.
/// Null slots may hold arbitrary bytes and never fail; their contents in the returned
/// buffer are likewise arbitrary and must not be read.
pub fn converted_buffer<W: NativeDecimalType>(
    array: &DecimalArray,
    validity: &Mask,
) -> VortexResult<Buffer<W>> {
    // Widening can never fail, so it needs no validation pass.
    if array.values_type() <= W::DECIMAL_TYPE {
        return Ok(widened_buffer(array));
    }
    match_each_decimal_value_type!(array.values_type(), |T| {
        let src = array.buffer::<T>();
        // Keeping the overflow scan branchless and vectorizable.
        // Only on overflow do we rescan with the mask: null-slot garbage is exempt, valid
        // out-of-range values are an error.
        let any_overflow = src.iter().fold(false, |acc, v| acc | W::from(*v).is_none());
        if any_overflow
            && let Some((i, v)) = src
                .iter()
                .enumerate()
                .find(|&(i, v)| W::from(*v).is_none() && validity.value(i))
        {
            vortex_bail!(
                "decimal value {v} at index {i} does not fit {}",
                W::DECIMAL_TYPE
            );
        }
        // The convert pass is infallible (every valid value fits; null-slot garbage becomes
        // zero), so a plain `collect` keeps the exact size hint and a single allocation.
        Ok(src
            .iter()
            .map(|v| W::from(*v).unwrap_or_default())
            .collect())
    })
}

macro_rules! try_downcast {
    ($array:expr, from: $src:ty, to: $($dst:ty),*) => {{
        use crate::dtype::BigCast;

        // Collect the min/max of the values
        let minmax = $array.buffer::<$src>().iter().copied().minmax();
        match minmax {
            MinMaxResult::NoElements => return $array,
            MinMaxResult::OneElement(_) => return $array,
            MinMaxResult::MinMax(min, max) => {
                $(
                    if <$dst as BigCast>::from(min).is_some() && <$dst as BigCast>::from(max).is_some() {
                        return DecimalArray::new::<$dst>(
                            $array
                                .buffer::<$src>()
                                .into_iter()
                                .map(|v| <$dst as BigCast>::from(v).vortex_expect("decimal conversion failure"))
                                .collect(),
                            $array.decimal_dtype(),
                            $array
                                .validity()
                                .vortex_expect("decimal validity should be derivable"),
                        );
                    }
                )*

                return $array;
            }
        }
    }};
}

/// Attempt to narrow the decimal array to any smaller supported type.
pub fn narrowed_decimal(decimal_array: DecimalArray) -> DecimalArray {
    match decimal_array.values_type() {
        // Cannot narrow any more
        DecimalType::I8 => decimal_array,
        DecimalType::I16 => {
            try_downcast!(decimal_array, from: i16, to: i8)
        }
        DecimalType::I32 => {
            try_downcast!(decimal_array, from: i32, to: i8, i16)
        }
        DecimalType::I64 => {
            try_downcast!(decimal_array, from: i64, to: i8, i16, i32)
        }
        DecimalType::I128 => {
            try_downcast!(decimal_array, from: i128, to: i8, i16, i32, i64)
        }
        DecimalType::I256 => {
            try_downcast!(decimal_array, from: i256, to: i8, i16, i32, i64, i128)
        }
    }
}
