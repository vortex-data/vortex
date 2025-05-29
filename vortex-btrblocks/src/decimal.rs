use itertools::{Itertools, MinMaxResult};
use vortex_array::ArrayRef;
use vortex_array::arrays::{DecimalArray, PrimitiveArray};
use vortex_array::vtable::ValidityHelper;
use vortex_decimal_byte_parts::DecimalBytePartsArray;
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::{BigCast, DecimalValueType, i256};

use crate::{Compressor, IntCompressor, MAX_CASCADE};

// TODO(joe): add support splitting i128/256 buffers into chunks primitive values for compression.
// 2 for i128 and 4 for i256
pub fn compress_decimal(decimal: &DecimalArray) -> VortexResult<ArrayRef> {
    let decimal = narrowed_decimal(decimal.clone());
    let validity = decimal.validity();
    let prim = match decimal.values_type() {
        DecimalValueType::I8 => PrimitiveArray::new(decimal.buffer::<i8>(), validity.clone()),
        DecimalValueType::I16 => PrimitiveArray::new(decimal.buffer::<i16>(), validity.clone()),
        DecimalValueType::I32 => PrimitiveArray::new(decimal.buffer::<i32>(), validity.clone()),
        DecimalValueType::I64 => PrimitiveArray::new(decimal.buffer::<i64>(), validity.clone()),
        _ => return Ok(decimal.to_array()),
    };

    let compressed = IntCompressor::compress(&prim, false, MAX_CASCADE, &[])?;

    DecimalBytePartsArray::try_new(compressed, vec![], decimal.decimal_dtype())
        .map(|d| d.to_array())
}

macro_rules! try_downcast {
    ($array:expr, from: $src:ty, to: $($dst:ty),*) => {{
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
                            $array.validity().clone(),
                        );
                    }
                )*

                return $array;
            }
        }
    }};
}

/// Attempt to narrow the decimal array to any smaller supported type.
fn narrowed_decimal(decimal_array: DecimalArray) -> DecimalArray {
    match decimal_array.values_type() {
        // Cannot narrow any more
        DecimalValueType::I8 => decimal_array,
        DecimalValueType::I16 => {
            try_downcast!(decimal_array, from: i16, to: i8)
        }
        DecimalValueType::I32 => {
            try_downcast!(decimal_array, from: i32, to: i8, i16)
        }
        DecimalValueType::I64 => {
            try_downcast!(decimal_array, from: i64, to: i8, i16, i32)
        }
        DecimalValueType::I128 => {
            try_downcast!(decimal_array, from: i128, to: i8, i16, i32, i64)
        }
        DecimalValueType::I256 => {
            try_downcast!(decimal_array, from: i256, to: i8, i16, i32, i64, i128)
        }
        _ => decimal_array,
    }
}
