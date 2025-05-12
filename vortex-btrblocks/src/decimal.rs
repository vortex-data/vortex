use vortex_array::arrays::{DecimalArray, PrimitiveArray};
use vortex_array::{Array, ArrayRef};
use vortex_decimal_byte_parts::DecimalBytePartsArray;
use vortex_error::VortexResult;
use vortex_scalar::DecimalValueType;

use crate::{Compressor, IntCompressor, MAX_CASCADE};

// TODO(joe): add decimal value type narrowing
// TODO(joe): add support splitting i128/256 buffers into chunks primitive values for compression.
// 2 for i128 and 4 for i256
pub fn compress_decimal(decimal: &DecimalArray) -> VortexResult<ArrayRef> {
    let validity = decimal.validity();
    let prim = match decimal.values_type() {
        DecimalValueType::I8 => PrimitiveArray::new(decimal.buffer::<i8>(), validity.clone()),
        DecimalValueType::I16 => PrimitiveArray::new(decimal.buffer::<i16>(), validity.clone()),
        DecimalValueType::I32 => PrimitiveArray::new(decimal.buffer::<i32>(), validity.clone()),
        DecimalValueType::I64 => PrimitiveArray::new(decimal.buffer::<i64>(), validity.clone()),
        _ => return Ok(decimal.clone().into_array()),
    };

    let compressed = IntCompressor::compress(&prim, false, MAX_CASCADE, &[])?;

    DecimalBytePartsArray::try_new(compressed, vec![], decimal.decimal_dtype())
        .map(|d| d.to_array())
}
