// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::arrays::{DecimalArray, PrimitiveArray, narrowed_decimal};
use vortex_array::vtable::ValidityHelper;
use vortex_decimal_byte_parts::DecimalBytePartsArray;
use vortex_error::VortexResult;
use vortex_scalar::DecimalType;

use crate::{Compressor, IntCompressor, MAX_CASCADE};

// TODO(joe): add support splitting i128/256 buffers into chunks primitive values for compression.
// 2 for i128 and 4 for i256
pub fn compress_decimal(decimal: &DecimalArray) -> VortexResult<ArrayRef> {
    let decimal = narrowed_decimal(decimal.clone());
    let validity = decimal.validity();
    let prim = match decimal.values_type() {
        DecimalType::I8 => PrimitiveArray::new(decimal.buffer::<i8>(), validity.clone()),
        DecimalType::I16 => PrimitiveArray::new(decimal.buffer::<i16>(), validity.clone()),
        DecimalType::I32 => PrimitiveArray::new(decimal.buffer::<i32>(), validity.clone()),
        DecimalType::I64 => PrimitiveArray::new(decimal.buffer::<i64>(), validity.clone()),
        _ => return Ok(decimal.to_array()),
    };

    let compressed = IntCompressor::compress(&prim, false, MAX_CASCADE, &[])?;

    DecimalBytePartsArray::try_new(compressed, decimal.decimal_dtype()).map(|d| d.to_array())
}
