// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::DecimalArray;
use crate::arrow::compute::to_arrow_decimal32;
use crate::arrow::compute::to_arrow_decimal64;
use crate::arrow::compute::to_arrow_decimal128;
use crate::arrow::compute::to_arrow_decimal256;

pub(super) fn to_arrow_decimal(
    array: ArrayRef,
    data_type: &DataType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    // Execute the array as a DecimalArray.
    let decimal_array = array.execute::<DecimalArray>(ctx)?;

    // Use the existing conversion functions from canonical.rs
    match data_type {
        DataType::Decimal32(..) => to_arrow_decimal32(decimal_array),
        DataType::Decimal64(..) => to_arrow_decimal64(decimal_array),
        DataType::Decimal128(..) => to_arrow_decimal128(decimal_array),
        DataType::Decimal256(..) => to_arrow_decimal256(decimal_array),
        _ => unreachable!("to_arrow_decimal called with non-decimal type"),
    }
}
