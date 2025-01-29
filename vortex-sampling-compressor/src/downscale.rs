use vortex_array::array::{PrimitiveArray, PrimitiveEncoding};
use vortex_array::compute::try_cast;
use vortex_array::stats::{ArrayStatistics, Stat};
use vortex_array::vtable::EncodingVTable;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_err, VortexExpect, VortexResult};

/// Downscale a primitive array to the narrowest PType that fits all the values.
pub fn downscale_integer_array(array: ArrayData) -> VortexResult<ArrayData> {
    if !array.is_encoding(PrimitiveEncoding.id()) {
        // This can happen if e.g. the array is ConstantArray.
        return Ok(array);
    }
    let array = PrimitiveArray::maybe_from(array).vortex_expect("Checked earlier");

    let min = array
        .statistics()
        .compute(Stat::Min)
        .ok_or_else(|| vortex_err!("Failed to compute min on primitive array"))?;
    let max = array
        .statistics()
        .compute(Stat::Max)
        .ok_or_else(|| vortex_err!("Failed to compute max on primitive array"))?;

    // If we can't cast to i64, then leave the array as its original type.
    // It's too big to downcast anyway.
    let Ok(min) = i64::try_from(&min) else {
        return Ok(array.into_array());
    };
    let Ok(max) = i64::try_from(&max) else {
        return Ok(array.into_array());
    };

    downscale_primitive_integer_array(array, min, max).map(|a| a.into_array())
}

/// Downscale a primitive array to the narrowest PType that fits all the values.
fn downscale_primitive_integer_array(
    array: PrimitiveArray,
    min: i64,
    max: i64,
) -> VortexResult<PrimitiveArray> {
    if min < 0 || max < 0 {
        // Signed
        if min >= i8::MIN as i64 && max <= i8::MAX as i64 {
            return try_cast(
                &array,
                &DType::Primitive(PType::I8, array.dtype().nullability()),
            )?
            .into_primitive();
        }

        if min >= i16::MIN as i64 && max <= i16::MAX as i64 {
            return try_cast(
                &array,
                &DType::Primitive(PType::I16, array.dtype().nullability()),
            )?
            .into_primitive();
        }

        if min >= i32::MIN as i64 && max <= i32::MAX as i64 {
            return try_cast(
                &array,
                &DType::Primitive(PType::I32, array.dtype().nullability()),
            )?
            .into_primitive();
        }
    } else {
        // Unsigned
        if max <= u8::MAX as i64 {
            return try_cast(
                &array,
                &DType::Primitive(PType::U8, array.dtype().nullability()),
            )?
            .into_primitive();
        }

        if max <= u16::MAX as i64 {
            return try_cast(
                &array,
                &DType::Primitive(PType::U16, array.dtype().nullability()),
            )?
            .into_primitive();
        }

        if max <= u32::MAX as i64 {
            return try_cast(
                &array,
                &DType::Primitive(PType::U32, array.dtype().nullability()),
            )?
            .into_primitive();
        }
    }

    Ok(array)
}
