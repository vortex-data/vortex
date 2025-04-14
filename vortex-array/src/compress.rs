// TODO(ngates): make this a function on a PrimitiveArray
use vortex_dtype::{DType, PType};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, PrimitiveArray, PrimitiveEncoding};
use crate::compute::{min_max, try_cast};
use crate::vtable::EncodingVTable;
use crate::{Array, ArrayExt, ArrayRef, ToCanonical};

/// Downscale a primitive array to the narrowest PType that fits all the values.
pub fn downscale_integer_array(array: ArrayRef) -> VortexResult<ArrayRef> {
    if !array.is_encoding(PrimitiveEncoding.id()) {
        // This can happen if e.g. the array is ConstantArray.
        return Ok(array);
    }
    if array.is_empty() {
        return Ok(array);
    }
    let array = array
        .as_opt::<PrimitiveArray>()
        .vortex_expect("Checked earlier");

    let Some(min_max) = min_max(array)? else {
        // This array but be all nulls.
        return Ok(
            ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
        );
    };

    // If we can't cast to i64, then leave the array as its original type.
    // It's too big to downcast anyway.
    let Ok(min) = i64::try_from(min_max.min.value()) else {
        return Ok(array.to_array());
    };
    let Ok(max) = i64::try_from(min_max.max.value()) else {
        return Ok(array.to_array());
    };

    downscale_primitive_integer_array(array.clone(), min, max).map(|a| a.into_array())
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
            .to_primitive();
        }

        if min >= i16::MIN as i64 && max <= i16::MAX as i64 {
            return try_cast(
                &array,
                &DType::Primitive(PType::I16, array.dtype().nullability()),
            )?
            .to_primitive();
        }

        if min >= i32::MIN as i64 && max <= i32::MAX as i64 {
            return try_cast(
                &array,
                &DType::Primitive(PType::I32, array.dtype().nullability()),
            )?
            .to_primitive();
        }
    } else {
        // Unsigned
        if max <= u8::MAX as i64 {
            return try_cast(
                &array,
                &DType::Primitive(PType::U8, array.dtype().nullability()),
            )?
            .to_primitive();
        }

        if max <= u16::MAX as i64 {
            return try_cast(
                &array,
                &DType::Primitive(PType::U16, array.dtype().nullability()),
            )?
            .to_primitive();
        }

        if max <= u32::MAX as i64 {
            return try_cast(
                &array,
                &DType::Primitive(PType::U32, array.dtype().nullability()),
            )?
            .to_primitive();
        }
    }

    Ok(array)
}
