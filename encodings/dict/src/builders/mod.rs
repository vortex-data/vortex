use bytes::BytesDictBuilder;
use primitive::PrimitiveDictBuilder;
use vortex_array::arrays::{
    ConstantArray, PrimitiveArray, PrimitiveEncoding, VarBinArray, VarBinViewArray,
};
use vortex_array::compute::try_cast;
use vortex_array::stats::Stat;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{Array, ArrayExt, ArrayRef, ToCanonical};
use vortex_dtype::{DType, PType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::DictArray;

mod bytes;
mod primitive;

pub trait DictEncoder {
    fn encode(&mut self, array: &dyn Array) -> VortexResult<ArrayRef>;

    fn values(&mut self) -> VortexResult<ArrayRef>;
}

pub fn dict_encode_max_sized(array: &dyn Array, max_dict_bytes: usize) -> VortexResult<DictArray> {
    let dict_builder: &mut dyn DictEncoder = if let Some(pa) = array.as_opt::<PrimitiveArray>() {
        match_each_native_ptype!(pa.ptype(), |$P| {
            &mut PrimitiveDictBuilder::<$P>::new(pa.dtype().nullability(), max_dict_bytes)
        })
    } else if let Some(vbv) = array.as_opt::<VarBinViewArray>() {
        &mut BytesDictBuilder::new(vbv.dtype().clone(), max_dict_bytes)
    } else if let Some(vb) = array.as_opt::<VarBinArray>() {
        &mut BytesDictBuilder::new(vb.dtype().clone(), max_dict_bytes)
    } else {
        vortex_bail!("Can only encode primitive or varbin/view arrays")
    };
    let codes = downscale_integer_array(dict_builder.encode(array)?)?;

    DictArray::try_new(codes, dict_builder.values()?)
}

pub fn dict_encode(array: &dyn Array) -> VortexResult<DictArray> {
    let dict_array = dict_encode_max_sized(array, usize::MAX)?;
    if dict_array.len() != array.len() {
        vortex_bail!(
            "must have encoded all {} elements, but only encoded {}",
            array.len(),
            dict_array.len(),
        );
    }
    Ok(dict_array)
}

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

    let min = array.statistics().compute_stat(Stat::Min)?;
    let max = array.statistics().compute_stat(Stat::Max)?;

    let (Some(min), Some(max)) = (min, max) else {
        // This array but be all nulls.
        return Ok(
            ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
        );
    };

    // If we can't cast to i64, then leave the array as its original type.
    // It's too big to downcast anyway.
    let Ok(min) = i64::try_from(&min) else {
        return Ok(array.to_array());
    };
    let Ok(max) = i64::try_from(&max) else {
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
