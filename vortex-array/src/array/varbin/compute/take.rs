use arrow_buffer::NullBuffer;
use num_traits::PrimInt;
use vortex_dtype::{match_each_integer_ptype, DType, NativePType};
use vortex_error::{vortex_err, vortex_panic, VortexResult};

use crate::array::varbin::builder::VarBinBuilder;
use crate::array::varbin::VarBinArray;
use crate::array::VarBinEncoding;
use crate::compute::TakeFn;
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, IntoArray, IntoArrayVariant};

impl TakeFn<VarBinArray> for VarBinEncoding {
    fn take(&self, array: &VarBinArray, indices: &Array) -> VortexResult<Array> {
        let offsets = array.offsets().into_primitive()?;
        let data = array.bytes();
        let indices = indices.clone().into_primitive()?;
        match_each_integer_ptype!(offsets.ptype(), |$O| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                Ok(take(
                    array.dtype().clone(),
                    offsets.as_slice::<$O>(),
                    data.as_slice(),
                    indices.as_slice::<$I>(),
                    array.validity(),
                )?.into_array())
            })
        })
    }
}

fn take<I: NativePType, O: NativePType + PrimInt>(
    dtype: DType,
    offsets: &[O],
    data: &[u8],
    indices: &[I],
    validity: Validity,
) -> VortexResult<VarBinArray> {
    let validity_mask = validity.to_logical(offsets.len() - 1)?;
    if let Some(v) = validity_mask.to_null_buffer() {
        return Ok(take_nullable(dtype, offsets, data, indices, v));
    }

    let mut builder = VarBinBuilder::<O>::with_capacity(indices.len());
    for &idx in indices {
        let idx = idx
            .to_usize()
            .ok_or_else(|| vortex_err!("Failed to convert index to usize: {}", idx))?;
        let start = offsets[idx]
            .to_usize()
            .ok_or_else(|| vortex_err!("Failed to convert offset to usize: {}", offsets[idx]))?;
        let stop = offsets[idx + 1].to_usize().ok_or_else(|| {
            vortex_err!("Failed to convert offset to usize: {}", offsets[idx + 1])
        })?;
        builder.append_value(&data[start..stop]);
    }
    Ok(builder.finish(dtype))
}

fn take_nullable<I: NativePType, O: NativePType + PrimInt>(
    dtype: DType,
    offsets: &[O],
    data: &[u8],
    indices: &[I],
    null_buffer: NullBuffer,
) -> VarBinArray {
    let mut builder = VarBinBuilder::<O>::with_capacity(indices.len());
    for &idx in indices {
        let idx = idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", idx));
        if null_buffer.is_valid(idx) {
            let start = offsets[idx].to_usize().unwrap_or_else(|| {
                vortex_panic!("Failed to convert offset to usize: {}", offsets[idx])
            });
            let stop = offsets[idx + 1].to_usize().unwrap_or_else(|| {
                vortex_panic!("Failed to convert offset to usize: {}", offsets[idx + 1])
            });
            builder.append_value(&data[start..stop]);
        } else {
            builder.append_null();
        }
    }
    builder.finish(dtype)
}
