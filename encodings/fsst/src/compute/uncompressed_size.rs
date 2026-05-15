// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::dtype::IntegerPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::FSST;
use crate::FSSTArrayExt;

#[derive(Debug)]
pub(crate) struct FSSTUncompressedSizeInBytesKernel;

impl DynAggregateKernel for FSSTUncompressedSizeInBytesKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<UncompressedSizeInBytes>() {
            return Ok(None);
        }

        let Some(array) = batch.as_opt::<FSST>() else {
            return Ok(None);
        };

        let views_size = checked_len_mul(array.len(), size_of::<BinaryView>(), "binary view")?;
        let data_size = uncompressed_lengths_size(
            &array
                .uncompressed_lengths()
                .clone()
                .execute::<PrimitiveArray>(ctx)?,
        )?;
        let validity_size = validity_uncompressed_size_in_bytes(
            array
                .as_ref()
                .validity()?
                .execute_mask(array.as_ref().len(), ctx)?,
        )?;

        let size = views_size
            .checked_add(data_size)
            .and_then(|size| size.checked_add(validity_size))
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;

        Ok(Some(Scalar::from(size)))
    }
}

fn uncompressed_lengths_size(lengths: &PrimitiveArray) -> VortexResult<u64> {
    match_each_integer_ptype!(lengths.ptype(), |P| {
        uncompressed_lengths_size_typed(lengths.as_slice::<P>())
    })
}

fn uncompressed_lengths_size_typed<P: IntegerPType>(lengths: &[P]) -> VortexResult<u64> {
    let mut size = 0u64;
    for len in lengths {
        let len = len
            .to_u64()
            .ok_or_else(|| vortex_err!("uncompressed length cannot be negative"))?;
        size = size
            .checked_add(len)
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
    }
    Ok(size)
}

fn validity_uncompressed_size_in_bytes(validity: Mask) -> VortexResult<u64> {
    match validity {
        Mask::AllTrue(_) => Ok(0),
        Mask::AllFalse(len) => Ok(ConstantArray::new(false, len).into_array().nbytes()),
        Mask::Values(values) => u64::try_from(values.len())
            .map(|len| len.div_ceil(8))
            .map_err(|e| vortex_err!("Failed to convert bit buffer length to u64: {e}")),
    }
}

fn checked_len_mul(len: usize, width: usize, name: &str) -> VortexResult<u64> {
    let len = u64::try_from(len)
        .map_err(|e| vortex_err!("Failed to convert {name} length to u64: {e}"))?;
    let width = u64::try_from(width)
        .map_err(|e| vortex_err!("Failed to convert {name} byte width to u64: {e}"))?;

    len.checked_mul(width)
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))
}
