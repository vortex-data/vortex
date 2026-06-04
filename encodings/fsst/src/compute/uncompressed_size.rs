// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::validity_uncompressed_size_in_bytes;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::dtype::IntegerPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

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

        let views_size = u64::try_from(array.len())
            .map_err(|e| vortex_err!("array length does not fit in u64: {e}"))?
            .checked_mul(size_of::<BinaryView>() as u64)
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
        let data_size = {
            let lengths: &PrimitiveArray = &array
                .uncompressed_lengths()
                .clone()
                .execute::<PrimitiveArray>(ctx)?;
            match_each_integer_ptype!(lengths.ptype(), |P| {
                uncompressed_lengths_size(lengths.as_slice::<P>())
            })
        }?;
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

fn uncompressed_lengths_size<P: IntegerPType>(lengths: &[P]) -> VortexResult<u64> {
    // The lengths child stores decoded byte counts for each logical value.
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
