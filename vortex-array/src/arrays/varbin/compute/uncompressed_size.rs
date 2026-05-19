// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::VarBin;
use crate::arrays::varbin::VarBinArrayExt;
use crate::arrays::varbinview::BinaryView;
use crate::scalar::Scalar;
use crate::validity::validity_uncompressed_size_in_bytes;

#[derive(Debug)]
pub(crate) struct VarBinUncompressedSizeInBytesKernel;

impl DynAggregateKernel for VarBinUncompressedSizeInBytesKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<UncompressedSizeInBytes>() {
            return Ok(None);
        }

        let Some(array) = batch.as_opt::<VarBin>() else {
            return Ok(None);
        };

        let first_offset = offset_at(array.offsets(), 0, ctx)?;
        let last_offset = offset_at(array.offsets(), array.len(), ctx)?;
        let data_size = u64::try_from(
            last_offset
                .checked_sub(first_offset)
                .ok_or_else(|| vortex_err!("VarBin offsets must be monotonically increasing"))?,
        )
        .map_err(|e| vortex_err!("Failed to convert VarBin data size to u64: {e}"))?;

        let views_size = u64::try_from(array.len())
            .map_err(|e| vortex_err!("array length does not fit in u64: {e}"))?
            .checked_mul(size_of::<BinaryView>() as u64)
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
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

fn offset_at(offsets: &ArrayRef, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
    offsets
        .execute_scalar(index, ctx)?
        .as_primitive()
        .as_::<usize>()
        .ok_or_else(|| vortex_err!("offset value does not fit in usize"))
}
