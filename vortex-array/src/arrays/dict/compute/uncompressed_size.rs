// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::FixedWidthUncompressedSizeInBytesKernel;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes_u64;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Dict;
use crate::scalar::Scalar;

#[derive(Debug)]
pub(crate) struct DictUncompressedSizeInBytesKernel;

impl DynAggregateKernel for DictUncompressedSizeInBytesKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<UncompressedSizeInBytes>() || !batch.is::<Dict>() {
            return Ok(None);
        }

        // Fixed-width decoded size only needs the logical width, row count, and derived validity.
        if let Some(size) =
            FixedWidthUncompressedSizeInBytesKernel.aggregate(aggregate_fn, batch, ctx)?
        {
            return Ok(Some(size));
        }

        // For variable-width dictionaries, apply the codes to the values, then let the resulting
        // array's aggregate kernel compute its decoded size.
        let decoded = batch.clone().execute::<ArrayRef>(ctx)?;
        Ok(Some(Scalar::from(uncompressed_size_in_bytes_u64(
            &decoded, ctx,
        )?)))
    }
}
