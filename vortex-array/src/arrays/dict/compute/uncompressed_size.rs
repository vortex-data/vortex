// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
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
        if !aggregate_fn.is::<UncompressedSizeInBytes>() {
            return Ok(None);
        }

        let Some(dict) = batch.as_opt::<Dict>() else {
            return Ok(None);
        };

        // We only want to use this kernel for variable length values
        if dict.dtype().element_size().is_some() {
            return Ok(None);
        }

        // array's aggregate kernel compute its decoded size.
        // For variable-width dictionaries, apply the codes to the values, then let the resulting
        let decoded = batch.clone().execute::<ArrayRef>(ctx)?;
        Ok(Some(Scalar::from(uncompressed_size_in_bytes_u64(
            &decoded, ctx,
        )?)))
    }
}
