// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Chunked;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::scalar::Scalar;

#[derive(Debug)]
pub struct ChunkedArrayAggregate;

impl DynAggregateKernel for ChunkedArrayAggregate {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(chunked) = batch.as_opt::<Chunked>() else {
            return Ok(None);
        };

        let mut acc = aggregate_fn.accumulator(chunked.dtype())?;
        // Skip empty chunks: they contribute no elements.
        for chunk in chunked.non_empty_chunks() {
            acc.accumulate(chunk, ctx)?;
        }
        // Return the partial (not finalized) result, since the outer accumulator
        // will merge this partial scalar into its own state.
        Ok(Some(acc.flush()?))
    }
}
