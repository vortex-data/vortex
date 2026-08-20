// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::EliasFano;

/// Elias-Fano-specific `is_sorted` kernel. Sortedness is a precondition of the encoding — the upper
/// array only decodes correctly for a non-decreasing sequence — so the answer needs no data.
///
/// Strict sortedness is a different question and is declined. Duplicates are legal — an empty list
/// contributes two identical offsets — and finding out whether any are present means comparing
/// adjacent low bits, which is a full scan. Returning `None` lets the generic path do that.
#[derive(Debug)]
pub(crate) struct EliasFanoIsSortedKernel;

impl DynAggregateKernel for EliasFanoIsSortedKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<IsSorted>() else {
            return Ok(None);
        };
        if options.strict || !batch.is::<EliasFano>() {
            return Ok(None);
        }
        Ok(Some(IsSorted::make_partial(
            batch,
            true,
            options.strict,
            ctx,
        )?))
    }
}
