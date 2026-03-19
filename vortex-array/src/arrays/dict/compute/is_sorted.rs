// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::is_sorted;
use crate::aggregate_fn::fns::is_sorted::is_strict_sorted;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Dict;
use crate::scalar::Scalar;

/// Dict-specific is_sorted kernel.
///
/// If both values and codes are sorted (with the same strictness), then the dict array is sorted.
#[derive(Debug)]
pub(crate) struct DictIsSortedKernel;

impl DynAggregateKernel for DictIsSortedKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<IsSorted>() else {
            return Ok(None);
        };

        let Some(dict) = batch.as_opt::<Dict>() else {
            return Ok(None);
        };

        let strict = options.strict;

        let result = if strict {
            is_strict_sorted(dict.values(), ctx)? && is_strict_sorted(dict.codes(), ctx)?
        } else {
            is_sorted(dict.values(), ctx)? && is_sorted(dict.codes(), ctx)?
        };

        if result {
            Ok(Some(IsSorted::make_partial(batch, true, strict)?))
        } else {
            // We can't definitively say it's NOT sorted without canonicalizing,
            // so return None to let the accumulator handle it.
            Ok(None)
        }
    }
}
