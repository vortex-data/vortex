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
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::scalar::Scalar;

/// Dict-specific is_sorted kernel.
///
/// Cases handled:
/// - When the dictionary is tagged `sorted_values`, the array is sorted iff the codes are
///   sorted. For strict-sorted, the codes must also be strict-sorted (codes are unique within
///   a sort-position because each value appears once in the values array).
/// - Otherwise, if both values and codes are independently sorted at the requested strictness,
///   the array is sorted.
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

        // Fast path: sorted_values means the array is sorted iff codes are sorted.
        let result = if dict.has_sorted_values() {
            if strict {
                is_strict_sorted(dict.codes(), ctx)?
            } else {
                is_sorted(dict.codes(), ctx)?
            }
        } else if strict {
            is_strict_sorted(dict.values(), ctx)? && is_strict_sorted(dict.codes(), ctx)?
        } else {
            is_sorted(dict.values(), ctx)? && is_sorted(dict.codes(), ctx)?
        };

        if result {
            Ok(Some(IsSorted::make_partial(batch, true, strict, ctx)?))
        } else {
            // We can't definitively say it's NOT sorted without canonicalizing,
            // so return None to let the accumulator handle it.
            Ok(None)
        }
    }
}
