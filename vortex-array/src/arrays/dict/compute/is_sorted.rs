// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::is_sorted::SortDirection;
use crate::aggregate_fn::fns::is_sorted::is_sorted_with_options;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Dict;
use crate::arrays::dict::DictArraySlotsExt;
use crate::scalar::Scalar;

/// Dict-specific is_sorted kernel.
///
/// | Values     | Codes     | Dict array sortedness |
/// |------------|-----------|-----------------------|
/// | Ascending  | Ascending | Ascending             |
/// | Descending | Ascending | Descending            |
///
/// Values and codes must be sorted with the same strictness.
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

        // Early exit for repeated invalids
        if options.strict && batch.invalid_count(ctx)? > 1 {
            return Ok(None);
        }

        let Some(dict) = batch.as_opt::<Dict>() else {
            return Ok(None);
        };

        let strict = options.strict;

        let sorted = match options.direction {
            SortDirection::Asc => {
                is_sorted_with_options(dict.values(), *options, ctx)?
                    && is_sorted_with_options(dict.codes(), *options, ctx)?
            }
            SortDirection::Desc => {
                // `DictEncoder` produce incremental asc codes, so a desc
                // column gets `DESC` values and `ASC` codes. So, using `DESC` codes,
                // e.g. from `take` or an imported dict, are out of scope.
                let asc_options = IsSortedOptions {
                    strict,
                    direction: SortDirection::Asc,
                };
                is_sorted_with_options(dict.values(), *options, ctx)?
                    && is_sorted_with_options(dict.codes(), asc_options, ctx)?
            }
        };

        if sorted {
            Ok(Some(IsSorted::make_partial(batch, true, strict, ctx)?))
        } else {
            // We can't definitively say it's NOT sorted without canonicalizing,
            // so return None to let the accumulator handle it.
            Ok(None)
        }
    }
}
