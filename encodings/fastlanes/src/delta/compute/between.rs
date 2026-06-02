// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::between::BetweenKernel;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_error::VortexResult;

use crate::Delta;
use crate::delta::compute::sorted::bool_range;
use crate::delta::compute::sorted::is_known_sorted;
use crate::delta::compute::sorted::lower_bound;
use crate::delta::compute::sorted::upper_bound;

impl BetweenKernel for Delta {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only a known-sorted array collapses to a contiguous range; otherwise fall back to
        // the default decode-then-compare path. Null values would make the binary-search
        // ordering ill-defined, so require non-nullable values (null bounds are handled by
        // the between precondition before we get here).
        if !is_known_sorted(array) || array.dtype().is_nullable() {
            return Ok(None);
        }

        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };

        let nullability =
            array.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();
        let arr = array.array().clone();
        let len = arr.len();

        match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
            let (Some(lo), Some(hi)) = (lower.as_primitive().pvalue(), upper.as_primitive().pvalue())
            else {
                return Ok(None);
            };
            let (Ok(lo), Ok(hi)) = (lo.cast::<T>(), hi.cast::<T>()) else {
                // The bound does not fit in the array's domain; let the generic path decide.
                return Ok(None);
            };

            // start: first index satisfying the lower bound (`lo < x` strict, `lo <= x` not).
            let start = match options.lower_strict {
                StrictComparison::Strict => upper_bound::<T>(&arr, len, lo, ctx)?,
                StrictComparison::NonStrict => lower_bound::<T>(&arr, len, lo, ctx)?,
            };
            // end: first index violating the upper bound (`x < hi` strict, `x <= hi` not).
            let end = match options.upper_strict {
                StrictComparison::Strict => lower_bound::<T>(&arr, len, hi, ctx)?,
                StrictComparison::NonStrict => upper_bound::<T>(&arr, len, hi, ctx)?,
            };

            Ok(Some(bool_range(len, start, end, false, nullability)))
        })
    }
}
