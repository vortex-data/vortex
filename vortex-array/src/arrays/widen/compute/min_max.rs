// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::MinMaxResult;
use crate::aggregate_fn::fns::min_max::make_minmax_dtype;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::array::decode_scalar;
use crate::scalar::Scalar;

/// Computes [`MinMax`], [`Min`] and [`Max`] of a [`Widen`] from the extrema of its values.
///
/// Decoding preserves order, so the decoded extrema of the values are the extrema of the `Widen`.
/// Values without a valid row have no extrema and give the null partial, as the canonical array
/// does.
#[derive(Debug)]
pub(crate) struct WidenMinMaxKernel;

impl DynAggregateKernel for WidenMinMaxKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn
            .as_opt::<MinMax>()
            .or_else(|| aggregate_fn.as_opt::<Min>())
            .or_else(|| aggregate_fn.as_opt::<Max>())
            .copied()
        else {
            return Ok(None);
        };

        let Some(widen) = batch.as_opt::<Widen>() else {
            return Ok(None);
        };

        let dtype = batch.dtype();
        let extrema = match min_max(widen.values(), ctx, options)? {
            Some(MinMaxResult { min, max }) => Some(MinMaxResult {
                min: decode_scalar(&min, dtype)?,
                max: decode_scalar(&max, dtype)?,
            }),
            None => None,
        };

        if aggregate_fn.is::<MinMax>() {
            let partial_dtype = make_minmax_dtype(dtype);
            let partial = match extrema {
                Some(MinMaxResult { min, max }) => Scalar::struct_(partial_dtype, vec![min, max]),
                None => Scalar::null(partial_dtype),
            };
            return Ok(Some(partial));
        }

        let extremum = if aggregate_fn.is::<Min>() {
            extrema.map(|extrema| extrema.min)
        } else {
            extrema.map(|extrema| extrema.max)
        };
        let partial = match extremum {
            Some(value) => value.into_nullable(),
            None => Scalar::null(dtype.as_nullable()),
        };

        Ok(Some(partial))
    }
}
