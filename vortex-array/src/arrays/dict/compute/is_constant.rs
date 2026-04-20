// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_constant::is_constant;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Dict;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::scalar::Scalar;

/// Dict-specific is_constant kernel.
///
/// If codes are constant, the whole array is constant.
/// When all dictionary values are referenced, is_constant can be computed directly on the values
/// array. Otherwise, unreferenced values are filtered out first.
#[derive(Debug)]
pub(crate) struct DictIsConstantKernel;

impl DynAggregateKernel for DictIsConstantKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<IsConstant>() {
            return Ok(None);
        }

        let Some(dict) = batch.as_opt::<Dict>() else {
            return Ok(None);
        };

        // If codes are constant, only one dictionary value is referenced → constant.
        if is_constant(dict.codes(), ctx)? {
            return Ok(Some(IsConstant::make_partial(batch, true, ctx)?));
        }

        // Otherwise, check the values array. Filter to only referenced values if needed.
        let result = if dict.has_all_values_referenced() {
            is_constant(dict.values(), ctx)?
        } else {
            let referenced_mask = dict.compute_referenced_values_mask(true)?;
            let mask = Mask::from(referenced_mask);
            let filtered_values = dict.values().filter(mask)?;
            is_constant(&filtered_values, ctx)?
        };

        Ok(Some(IsConstant::make_partial(batch, result, ctx)?))
    }
}
