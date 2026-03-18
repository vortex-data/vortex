// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::make_struct_dtype;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Dict;
use crate::scalar::Scalar;

/// Dict-specific min/max kernel.
///
/// When all dictionary values are referenced, min/max can be computed directly on the values
/// array. Otherwise, unreferenced values are filtered out first.
#[derive(Debug)]
pub(crate) struct DictMinMaxKernel;

impl DynAggregateKernel for DictMinMaxKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<MinMax>() {
            return Ok(None);
        }

        let Some(dict) = batch.as_opt::<Dict>() else {
            return Ok(None);
        };

        let struct_dtype = make_struct_dtype(batch.dtype());

        let result = if dict.has_all_values_referenced() {
            // All values are referenced, compute min/max directly on the values array.
            min_max(dict.values(), ctx)?
        } else {
            // Filter to only referenced values, then compute min/max.
            let referenced_mask = dict.compute_referenced_values_mask(true)?;
            let mask = Mask::from(referenced_mask);
            let filtered_values = dict.values().filter(mask)?;
            min_max(&filtered_values, ctx)?
        };

        match result {
            Some(r) => Ok(Some(Scalar::struct_(struct_dtype, vec![r.min, r.max]))),
            None => Ok(Some(Scalar::null(struct_dtype))),
        }
    }
}
