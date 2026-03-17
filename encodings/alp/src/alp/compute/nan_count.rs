// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::nan_count::nan_count;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::ALP;

/// ALP-specific NaN count kernel.
///
/// NaN values can only appear in the patches array of an ALP-encoded array, since the encoded
/// integer values cannot represent NaN. This avoids decoding the entire array.
#[derive(Debug)]
pub(crate) struct ALPNanCountKernel;

impl DynAggregateKernel for ALPNanCountKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<NanCount>() {
            return Ok(None);
        }

        let Some(alp) = batch.as_opt::<ALP>() else {
            return Ok(None);
        };

        let count = if let Some(patches) = alp.patches() {
            nan_count(patches.values(), ctx)?
        } else {
            0
        };

        Ok(Some(Scalar::from(count as u64)))
    }
}
