// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::nan_count::NanCount;
use crate::aggregate_fn::fns::null_count::NullCount;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::sum_v2::SumV2;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Patched;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// The aggregate functions handled by [`PatchedConstantAggregate`].
pub(crate) fn patched_aggregate_kernels()
-> impl Iterator<Item = (AggregateFnId, &'static dyn DynAggregateKernel)> {
    [
        IsConstant.id(),
        MinMax.id(),
        NanCount.id(),
        NullCount.id(),
        Sum.id(),
        SumV2.id(),
    ]
    .into_iter()
    .map(|id| {
        (
            id,
            &PatchedConstantAggregate as &'static dyn DynAggregateKernel,
        )
    })
}

/// Aggregates a patched array over a constant inner array, such as a sparse array, in time
/// proportional to the number of patches.
///
/// The aggregates it handles do not depend on the order of their input, so the result is that of
/// the unpatched copies of the constant followed by the visible patch values.
#[derive(Debug)]
pub(crate) struct PatchedConstantAggregate;

impl DynAggregateKernel for PatchedConstantAggregate {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let order_independent = aggregate_fn.is::<IsConstant>()
            || aggregate_fn.is::<MinMax>()
            || aggregate_fn.is::<NanCount>()
            || aggregate_fn.is::<NullCount>()
            || aggregate_fn.is::<Sum>()
            || aggregate_fn.is::<SumV2>();
        if !order_independent {
            return Ok(None);
        }
        let Some(patched) = batch.as_opt::<Patched>() else {
            return Ok(None);
        };
        let Some(constant) = patched.inner().as_opt::<Constant>() else {
            return Ok(None);
        };

        let values = patched.patch_values();
        let executed = patched.execute_layout(ctx)?;
        let layout = executed.layout()?;
        // Usually every stored patch is visible. Otherwise, as after slicing, gather the
        // visible ones.
        let visible = if layout.all_patches().len() == values.len()
            && !layout.has_out_of_bounds_patches()
        {
            values.clone()
        } else {
            let mut sources = BufferMut::<u32>::with_capacity(layout.all_patches().len());
            #[expect(clippy::cast_possible_truncation, reason = "patch counts fit in u32")]
            layout.for_each(|_, patch| sources.push(patch as u32));
            values
                .take(PrimitiveArray::new(sources.freeze(), Validity::NonNullable).into_array())?
        };

        let mut accumulator = aggregate_fn.accumulator(batch.dtype())?;
        let unpatched = patched.len() - visible.len();
        if unpatched > 0 {
            let fill = ConstantArray::new(constant.scalar().clone(), unpatched).into_array();
            accumulator.accumulate(&fill, ctx)?;
        }
        if !visible.is_empty() {
            accumulator.accumulate(&visible, ctx)?;
        }
        Ok(Some(accumulator.partial_scalar()?))
    }
}
