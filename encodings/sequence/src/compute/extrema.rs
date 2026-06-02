// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::match_each_pvalue;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;

use crate::Sequence;
use crate::SequenceData;

/// Sequence-specific min/max kernel.
///
/// A sequence array represents `A[i] = base + i * multiplier`, so extrema can be computed
/// algebraically from `base` and `last` based on the sign of the multiplier.
#[derive(Debug)]
pub(crate) struct SequenceExtremaKernel;

impl DynAggregateKernel for SequenceExtremaKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let is_min = aggregate_fn.is::<Min>();
        let is_max = aggregate_fn.is::<Max>();
        if !is_min && !is_max {
            return Ok(None);
        }

        let Some(seq) = batch.as_opt::<Sequence>() else {
            return Ok(None);
        };

        let partial_dtype = batch.dtype().as_nullable();
        if seq.is_empty() {
            return Ok(Some(Scalar::null(partial_dtype)));
        }

        let base = seq.base();
        let last = SequenceData::try_last(base, seq.multiplier(), seq.ptype(), seq.len())?;
        let value = selected_extremum(is_min, base, last, seq.multiplier());

        Ok(Some(Scalar::try_new(
            partial_dtype,
            Some(ScalarValue::Primitive(value)),
        )?))
    }
}

fn selected_extremum(is_min: bool, base: PValue, last: PValue, multiplier: PValue) -> PValue {
    let (min_value, max_value) = match_each_pvalue!(
        multiplier,
        uint: |_v| { (base, last) },
        int: |v| {
            if v >= 0 {
                (base, last)
            } else {
                (last, base)
            }
        },
        float: |_v| { unreachable!("float multiplier not supported for SequenceArray") }
    );

    if is_min { min_value } else { max_value }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFn;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;

    use crate::Sequence;
    use crate::compute::extrema::SequenceExtremaKernel;

    fn kernel_extrema(array: &ArrayRef, is_min: bool) -> VortexResult<Option<i32>> {
        let aggregate_fn = if is_min {
            AggregateFn::new(Min, EmptyOptions).erased()
        } else {
            AggregateFn::new(Max, EmptyOptions).erased()
        };
        let scalar = SequenceExtremaKernel
            .aggregate(
                &aggregate_fn,
                array,
                &mut LEGACY_SESSION.create_execution_ctx(),
            )?
            .expect("sequence extrema kernel should handle sequence arrays");

        Option::<i32>::try_from(&scalar)
    }

    #[rstest]
    #[case::increasing(10, 3, 4, Some(10), Some(19))]
    #[case::decreasing(100, -10, 5, Some(60), Some(100))]
    fn sequence_extrema_kernel(
        #[case] base: i32,
        #[case] multiplier: i32,
        #[case] len: usize,
        #[case] expected_min: Option<i32>,
        #[case] expected_max: Option<i32>,
    ) -> VortexResult<()> {
        let array =
            Sequence::try_new_typed(base, multiplier, Nullability::NonNullable, len)?.into_array();

        assert_eq!(kernel_extrema(&array, true)?, expected_min);
        assert_eq!(kernel_extrema(&array, false)?, expected_max);
        Ok(())
    }
}
