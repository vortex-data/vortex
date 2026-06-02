// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnSatisfaction;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::fns::bounded_max::BoundedMax;
use crate::aggregate_fn::fns::extrema::Extremum;
use crate::aggregate_fn::fns::extrema::ExtremumPartial;
use crate::aggregate_fn::fns::extrema::accumulate_extremum;
use crate::aggregate_fn::fns::extrema::compute_extremum;
use crate::aggregate_fn::fns::extrema::extrema_return_dtype;
use crate::dtype::DType;
use crate::scalar::Scalar;

/// Compute the maximum non-null value of an array.
#[derive(Clone, Debug)]
pub struct Max;

/// Partial accumulator state for the maximum aggregate.
///
/// The shared extrema state tracks both the current maximum and whether a runtime comparison was
/// unordered. An unordered partial finalizes to a typed null, meaning the statistic is unknown.
pub struct MaxPartial {
    inner: ExtremumPartial,
}

/// Compute the maximum non-null value of an array, or `None` if the statistic is unknown.
///
/// Null values are ignored. Top-level primitive NaNs are ignored consistently with the aggregate
/// semantics used by pruning; nested unordered comparisons make the result unknown.
pub fn max(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<Scalar>> {
    compute_extremum(Extremum::Max, Max, array, ctx)
}

impl AggregateFnVTable for Max {
    type Options = EmptyOptions;
    type Partial = MaxPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.max")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        extrema_return_dtype(input_dtype)
    }

    fn can_satisfy(
        &self,
        _options: &Self::Options,
        requested: &AggregateFnRef,
    ) -> AggregateFnSatisfaction {
        if requested.is::<Self>() {
            AggregateFnSatisfaction::Exact
        } else if requested.is::<BoundedMax>() {
            AggregateFnSatisfaction::Approximate
        } else {
            AggregateFnSatisfaction::No
        }
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(MaxPartial {
            inner: ExtremumPartial::new(input_dtype.as_nullable()),
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        partial.inner.merge_scalar(Extremum::Max, other)
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        partial.inner.to_scalar()
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.inner.reset();
    }

    fn is_saturated(&self, _partial: &Self::Partial) -> bool {
        false
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        accumulate_extremum(Extremum::Max, &mut partial.inner, batch, ctx)
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray as _;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::max::Max;
    use crate::aggregate_fn::fns::max::max;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    #[test]
    fn max_aggregate_fn() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Max, EmptyOptions, dtype)?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20, 5], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;

        let batch2 = PrimitiveArray::new(buffer![3i32, 25], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;

        assert_eq!(
            acc.finish()?,
            Scalar::primitive(25i32, Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn max_empty_group_returns_null() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Max, EmptyOptions, dtype)?;

        assert_eq!(
            acc.finish()?,
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable))
        );
        Ok(())
    }

    #[test]
    fn max_casts_nonnullable_legacy_stat_to_nullable_partial() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let batch = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        batch
            .statistics()
            .set(Stat::Max, Precision::Exact(ScalarValue::from(25i32)));
        let mut acc = Accumulator::try_new(Max, EmptyOptions, batch.dtype().clone())?;

        acc.accumulate(&batch, &mut ctx)?;

        assert_eq!(
            acc.finish()?,
            Scalar::primitive(25i32, Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn max_fixed_size_list_uses_element_order() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let elements = buffer![2i32, 1, 3].into_array();
        let array = FixedSizeListArray::new(elements, 1, Validity::NonNullable, 3).into_array();
        let expected = array.execute_scalar(2, &mut ctx)?;

        assert_eq!(max(&array, &mut ctx)?, Some(expected));
        Ok(())
    }

    #[test]
    fn max_aggregate_accepts_fixed_size_list() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let elements = buffer![2i32, 1, 3].into_array();
        let array = FixedSizeListArray::new(elements, 1, Validity::NonNullable, 3).into_array();
        let expected = array.execute_scalar(2, &mut ctx)?;
        let mut acc = Accumulator::try_new(Max, EmptyOptions, array.dtype().clone())?;

        acc.accumulate(&array, &mut ctx)?;

        assert_eq!(acc.finish()?, expected.cast(&array.dtype().as_nullable())?);
        Ok(())
    }
}
