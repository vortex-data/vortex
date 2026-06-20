// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnSatisfaction;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::fns::bounded_max::BoundedMax;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::dtype::DType;
use crate::partial_ord::partial_max;
use crate::scalar::Scalar;

/// Compute the maximum non-null value of an array.
#[derive(Clone, Debug)]
pub struct Max;

/// Partial accumulator state for the maximum aggregate.
pub struct MaxPartial {
    max: Option<Scalar>,
    element_dtype: DType,
}

impl MaxPartial {
    fn merge(&mut self, max: Scalar) {
        if max.is_null() {
            return;
        }

        self.max = Some(match self.max.take() {
            Some(current) => partial_max(max, current).vortex_expect("incomparable max scalars"),
            None => max,
        });
    }
}

impl AggregateFnVTable for Max {
    type Options = EmptyOptions;
    type Partial = MaxPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.max")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        MinMax
            .return_dtype(&EmptyOptions, input_dtype)
            .map(|_| input_dtype.as_nullable())
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
            max: None,
            element_dtype: input_dtype.clone(),
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        partial.merge(other);
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let dtype = partial.element_dtype.as_nullable();
        match &partial.max {
            Some(max) => max.cast(&dtype),
            None => Ok(Scalar::null(dtype)),
        }
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.max = None;
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
        // Delegate to the existing min_max implementation for now. A dedicated max aggregate
        // would avoid computing min when only max is needed.
        let array = match batch {
            Columnar::Canonical(canonical) => canonical.clone().into_array(),
            Columnar::Constant(constant) => constant.clone().into_array(),
        };
        if let Some(result) = min_max(&array, ctx)? {
            partial.merge(result.max);
        }
        Ok(())
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
}
