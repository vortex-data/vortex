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
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::bounded_max::BoundedMax;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::aggregate_fn::fns::min_max::nan_scalar;
use crate::aggregate_fn::fns::min_max::scalar_is_nan;
use crate::dtype::DType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::expr::stats::StatsProviderExt;
use crate::partial_ord::partial_max;
use crate::scalar::Scalar;

/// Compute the maximum non-null value of an array.
///
/// NaN handling for float inputs is controlled by [`NumericalAggregateOpts`]: with `skip_nans` (the
/// default) NaN values are ignored, otherwise any NaN value poisons the maximum to NaN.
#[derive(Clone, Debug)]
pub struct Max;

/// Partial accumulator state for the maximum aggregate.
pub struct MaxPartial {
    max: Option<Scalar>,
    element_dtype: DType,
    skip_nans: bool,
}

impl MaxPartial {
    fn merge(&mut self, max: Scalar) {
        if max.is_null() {
            return;
        }

        // NaN scalars are incomparable under `partial_max`; they poison the maximum when NaNs
        // participate, and are dropped when they are skipped.
        if scalar_is_nan(&max) || self.is_poisoned() {
            if !self.skip_nans {
                self.poison();
            }
            return;
        }

        self.max = Some(match self.max.take() {
            Some(current) => partial_max(max, current).vortex_expect("incomparable max scalars"),
            None => max,
        });
    }

    fn poison(&mut self) {
        self.max = Some(nan_scalar(&self.element_dtype));
    }

    fn is_poisoned(&self) -> bool {
        self.max.as_ref().is_some_and(scalar_is_nan)
    }
}

impl AggregateFnVTable for Max {
    type Options = NumericalAggregateOpts;
    type Partial = MaxPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.max")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(options.serialize()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        NumericalAggregateOpts::deserialize(metadata)
    }

    fn return_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        MinMax
            .return_dtype(options, input_dtype)
            .map(|_| input_dtype.as_nullable())
    }

    fn can_satisfy(
        &self,
        options: &Self::Options,
        requested: &AggregateFnRef,
    ) -> AggregateFnSatisfaction {
        if requested
            .as_opt::<Self>()
            .is_some_and(|other| other == options)
        {
            AggregateFnSatisfaction::Exact
        } else if requested.is::<BoundedMax>() && options.skip_nans {
            // A NaN-including maximum may be NaN, which is not a usable upper bound.
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
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(MaxPartial {
            max: None,
            element_dtype: input_dtype.clone(),
            skip_nans: options.skip_nans,
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

    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        // A poisoned NaN-including maximum is fully determined.
        partial.is_poisoned()
    }

    fn try_accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        // NaN-aware shortcircuits only apply to the NaN-including float maximum; everything else
        // takes the default dispatch path.
        if partial.skip_nans || !partial.element_dtype.is_float() {
            return Ok(false);
        }
        match batch.statistics().get_as::<u64>(Stat::NaNCount) {
            Precision::Exact(0) => {
                // NaN-free batch: the cached NaN-skipping maximum (if any) is valid. `to_scalar`
                // re-casts to the result dtype, so the cached scalar can merge as-is.
                if let Some(max) = batch.statistics().get(Stat::Max).as_exact() {
                    partial.merge(max);
                    return Ok(true);
                }
                Ok(false)
            }
            Precision::Exact(_) => {
                partial.poison();
                Ok(true)
            }
            _ => Ok(false),
        }
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
        let options = NumericalAggregateOpts {
            skip_nans: partial.skip_nans,
        };
        if let Some(result) = min_max(&array, ctx, options)? {
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
    use crate::aggregate_fn::NumericalAggregateOpts;
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
        let mut acc = Accumulator::try_new(Max, NumericalAggregateOpts::default(), dtype)?;

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
        let mut acc = Accumulator::try_new(Max, NumericalAggregateOpts::default(), dtype)?;

        assert_eq!(
            acc.finish()?,
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable))
        );
        Ok(())
    }

    #[test]
    fn max_with_nan_not_skipping() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Max, NumericalAggregateOpts::include_nans(), dtype)?;

        let batch = PrimitiveArray::new(buffer![1.0f64, f64::NAN, -5.0], Validity::NonNullable)
            .into_array();
        acc.accumulate(&batch, &mut ctx)?;
        assert!(acc.is_saturated());

        let result = acc.finish()?;
        assert!(
            result
                .as_primitive()
                .typed_value::<f64>()
                .is_some_and(f64::is_nan)
        );
        Ok(())
    }

    #[test]
    fn max_not_skipping_shortcircuits_on_exact_nan_count_stat() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // The array has no NaNs; a planted exact NaNCount stat proves the poisoning came from
        // the stat rather than a scan.
        let batch = PrimitiveArray::new(buffer![1.0f64, 2.0], Validity::NonNullable).into_array();
        batch
            .statistics()
            .set(Stat::NaNCount, Precision::Exact(ScalarValue::from(1u64)));
        let mut acc = Accumulator::try_new(
            Max,
            NumericalAggregateOpts::include_nans(),
            batch.dtype().clone(),
        )?;
        acc.accumulate(&batch, &mut ctx)?;
        let result = acc.finish()?;
        assert!(
            result
                .as_primitive()
                .typed_value::<f64>()
                .is_some_and(f64::is_nan)
        );
        Ok(())
    }

    #[test]
    fn max_nan_including_nullable_cached_stat() -> VortexResult<()> {
        // A nullable float array's cached Max stat is reconstructed as a nullable scalar. The
        // NaN-including shortcircuit merges it as-is; `to_scalar` re-casts to the result dtype.
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array =
            PrimitiveArray::from_option_iter([Some(1.0f64), Some(2.0), Some(3.0)]).into_array();
        array
            .statistics()
            .set(Stat::NaNCount, Precision::Exact(ScalarValue::from(0u64)));
        array
            .statistics()
            .set(Stat::Max, Precision::Exact(ScalarValue::from(3.0f64)));
        let mut acc = Accumulator::try_new(
            Max,
            NumericalAggregateOpts::include_nans(),
            array.dtype().clone(),
        )?;
        acc.accumulate(&array, &mut ctx)?;
        assert_eq!(
            acc.finish()?,
            Scalar::primitive(3.0f64, Nullability::Nullable)
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
        let mut acc = Accumulator::try_new(
            Max,
            NumericalAggregateOpts::default(),
            batch.dtype().clone(),
        )?;

        acc.accumulate(&batch, &mut ctx)?;

        assert_eq!(
            acc.finish()?,
            Scalar::primitive(25i32, Nullability::Nullable)
        );
        Ok(())
    }
}
