// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Storage dispatch and logical partials for aggregates over extension values.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::dtype::Claim;
use crate::dtype::DType;
use crate::scalar::Scalar;

/// The storage dtype of an extension input that holds `claim`.
pub(crate) fn claimed_storage_dtype(input: &DType, claim: Claim) -> Option<&DType> {
    let ext = input.as_extension_opt()?;
    input.holds(claim).then(|| ext.storage_dtype())
}

/// The innermost storage dtype, which the aggregate uses for its in-memory state.
pub(super) fn storage_dtype(mut dtype: &DType) -> &DType {
    while let Some(ext) = dtype.as_extension_opt() {
        dtype = ext.storage_dtype();
    }
    dtype
}

/// Remove extension wrappers when a value enters an aggregate's partial state.
pub(super) fn into_storage_scalar(mut scalar: Scalar) -> Scalar {
    while let Some(ext) = scalar.as_extension_opt() {
        scalar = ext.to_storage_scalar();
    }
    scalar
}

/// Restore the logical dtype of a value selected by the aggregate.
pub(super) fn wrap_storage_scalar(value: &Scalar, dtype: &DType) -> VortexResult<Scalar> {
    if !dtype.is_extension() {
        return value.cast(dtype);
    }

    let storage = storage_dtype(dtype);
    vortex_ensure_eq!(
        storage,
        &value.dtype().with_nullability(storage.nullability())
    );
    Scalar::try_new(dtype.clone(), value.value().cloned())
}

/// Run an aggregate on the storage encoding and fold its in-memory state into the logical call.
pub(super) fn accumulate_storage<A: AggregateFnVTable>(
    aggregate: &A,
    args: AggregateArgs<'_, A::Options>,
    partial: &mut A::Partial,
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let mut accumulator = Accumulator::try_new(
        aggregate.clone(),
        args.options.clone(),
        storage.dtype().clone(),
    )?;
    accumulator.accumulate(storage, ctx)?;
    if let Some(storage_partial) = accumulator.take_partial() {
        let running = std::mem::replace(partial, aggregate.empty_partial(args)?);
        *partial = aggregate.merge_partials(args, running, storage_partial)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use rstest::rstest;
    use vortex_session::VortexSession;

    use super::*;
    use crate::Canonical;
    use crate::Columnar;
    use crate::EmptyMetadata;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::all_non_distinct::AllNonDistinct;
    use crate::aggregate_fn::fns::bounded_max::BoundedMax;
    use crate::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
    use crate::aggregate_fn::fns::bounded_min::BoundedMin;
    use crate::aggregate_fn::fns::bounded_min::BoundedMinOptions;
    use crate::aggregate_fn::fns::is_constant::IsConstant;
    use crate::aggregate_fn::fns::is_sorted::IsSorted;
    use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use crate::aggregate_fn::fns::max::Max;
    use crate::aggregate_fn::fns::min::Min;
    use crate::aggregate_fn::fns::min_max::MinMax;
    use crate::arrays::ConstantArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::dtype::Nullability;
    use crate::extension::tests::Opaque;

    #[rstest]
    #[case::is_constant(IsConstant, EmptyOptions, "is_constant", "Eq")]
    #[case::is_sorted(IsSorted, IsSortedOptions { strict: false }, "is_sorted", "Ord")]
    #[case::min_max(MinMax, NumericalAggregateOpts::default(), "min_max", "Ord")]
    fn direct_aggregate_accumulation_checks_claims<A: AggregateFnVTable>(
        #[case] aggregate: A,
        #[case] options: A::Options,
        #[case] name: &str,
        #[case] claim: &str,
    ) -> VortexResult<()> {
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let scalar = Scalar::extension::<Opaque>(EmptyMetadata, 1i64.into());
        let dtype = scalar.dtype();
        let canonical = ExtensionArray::new(
            dtype.as_extension().clone(),
            PrimitiveArray::from_iter([1i64]).into_array(),
        );
        let args = AggregateArgs {
            options: &options,
            dtype,
            return_dtype: dtype,
            partial_dtype: dtype,
        };
        let mut partial = aggregate.empty_partial(args)?;

        assert!(!aggregate.try_accumulate(
            args,
            &mut partial,
            &canonical.clone().into_array(),
            &mut ctx,
        )?);

        let expected = format!("{name} requires the {claim} claim, but {dtype} does not hold it");
        for batch in [
            Columnar::Canonical(Canonical::Extension(canonical)),
            Columnar::Constant(ConstantArray::new(scalar.clone(), 1)),
        ] {
            let error = aggregate
                .accumulate(args, &mut partial, &batch, &mut ctx)
                .expect_err("direct accumulation requires the storage claim");
            assert_eq!(error.to_string().lines().next(), Some(expected.as_str()));
        }
        Ok(())
    }

    #[rstest]
    #[case::is_constant(IsConstant, EmptyOptions, "is_constant", "Eq")]
    #[case::is_sorted(IsSorted, IsSortedOptions { strict: false }, "is_sorted", "Ord")]
    #[case::min_max(MinMax, NumericalAggregateOpts::default(), "min_max", "Ord")]
    #[case::min(Min, NumericalAggregateOpts::default(), "min", "Ord")]
    #[case::max(Max, NumericalAggregateOpts::default(), "max", "Ord")]
    #[case::bounded_min(BoundedMin, BoundedMinOptions { max_bytes: NonZeroUsize::MIN }, "bounded_min", "Ord")]
    #[case::bounded_max(BoundedMax, BoundedMaxOptions { max_bytes: NonZeroUsize::MIN }, "bounded_max", "Ord")]
    fn direct_aggregate_partial_decoding_checks_claims<A: AggregateFnVTable>(
        #[case] aggregate: A,
        #[case] options: A::Options,
        #[case] name: &str,
        #[case] claim: &str,
    ) {
        let scalar = Scalar::extension::<Opaque>(EmptyMetadata, 1i64.into());
        let dtype = scalar.dtype();
        // Use the aggregate's normal partial shape with an opaque input substituted for storage.
        let storage = dtype.as_extension().storage_dtype();
        let partial_dtype = aggregate
            .partial_dtype(&options, storage)
            .expect("the storage dtype supports the aggregate")
            .as_nullable();
        let args = AggregateArgs {
            options: &options,
            dtype,
            return_dtype: dtype,
            partial_dtype: &partial_dtype,
        };
        let error = aggregate
            .partial_from_scalar(args, &Scalar::null(partial_dtype.clone()))
            .err()
            .expect("partial decoding requires the storage claim");
        let expected = format!("{name} requires the {claim} claim, but {dtype} does not hold it");
        assert_eq!(error.to_string().lines().next(), Some(expected.as_str()));
    }

    #[test]
    fn direct_all_non_distinct_accumulation_checks_nested_eq() -> VortexResult<()> {
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let scalar = Scalar::extension::<Opaque>(EmptyMetadata, 1i64.into());
        let array = ConstantArray::new(scalar, 1).into_array();
        let pair = StructArray::from_fields(&[("lhs", array.clone()), ("rhs", array)])?;
        let dtype = pair.dtype().clone();
        let result_dtype = DType::Bool(Nullability::NonNullable);
        let args = AggregateArgs {
            options: &EmptyOptions,
            dtype: &dtype,
            return_dtype: &result_dtype,
            partial_dtype: &result_dtype,
        };
        let mut partial = AllNonDistinct.empty_partial(args)?;
        let error = AllNonDistinct
            .accumulate(
                args,
                &mut partial,
                &Columnar::Canonical(Canonical::Struct(pair)),
                &mut ctx,
            )
            .expect_err("comparing aggregate fields requires Eq");
        let expected =
            format!("all_non_distinct requires the Eq claim, but {dtype} does not hold it");
        assert_eq!(error.to_string().lines().next(), Some(expected.as_str()));
        Ok(())
    }
}
