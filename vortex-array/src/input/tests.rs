// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::AggregateCacheMode;
use super::ArrayInput;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::make_minmax_dtype;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::dict::TakeExecute;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::kernels::KernelSession;
use crate::scalar::Scalar;
use crate::validity::Validity;

#[rstest]
#[case(AggregateCacheMode::Array)]
#[case(AggregateCacheMode::Input)]
#[case(AggregateCacheMode::Disabled)]
fn selected_store_and_clones(#[case] mode: AggregateCacheMode) -> VortexResult<()> {
    let array = buffer![1i32, 2, 3].into_array();
    let input = ArrayInput::new(array.clone()).with_cache_mode(mode);
    let clone = input.clone();
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let mut ctx = array_session().create_execution_ctx();
    assert_eq!(input.get_result(&sum), Precision::Absent);
    assert_eq!(i64::try_from(&input.compute_result(&sum, &mut ctx)?)?, 6);
    let cached = Precision::Exact(Scalar::primitive(6i64, Nullability::Nullable));
    match mode {
        AggregateCacheMode::Array => {
            assert_eq!(clone.get_result(&sum), cached);
            assert_eq!(array.aggregations().get_result(&sum), cached);
        }
        AggregateCacheMode::Input => {
            assert_eq!(clone.get_result(&sum), cached);
            assert_eq!(array.aggregations().get_result(&sum), Precision::Absent);
            assert_eq!(ArrayInput::new(array).get_result(&sum), Precision::Absent);
        }
        AggregateCacheMode::Disabled => {
            assert_eq!(clone.get_result(&sum), Precision::Absent);
            assert_eq!(array.aggregations().get_result(&sum), Precision::Absent);
        }
    }
    Ok(())
}

#[rstest]
#[case(false)]
#[case(true)]
fn numerical_options_do_not_alias(#[case] include_first: bool) -> VortexResult<()> {
    let input = ArrayInput::new(buffer![1.0f64, f64::NAN].into_array());
    let skip = Sum.bind(NumericalAggregateOpts::skip_nans());
    let include = Sum.bind(NumericalAggregateOpts::include_nans());
    let order = if include_first {
        [&include, &skip]
    } else {
        [&skip, &include]
    };
    let mut ctx = array_session().create_execution_ctx();
    input.compute_result(order[0], &mut ctx)?;
    assert_eq!(input.get_result(order[1]), Precision::Absent);
    input.compute_result(order[1], &mut ctx)?;
    assert_eq!(f64::try_from(&input.compute_result(&skip, &mut ctx)?)?, 1.0);
    assert!(f64::try_from(&input.compute_result(&include, &mut ctx)?)?.is_nan());
    Ok(())
}

#[rstest]
#[case(AggregateCacheMode::Array)]
#[case(AggregateCacheMode::Input)]
#[case(AggregateCacheMode::Disabled)]
fn producer_indices_handle_empty_all_and_sparse(
    #[case] mode: AggregateCacheMode,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    for (mask, expected) in [
        (Mask::AllFalse(4), vec![]),
        (Mask::AllTrue(4), vec![0u64, 1, 2, 3]),
        (Mask::from_indices(8, [1, 4, 6]), vec![1u64, 4, 6]),
    ] {
        let input = ArrayInput::from_mask_indices_with_cache_mode(&mask, mode)?;
        assert_arrays_eq!(input.array(), PrimitiveArray::from_iter(expected), &mut ctx);
        let sorted = IsSorted.bind(IsSortedOptions { strict: true });
        if mode == AggregateCacheMode::Disabled {
            assert_eq!(input.get_result(&sorted), Precision::Absent);
        } else {
            assert_eq!(input.get_result(&sorted), Precision::Exact(true.into()));
        }
    }
    Ok(())
}

#[rstest]
#[case::empty(Mask::AllFalse(0), 0)]
#[case::nonempty(Mask::from_indices(10, [1, 4, 6, 9]), 20)]
fn full_slice_and_all_true_filter_share_owner(
    #[case] mask: Mask,
    #[case] expected_sum: u64,
    #[values(
        AggregateCacheMode::Array,
        AggregateCacheMode::Input,
        AggregateCacheMode::Disabled
    )]
    mode: AggregateCacheMode,
) -> VortexResult<()> {
    let input = ArrayInput::from_mask_indices_with_cache_mode(&mask, mode)?;
    let sliced = input.slice(0..input.array().len())?;
    let filtered = input.filter(Mask::AllTrue(input.array().len()))?;
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let min = Min.bind(NumericalAggregateOpts::skip_nans());
    let sorted = IsSorted.bind(IsSortedOptions { strict: true });
    let mut ctx = array_session().create_execution_ctx();
    assert_eq!(
        u64::try_from(&sliced.compute_result(&sum, &mut ctx)?)?,
        expected_sum
    );
    assert_eq!(
        input.verified_bounds().is_some(),
        mode != AggregateCacheMode::Disabled
    );

    for output in [&sliced, &filtered] {
        assert!(Arc::ptr_eq(&input.inner, &output.inner));
        assert!(ArrayRef::ptr_eq(input.array(), output.array()));
        assert_eq!(output.cache_mode(), mode);
        assert_eq!(
            output.verified_bounds().map(std::ptr::from_ref),
            input.verified_bounds().map(std::ptr::from_ref)
        );
        assert_eq!(output.get_result(&min), input.get_result(&min));
        assert_eq!(output.get_result(&sum), input.get_result(&sum));
        assert_eq!(output.get_result(&sorted), input.get_result(&sorted));
    }

    if mode == AggregateCacheMode::Disabled {
        assert_eq!(input.get_result(&sum), Precision::Absent);
        assert_eq!(input.get_result(&sorted), Precision::Absent);
    } else {
        assert_eq!(
            input.get_result(&sum),
            Precision::Exact(Scalar::primitive(expected_sum, Nullability::Nullable))
        );
        assert!(matches!(input.get_result(&min), Precision::Exact(_)));
        assert_eq!(input.get_result(&sorted), Precision::Exact(true.into()));
    }

    Ok(())
}

#[test]
fn fresh_disabled_validation_retains_no_proof_or_results() -> VortexResult<()> {
    let input = ArrayInput::new(buffer![3i32, 1, 2].into_array())
        .with_cache_mode(AggregateCacheMode::Disabled);
    let mut ctx = array_session().create_execution_ctx();
    input.validate_integer_bounds(&mut ctx)?;
    assert!(input.inner.verified_bounds.get().is_none());

    for mode in [AggregateCacheMode::Array, AggregateCacheMode::Input] {
        let enabled = input.clone().with_cache_mode(mode);
        assert!(enabled.verified_bounds().is_none());
        assert!(enabled.snapshot_results().iter().next().is_none());
    }

    Ok(())
}

#[test]
fn switching_modes_keeps_shared_results_and_private_proof() -> VortexResult<()> {
    let input = ArrayInput::new(buffer![2i32, 3, 5].into_array());
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let expected = Scalar::primitive(10i64, Nullability::Nullable);
    let mut ctx = array_session().create_execution_ctx();
    input.validate_integer_bounds(&mut ctx)?;
    assert_eq!(input.compute_result(&sum, &mut ctx)?, expected);
    let proof = std::ptr::from_ref(
        input
            .verified_bounds()
            .vortex_expect("direct validation retained proof"),
    );
    let array = input.clone().with_cache_mode(AggregateCacheMode::Array);
    assert_eq!(array.compute_result(&sum, &mut ctx)?, expected);
    let disabled = array.clone().with_cache_mode(AggregateCacheMode::Disabled);
    assert!(disabled.verified_bounds().is_none());
    assert_eq!(disabled.get_result(&sum), Precision::Absent);
    disabled.validate_integer_bounds(&mut ctx)?;

    for mode in [AggregateCacheMode::Array, AggregateCacheMode::Input] {
        let restored = disabled.clone().with_cache_mode(mode);
        assert!(Arc::ptr_eq(&input.inner, &restored.inner));
        assert_eq!(
            restored.verified_bounds().map(std::ptr::from_ref),
            Some(proof)
        );
        assert_eq!(
            restored.get_result(&sum),
            Precision::Exact(expected.clone())
        );
    }
    assert_eq!(input.cache_mode(), AggregateCacheMode::Input);
    assert_eq!(array.cache_mode(), AggregateCacheMode::Array);

    Ok(())
}

#[test]
fn stable_subsets_keep_bounds_and_positive_sortedness() -> VortexResult<()> {
    let input = ArrayInput::from_mask_indices(&Mask::from_indices(10, [1, 4, 6, 9]))?;
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let min = Min.bind(NumericalAggregateOpts::skip_nans());
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let sorted = IsSorted.bind(IsSortedOptions { strict: true });
    let mut ctx = array_session().create_execution_ctx();
    input.compute_result(&sum, &mut ctx)?;
    let sliced = input.slice(1..3)?;
    assert!(matches!(sliced.get_result(&min), Precision::Inexact(_)));
    assert!(matches!(sliced.get_result(&max), Precision::Inexact(_)));
    assert_eq!(sliced.get_result(&sorted), Precision::Exact(true.into()));
    assert_eq!(sliced.get_result(&sum), Precision::Absent);
    assert_eq!(u64::try_from(&sliced.compute_result(&min, &mut ctx)?)?, 4);
    assert_eq!(u64::try_from(&input.compute_result(&min, &mut ctx)?)?, 1);
    let filtered = input.filter(Mask::from_indices(4, [0, 2]))?;
    assert!(matches!(filtered.get_result(&max), Precision::Inexact(_)));
    assert_eq!(filtered.get_result(&sorted), Precision::Exact(true.into()));
    assert_arrays_eq!(filtered.array(), buffer![1u64, 6].into_array(), &mut ctx);
    Ok(())
}

#[test]
fn negative_sortedness_and_noninteger_facts_do_not_propagate() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let sorted = IsSorted.bind(IsSortedOptions { strict: false });
    let input = ArrayInput::new(buffer![3i32, 1, 2].into_array());
    assert_eq!(input.compute_result(&sorted, &mut ctx)?, false.into());
    let subset = input.slice(1..3)?;
    assert_eq!(subset.get_result(&sorted), Precision::Absent);
    assert_eq!(subset.compute_result(&sorted, &mut ctx)?, true.into());
    let float = ArrayInput::new(buffer![1.0f64, 2.0].into_array());
    float.compute_result(&sorted, &mut ctx)?;
    assert_eq!(float.slice(0..1)?.get_result(&sorted), Precision::Absent);
    Ok(())
}

#[test]
fn sortedness_uses_nulls_first_and_preserves_strictness() -> VortexResult<()> {
    let input = ArrayInput::new(
        PrimitiveArray::from_option_iter([None, Some(1i32), Some(1), Some(2)]).into_array(),
    );
    let sorted = IsSorted.bind(IsSortedOptions { strict: false });
    let strict = IsSorted.bind(IsSortedOptions { strict: true });
    let mut ctx = array_session().create_execution_ctx();
    assert_eq!(input.compute_result(&sorted, &mut ctx)?, true.into());
    assert_eq!(input.compute_result(&strict, &mut ctx)?, false.into());
    let subset = input.filter(Mask::from_indices(4, [0, 1, 3]))?;
    assert_eq!(subset.get_result(&sorted), Precision::Exact(true.into()));
    assert_eq!(subset.get_result(&strict), Precision::Absent);
    assert_eq!(subset.compute_result(&strict, &mut ctx)?, true.into());
    Ok(())
}

#[test]
fn scoped_contexts_retain_owners_and_leave_parent_unchanged() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let outer = ArrayInput::from_mask_indices(&Mask::AllTrue(3))?;
    let array = outer.array().clone();
    let sorted = IsSorted.bind(IsSortedOptions { strict: false });
    let parent = ctx.with_aggregate_input(&outer);
    drop(outer);
    assert_eq!(
        parent.aggregate_result(&array, &sorted),
        Precision::Exact(true.into())
    );
    let different = ArrayInput::new(buffer![9u64, 2, 1].into_array());
    let child = parent.with_aggregate_input(&different);
    assert_eq!(
        child.aggregate_result(&array, &sorted),
        Precision::Exact(true.into())
    );
    assert_eq!(
        child.aggregate_result(different.array(), &sorted),
        Precision::Absent
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _child = parent.with_aggregate_input(&different);
            panic!("scope unwind");
        }))
        .is_err()
    );
    assert_eq!(
        parent.aggregate_result(&array, &sorted),
        Precision::Exact(true.into())
    );
    assert!(different.execute_cast(PType::U8.into(), &mut ctx).is_ok());
    let invalid = ArrayInput::new(buffer![-1i32].into_array());
    assert!(invalid.execute_cast(PType::U32.into(), &mut ctx).is_err());
    assert_eq!(ctx.aggregate_cache_mode(), AggregateCacheMode::Array);
    Ok(())
}

#[derive(Debug)]
struct IncorrectMinMax;

impl DynAggregateKernel for IncorrectMinMax {
    fn aggregate(
        &self,
        _: &AggregateFnRef,
        batch: &ArrayRef,
        _: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        Ok(Some(Scalar::struct_(
            make_minmax_dtype(batch.dtype()),
            vec![
                Scalar::primitive(0i16, Nullability::NonNullable),
                Scalar::primitive(1i16, Nullability::NonNullable),
            ],
        )))
    }
}

#[test]
fn safe_aggregate_results_cannot_bypass_decimal_precision() -> VortexResult<()> {
    let session = array_session();
    session.aggregate_fns().register_aggregate_kernel(
        Primitive.id(),
        Some(MinMax.id()),
        &IncorrectMinMax,
    );
    let mut ctx = session.create_execution_ctx();
    let input = ArrayInput::new(buffer![1000i16].into_array());
    input.compute_result(&MinMax.bind(NumericalAggregateOpts::skip_nans()), &mut ctx)?;
    assert!(input.verified_bounds().is_none());
    let decimal = DType::Decimal(DecimalDType::new(3, 0), Nullability::NonNullable);
    assert!(input.execute_cast(decimal.clone(), &mut ctx).is_err());
    input.validate_integer_bounds(&mut ctx)?;
    assert!(
        !input
            .verified_bounds()
            .vortex_expect("direct validation retained proof")
            .fits(&decimal)
    );
    assert!(input.execute_cast(decimal, &mut ctx).is_err());
    Ok(())
}

#[test]
fn physical_proof_includes_null_payloads_and_cast_checks_nullability() -> VortexResult<()> {
    let array = PrimitiveArray::new(buffer![1000i16, 1], Validity::from_iter([false, true]));
    let input = ArrayInput::new(array.into_array());
    let mut ctx = array_session().create_execution_ctx();
    input.validate_integer_bounds(&mut ctx)?;
    let dtype = DType::Decimal(DecimalDType::new(2, 0), Nullability::Nullable);
    assert!(
        !input
            .verified_bounds()
            .vortex_expect("direct validation retained proof")
            .fits(&dtype)
    );
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    assert_eq!(
        input.get_result(&max),
        Precision::Exact(Scalar::primitive(1i16, Nullability::Nullable))
    );
    assert!(input.execute_cast(dtype, &mut ctx).is_ok());
    assert!(
        input
            .execute_cast(
                DType::Decimal(DecimalDType::new(2, 0), Nullability::NonNullable),
                &mut ctx
            )
            .is_err()
    );
    assert!(
        ArrayInput::new(ConstantArray::new(1i32, 4).into_array())
            .validate_integer_bounds(&mut ctx)
            .is_err()
    );
    assert!(
        ArrayInput::new(buffer![1.0f32].into_array())
            .validate_integer_bounds(&mut ctx)
            .is_err()
    );
    Ok(())
}

#[test]
fn input_subsets_bypass_array_result_inheritance() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = buffer![1i32, 2, 3].into_array();
    let sorted = IsSorted.bind(IsSortedOptions { strict: false });
    array.aggregations().compute_result(&sorted, &mut ctx)?;
    let input = ArrayInput::new(array);
    let sliced = input.slice(0..2)?;
    assert_eq!(sliced.get_result(&sorted), Precision::Absent);
    assert_eq!(
        sliced.array().aggregations().get_result(&sorted),
        Precision::Absent
    );
    Ok(())
}

#[rstest]
#[case(AggregateCacheMode::Input)]
#[case(AggregateCacheMode::Disabled)]
fn nullable_cast_does_not_populate_validity_array_cache(
    #[case] mode: AggregateCacheMode,
) -> VortexResult<()> {
    let validity = BoolArray::from_iter([true, true]).into_array();
    let input = ArrayInput::new(
        PrimitiveArray::new(buffer![12i16, 34], Validity::Array(validity.clone())).into_array(),
    )
    .with_cache_mode(mode);
    let mut ctx = array_session().create_execution_ctx();
    input.validate_integer_bounds(&mut ctx)?;
    input.execute_cast(
        DType::Decimal(DecimalDType::new(3, 0), Nullability::NonNullable),
        &mut ctx,
    )?;
    assert_eq!(
        validity
            .aggregations()
            .get_result(&Min.bind(NumericalAggregateOpts::skip_nans())),
        Precision::Absent
    );
    Ok(())
}

#[rstest]
#[case(AggregateCacheMode::Array)]
#[case(AggregateCacheMode::Input)]
#[case(AggregateCacheMode::Disabled)]
fn an_empty_slice_does_not_inherit_constantness(
    #[case] mode: AggregateCacheMode,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = ArrayInput::new(buffer![7i32, 7].into_array()).with_cache_mode(mode);
    let constant = IsConstant.bind(EmptyOptions);
    assert_eq!(input.compute_result(&constant, &mut ctx)?, true.into());
    let empty = input.slice(0..0)?;
    assert_eq!(empty.get_result(&constant), Precision::Absent);
    assert_eq!(empty.compute_result(&constant, &mut ctx)?, false.into());
    Ok(())
}

#[test]
fn private_cast_proof_does_not_follow_nonempty_subsets() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = ArrayInput::new(buffer![1i16, 2, 3].into_array());
    input.validate_integer_bounds(&mut ctx)?;
    assert!(input.verified_bounds().is_some());
    assert!(input.slice(1..3)?.verified_bounds().is_none());
    assert!(
        input
            .filter(Mask::from_indices(3, [0, 2]))?
            .verified_bounds()
            .is_none()
    );
    assert!(input.slice(0..0)?.verified_bounds().is_some());
    Ok(())
}

#[derive(Debug)]
struct ObserveTakeInputs {
    source_result: Precision<Scalar>,
    indices_result: Precision<Scalar>,
    called: Arc<AtomicBool>,
}

impl ExecuteParentKernel<Primitive> for ObserveTakeInputs {
    type Parent = Dict;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Primitive>,
        parent: ArrayView<'_, Dict>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 1 {
            return Ok(None);
        }
        let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
        assert_eq!(
            ctx.aggregate_result(array.array(), &sum),
            self.source_result
        );
        assert_eq!(
            ctx.aggregate_result(parent.codes(), &sum),
            self.indices_result
        );
        self.called.store(true, Ordering::Relaxed);
        <Primitive as TakeExecute>::take(array, parent.codes(), ctx)
    }
}

#[rstest]
fn eager_take_keeps_both_owners_and_their_modes(
    #[values(
        AggregateCacheMode::Array,
        AggregateCacheMode::Input,
        AggregateCacheMode::Disabled
    )]
    source_mode: AggregateCacheMode,
    #[values(
        AggregateCacheMode::Array,
        AggregateCacheMode::Input,
        AggregateCacheMode::Disabled
    )]
    indices_mode: AggregateCacheMode,
) -> VortexResult<()> {
    let source = ArrayInput::new(buffer![11i32, 22, 33].into_array()).with_cache_mode(source_mode);
    let indices = ArrayInput::new(buffer![2u64, 0].into_array()).with_cache_mode(indices_mode);
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let mut preparation = array_session().create_execution_ctx();
    source.compute_result(&sum, &mut preparation)?;
    indices.compute_result(&sum, &mut preparation)?;

    let session = VortexSession::empty().with_some(KernelSession::empty());
    let called = Arc::new(AtomicBool::new(false));
    session.kernels().register_execute_parent_kernel(
        Dict.id(),
        Primitive,
        ObserveTakeInputs {
            source_result: source.get_result(&sum),
            indices_result: indices.get_result(&sum),
            called: Arc::clone(&called),
        },
    );
    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(
        source.execute_take(&indices, &mut ctx)?.into_array(),
        buffer![33i32, 11].into_array(),
        &mut preparation
    );
    assert!(called.load(Ordering::Relaxed));
    assert!(ctx.aggregate_inputs.is_none());
    assert_eq!(ctx.aggregate_cache_mode(), AggregateCacheMode::Array);
    Ok(())
}

#[test]
fn nearest_owner_wins_for_the_same_array_handle() -> VortexResult<()> {
    let input = ArrayInput::from_mask_indices(&Mask::AllTrue(3))?;
    let array_owner =
        ArrayInput::new(input.array().clone()).with_cache_mode(AggregateCacheMode::Array);
    let sorted = IsSorted.bind(IsSortedOptions { strict: false });
    let parent = array_session()
        .create_execution_ctx()
        .with_aggregate_input(&input);
    let child = parent.with_aggregate_input(&array_owner);
    assert_eq!(
        child.aggregate_result(input.array(), &sorted),
        Precision::Absent
    );
    assert_eq!(
        parent.aggregate_result(input.array(), &sorted),
        Precision::Exact(true.into())
    );
    Ok(())
}
