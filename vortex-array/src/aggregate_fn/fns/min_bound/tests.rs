// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::MinBound;
use super::MinBoundPartial;
use super::MinBoundState;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateDTypes;
use crate::aggregate_fn::AggregateFnSatisfaction;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::bound::BOUND_IS_EXACT;
use crate::aggregate_fn::fns::bound::BOUND_VALUE;
use crate::aggregate_fn::fns::bound::BoundOptions;
use crate::aggregate_fn::fns::bounded_min::BoundedMin;
use crate::aggregate_fn::fns::bounded_min::BoundedMinOptions;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::max_bound::MaxBound;
use crate::aggregate_fn::fns::min::Min;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::validity::Validity;

fn options(max_bytes: usize) -> BoundOptions {
    BoundOptions::new(NonZeroUsize::new(max_bytes).vortex_expect("non-zero max_bytes"))
}

fn utf8(value: &str) -> Scalar {
    Scalar::utf8(value, Nullability::NonNullable)
}

fn accumulator(dtype: &DType, max_bytes: usize) -> VortexResult<Accumulator<MinBound>> {
    Accumulator::try_new(MinBound, options(max_bytes), dtype.clone())
}

/// Accumulate `batches` one at a time under the byte limit and return the final state.
fn state_of(batches: &[ArrayRef], max_bytes: usize) -> VortexResult<MinBoundState> {
    let mut ctx = array_session().create_execution_ctx();
    let mut acc = accumulator(batches[0].dtype(), max_bytes)?;
    for batch in batches {
        acc.accumulate(batch, &mut ctx)?;
    }
    let options = options(max_bytes);
    let dtypes = AggregateDTypes::try_new(&MinBound, &options, batches[0].dtype().clone())?;
    Ok(MinBound
        .partial_from_scalar(dtypes.args(&options), &acc.partial_scalar()?)?
        .into_state())
}

fn strings<const N: usize>(values: [&str; N]) -> ArrayRef {
    VarBinViewArray::from_iter_str(values).into_array()
}

#[test]
fn short_strings_are_exact() -> VortexResult<()> {
    assert_eq!(
        state_of(&[strings(["snowman", "untruncated"])], 9)?,
        MinBoundState::Exact(utf8("snowman"))
    );
    Ok(())
}

#[test]
fn long_string_truncates_to_lower_bound() -> VortexResult<()> {
    // The 9-byte prefix of "snowman⛄️snowman" ends inside the snowman, so the bound is the prefix
    // up to the last character boundary.
    assert_eq!(
        state_of(&[strings(["snowman⛄️snowman", "untruncated"])], 9)?,
        MinBoundState::LowerBound(utf8("snowman"))
    );
    Ok(())
}

#[test]
fn a_lower_bound_always_exists() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = VarBinViewArray::from_iter_bin([&[0u8, 0, 0][..]]).into_array();
    let mut acc = accumulator(array.dtype(), 2)?;

    acc.accumulate(&array, &mut ctx)?;

    assert!(!acc.is_saturated());
    assert_eq!(
        acc.finish()?,
        Scalar::binary(buffer![0u8, 0], Nullability::Nullable)
    );
    Ok(())
}

#[test]
fn fixed_width_values_are_exact() -> VortexResult<()> {
    let array = PrimitiveArray::new(buffer![10i32, 20, 5], Validity::NonNullable).into_array();
    assert_eq!(
        state_of(&[array], 1)?,
        MinBoundState::Exact(Scalar::primitive(5i32, Nullability::NonNullable))
    );
    Ok(())
}

#[test]
fn empty_and_all_null_batches_are_empty() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let empty = VarBinViewArray::from_iter_bin(Vec::<&[u8]>::new()).into_array();
    let nulls = VarBinViewArray::from_iter_nullable_str([None::<&str>, None]).into_array();
    let mut acc = accumulator(empty.dtype(), 2)?;

    acc.accumulate(&empty, &mut ctx)?;
    assert_eq!(acc.finish()?, Scalar::null(empty.dtype().as_nullable()));

    assert_eq!(
        state_of(&[nulls], 2)?,
        MinBoundState::Empty,
        "all-null input"
    );
    Ok(())
}

/// The exact minimum of one batch is kept only when it is at or below the other batch's bound.
#[rstest]
#[case::exact_below_bound(["aa"], ["xyzxyz"], MinBoundState::Exact(utf8("aa")))]
#[case::exact_at_bound(["xyz"], ["xyzxyz"], MinBoundState::Exact(utf8("xyz")))]
#[case::exact_above_bound(["xzz"], ["xyzxyz"], MinBoundState::LowerBound(utf8("xyz")))]
#[case::both_bounds(["abcdef"], ["xyzxyz"], MinBoundState::LowerBound(utf8("abc")))]
#[case::both_exact(["abc"], ["xyz"], MinBoundState::Exact(utf8("abc")))]
fn merging_batches(
    #[case] first: [&str; 1],
    #[case] second: [&str; 1],
    #[case] expected: MinBoundState,
) -> VortexResult<()> {
    assert_eq!(state_of(&[strings(first), strings(second)], 3)?, expected);
    // The merge is commutative.
    assert_eq!(state_of(&[strings(second), strings(first)], 3)?, expected);
    Ok(())
}

#[test]
fn empty_partial_is_the_merge_identity() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values = VarBinViewArray::from_iter_bin([&[1u8][..]]).into_array();
    let mut acc = accumulator(values.dtype(), 2)?;

    acc.accumulate(&values, &mut ctx)?;
    let empty = acc.empty_partial()?;
    acc.fold_partial(empty)?;

    assert_eq!(
        acc.finish()?,
        Scalar::binary(buffer![1u8], Nullability::Nullable)
    );
    Ok(())
}

#[test]
fn partial_dtype_is_non_null_value_and_is_exact() {
    let dtype = DType::Utf8(Nullability::Nullable);
    let partial_dtype = MinBound
        .partial_dtype(&options(8), &dtype)
        .expect("supported dtype");

    assert_eq!(partial_dtype.nullability(), Nullability::Nullable);
    let fields = partial_dtype.as_struct_fields();
    assert_eq!(fields.names().as_ref(), &[BOUND_VALUE, BOUND_IS_EXACT]);
    assert_eq!(
        fields.field(BOUND_VALUE),
        Some(DType::Utf8(Nullability::NonNullable))
    );
    assert_eq!(
        fields.field(BOUND_IS_EXACT),
        Some(DType::Bool(Nullability::NonNullable))
    );
    assert_eq!(
        MinBound.return_dtype(&options(8), &dtype),
        Some(DType::Utf8(Nullability::Nullable))
    );
}

#[rstest]
#[case::empty(MinBoundState::Empty)]
#[case::exact(MinBoundState::Exact(utf8("abc")))]
#[case::lower_bound(MinBoundState::LowerBound(utf8("abc")))]
fn partial_scalar_round_trips(#[case] state: MinBoundState) -> VortexResult<()> {
    let options = options(3);
    let dtypes =
        AggregateDTypes::try_new(&MinBound, &options, DType::Utf8(Nullability::NonNullable))?;
    let args = dtypes.args(&options);
    let partial = MinBoundPartial::new(state.clone());

    let scalar = MinBound.to_scalar(args, &partial)?;
    assert_eq!(scalar.dtype(), args.partial_dtype);
    assert_eq!(
        MinBound.partial_from_scalar(args, &scalar)?.into_state(),
        state
    );

    let finalized = MinBound.finalize_scalar(args, &partial)?;
    match state {
        MinBoundState::Exact(value) | MinBoundState::LowerBound(value) => {
            assert_eq!(finalized, value.into_nullable());
        }
        MinBoundState::Empty => {
            assert_eq!(finalized, Scalar::null(DType::Utf8(Nullability::Nullable)));
        }
    }
    Ok(())
}

#[test]
fn partial_without_value_is_rejected() -> VortexResult<()> {
    let dtype = DType::Utf8(Nullability::NonNullable);
    let options = options(3);
    let dtypes = AggregateDTypes::try_new(&MinBound, &options, dtype.clone())?;
    let args = dtypes.args(&options);
    // Build the malformed partial under a nullable value dtype, since the real one forbids it.
    let malformed = Scalar::struct_(
        MaxBound
            .partial_dtype(&options, &dtype)
            .expect("supported dtype"),
        vec![
            Scalar::null(dtype.as_nullable()),
            Scalar::bool(false, Nullability::NonNullable),
        ],
    );

    let error = MinBound
        .partial_from_scalar(args, &malformed)
        .expect_err("a minimum bound needs a value");
    assert!(error.to_string().contains("no value"), "{error}");
    Ok(())
}

#[test]
fn finalize_projects_values_through_null_partials() -> VortexResult<()> {
    let ctx = &mut array_session().create_execution_ctx();
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let dtypes = AggregateDTypes::try_new(&MinBound, &options(8), dtype.clone())?;
    let partials = StructArray::try_new(
        [BOUND_VALUE, BOUND_IS_EXACT].into(),
        vec![
            PrimitiveArray::new(buffer![5i32, 6, 7], Validity::NonNullable).into_array(),
            BoolArray::from_iter([true, false, false]).into_array(),
        ],
        3,
        Validity::from_iter([true, true, false]),
    )?
    .into_array();

    let finalized = MinBound.finalize(dtypes.args(&options(8)), partials)?;

    assert_eq!(finalized.dtype(), &dtype.as_nullable());
    assert_arrays_eq!(
        finalized,
        PrimitiveArray::from_option_iter([Some(5i32), Some(6), None]),
        ctx
    );
    Ok(())
}

#[test]
fn satisfies_min_requests() {
    let stored = MinBound.bind(options(5));

    assert_eq!(
        stored.can_satisfy(&MinBound.bind(options(5))),
        AggregateFnSatisfaction::Exact
    );
    assert_eq!(
        stored.can_satisfy(&MinBound.bind(options(4))),
        AggregateFnSatisfaction::Approximate
    );
    assert_eq!(
        stored.can_satisfy(&MinBound.bind(options(6))),
        AggregateFnSatisfaction::Approximate
    );
    assert_eq!(
        stored.can_satisfy(&Min.bind(NumericalAggregateOpts::skip_nans())),
        AggregateFnSatisfaction::Approximate
    );
    assert_eq!(
        stored.can_satisfy(&Min.bind(NumericalAggregateOpts::include_nans())),
        AggregateFnSatisfaction::No
    );
    assert_eq!(
        Min.bind(NumericalAggregateOpts::skip_nans())
            .can_satisfy(&stored),
        AggregateFnSatisfaction::Approximate
    );
    assert_eq!(
        Min.bind(NumericalAggregateOpts::include_nans())
            .can_satisfy(&stored),
        AggregateFnSatisfaction::No
    );
    assert_eq!(
        stored.can_satisfy(&Max.bind(NumericalAggregateOpts::skip_nans())),
        AggregateFnSatisfaction::No
    );
    assert_eq!(
        stored.can_satisfy(&MaxBound.bind(options(5))),
        AggregateFnSatisfaction::No
    );
}

#[test]
fn bounded_min_and_min_bound_satisfy_each_other() {
    let legacy = BoundedMin.bind(BoundedMinOptions {
        max_bytes: options(5).max_bytes,
    });
    let stored = MinBound.bind(options(5));

    assert_eq!(
        stored.can_satisfy(&legacy),
        AggregateFnSatisfaction::Approximate
    );
    assert_eq!(
        legacy.can_satisfy(&stored),
        AggregateFnSatisfaction::Approximate
    );
}

#[test]
fn options_round_trip() -> VortexResult<()> {
    let options = options(64);
    let metadata = MinBound.serialize(&options)?.expect("serializable options");
    let roundtrip = MinBound.deserialize(&metadata, &VortexSession::empty())?;

    assert_eq!(roundtrip, options);
    assert!(
        MinBound
            .deserialize(&[0; 8], &VortexSession::empty())
            .is_err()
    );
    assert!(
        MinBound
            .deserialize(&[1; 3], &VortexSession::empty())
            .is_err()
    );
    Ok(())
}

/// `vortex.bounded_min` partials are the nullable bound itself: null is empty, a string value
/// can only be read as a lower bound, and a fixed-width value was never truncated.
#[rstest]
#[case::empty(Scalar::null(DType::Utf8(Nullability::Nullable)), MinBoundState::Empty)]
#[case::string(
    Scalar::utf8("abc", Nullability::Nullable),
    MinBoundState::LowerBound(Scalar::utf8("abc", Nullability::Nullable))
)]
#[case::fixed_width(
    Scalar::primitive(7i32, Nullability::Nullable),
    MinBoundState::Exact(Scalar::primitive(7i32, Nullability::Nullable))
)]
fn maps_bounded_min_partials(
    #[case] legacy: Scalar,
    #[case] expected: MinBoundState,
) -> VortexResult<()> {
    assert_eq!(
        MinBound.partial_from_bounded_min(&legacy)?.into_state(),
        expected
    );
    Ok(())
}
