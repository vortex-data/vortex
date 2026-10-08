// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::MaxBound;
use super::MaxBoundPartial;
use super::MaxBoundState;
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
use crate::aggregate_fn::fns::bound::bound_partial_dtype;
use crate::aggregate_fn::fns::bounded_max::BoundedMax;
use crate::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
use crate::aggregate_fn::fns::bounded_max::make_bounded_max_partial_dtype;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_bound::MinBound;
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
    Scalar::utf8(value, Nullability::Nullable)
}

fn accumulator(dtype: &DType, max_bytes: usize) -> VortexResult<Accumulator<MaxBound>> {
    Accumulator::try_new(MaxBound, options(max_bytes), dtype.clone())
}

/// Accumulate `batches` one at a time under the byte limit and return the final state.
fn state_of(batches: &[ArrayRef], max_bytes: usize) -> VortexResult<MaxBoundState> {
    let mut ctx = array_session().create_execution_ctx();
    let mut acc = accumulator(batches[0].dtype(), max_bytes)?;
    for batch in batches {
        acc.accumulate(batch, &mut ctx)?;
    }
    let options = options(max_bytes);
    let dtypes = AggregateDTypes::try_new(&MaxBound, &options, batches[0].dtype().clone())?;
    Ok(MaxBound
        .partial_from_scalar(dtypes.args(&options), &acc.partial_scalar()?)?
        .into_state())
}

fn strings<const N: usize>(values: [&str; N]) -> ArrayRef {
    VarBinViewArray::from_iter_str(values).into_array()
}

#[test]
fn short_strings_are_exact() -> VortexResult<()> {
    assert_eq!(
        state_of(&[strings(["aardvark", "char"])], 8)?,
        MaxBoundState::Exact(utf8("char"))
    );
    Ok(())
}

#[test]
fn long_string_truncates_to_upper_bound() -> VortexResult<()> {
    // The 5-byte prefix of "char🪩" ends inside the emoji, so the bound is "char" incremented.
    assert_eq!(
        state_of(&[strings(["aardvark", "char🪩"])], 5)?,
        MaxBoundState::UpperBound(utf8("chas"))
    );
    Ok(())
}

#[test]
fn unrepresentable_bound_returns_null_and_saturates() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = VarBinViewArray::from_iter_bin([&[255u8, 255, 255][..]]).into_array();
    let mut acc = accumulator(array.dtype(), 2)?;

    acc.accumulate(&array, &mut ctx)?;

    assert!(acc.is_saturated());
    assert_eq!(acc.finish()?, Scalar::null(array.dtype().as_nullable()));
    Ok(())
}

#[test]
fn fixed_width_values_are_exact() -> VortexResult<()> {
    let array = PrimitiveArray::new(buffer![10i32, 20, 5], Validity::NonNullable).into_array();
    assert_eq!(
        state_of(&[array], 1)?,
        MaxBoundState::Exact(Scalar::primitive(20i32, Nullability::Nullable))
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
        MaxBoundState::Empty,
        "all-null input"
    );
    Ok(())
}

/// The exact maximum of one batch is kept only when it is at or above the other batch's bound.
#[rstest]
#[case::exact_above_bound(["zz"], ["abcdef"], MaxBoundState::Exact(utf8("zz")))]
#[case::exact_at_bound(["abd"], ["abcdef"], MaxBoundState::Exact(utf8("abd")))]
#[case::exact_below_bound(["abc"], ["abcdef"], MaxBoundState::UpperBound(utf8("abd")))]
#[case::both_bounds(["abcdef"], ["xyzxyz"], MaxBoundState::UpperBound(utf8("xy{")))]
#[case::both_exact(["abc"], ["xyz"], MaxBoundState::Exact(utf8("xyz")))]
fn merging_batches(
    #[case] first: [&str; 1],
    #[case] second: [&str; 1],
    #[case] expected: MaxBoundState,
) -> VortexResult<()> {
    assert_eq!(state_of(&[strings(first), strings(second)], 3)?, expected);
    // The merge is commutative.
    assert_eq!(state_of(&[strings(second), strings(first)], 3)?, expected);
    Ok(())
}

#[test]
fn unrepresentable_poisons_in_either_order() -> VortexResult<()> {
    let unrepresentable = VarBinViewArray::from_iter_bin([&[255u8, 255, 255][..]]).into_array();
    let values = VarBinViewArray::from_iter_bin([&[1u8][..]]).into_array();

    assert_eq!(
        state_of(&[unrepresentable.clone(), values.clone()], 2)?,
        MaxBoundState::Unrepresentable
    );
    assert_eq!(
        state_of(&[values, unrepresentable], 2)?,
        MaxBoundState::Unrepresentable
    );
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
fn partial_dtype_is_value_and_is_exact() {
    let dtype = DType::Utf8(Nullability::NonNullable);
    let partial_dtype = MaxBound
        .partial_dtype(&options(8), &dtype)
        .expect("supported dtype");

    assert_eq!(partial_dtype.nullability(), Nullability::Nullable);
    let fields = partial_dtype.as_struct_fields();
    assert_eq!(fields.names().as_ref(), &[BOUND_VALUE, BOUND_IS_EXACT]);
    assert_eq!(
        fields.field(BOUND_VALUE),
        Some(DType::Utf8(Nullability::Nullable))
    );
    assert_eq!(
        fields.field(BOUND_IS_EXACT),
        Some(DType::Bool(Nullability::NonNullable))
    );
    assert_eq!(
        MaxBound.return_dtype(&options(8), &dtype),
        Some(DType::Utf8(Nullability::Nullable))
    );
}

#[rstest]
#[case::empty(MaxBoundState::Empty)]
#[case::exact(MaxBoundState::Exact(utf8("abc")))]
#[case::upper_bound(MaxBoundState::UpperBound(utf8("abd")))]
#[case::unrepresentable(MaxBoundState::Unrepresentable)]
fn partial_scalar_round_trips(#[case] state: MaxBoundState) -> VortexResult<()> {
    let options = options(3);
    let dtypes =
        AggregateDTypes::try_new(&MaxBound, &options, DType::Utf8(Nullability::NonNullable))?;
    let args = dtypes.args(&options);
    let partial = MaxBoundPartial::new(state.clone());

    let scalar = MaxBound.to_scalar(args, &partial)?;
    assert_eq!(scalar.dtype(), args.partial_dtype);
    assert_eq!(
        MaxBound.partial_from_scalar(args, &scalar)?.into_state(),
        state
    );

    let finalized = MaxBound.finalize_scalar(args, &partial)?;
    match state {
        MaxBoundState::Exact(value) | MaxBoundState::UpperBound(value) => {
            assert_eq!(finalized, value);
        }
        MaxBoundState::Empty | MaxBoundState::Unrepresentable => {
            assert_eq!(finalized, Scalar::null(DType::Utf8(Nullability::Nullable)));
        }
    }
    Ok(())
}

#[test]
fn exact_partial_without_value_is_rejected() -> VortexResult<()> {
    let dtype = DType::Utf8(Nullability::NonNullable);
    let options = options(3);
    let dtypes = AggregateDTypes::try_new(&MaxBound, &options, dtype.clone())?;
    let args = dtypes.args(&options);
    let malformed = Scalar::struct_(
        args.partial_dtype.clone(),
        vec![
            Scalar::null(dtype.as_nullable()),
            Scalar::bool(true, Nullability::NonNullable),
        ],
    );

    let error = MaxBound
        .partial_from_scalar(args, &malformed)
        .expect_err("an exact maximum needs a value");
    assert!(error.to_string().contains("no value"), "{error}");
    Ok(())
}

#[test]
fn finalize_projects_values_through_null_partials() -> VortexResult<()> {
    let ctx = &mut array_session().create_execution_ctx();
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let dtypes = AggregateDTypes::try_new(&MaxBound, &options(8), dtype.clone())?;
    let partials = StructArray::try_new(
        [BOUND_VALUE, BOUND_IS_EXACT].into(),
        vec![
            PrimitiveArray::from_option_iter([Some(5i32), None, Some(7)]).into_array(),
            BoolArray::from_iter([true, false, false]).into_array(),
        ],
        3,
        Validity::from_iter([true, true, false]),
    )?
    .into_array();

    let finalized = MaxBound.finalize(dtypes.args(&options(8)), partials)?;

    assert_eq!(finalized.dtype(), &dtype.as_nullable());
    assert_arrays_eq!(
        finalized,
        PrimitiveArray::from_option_iter([Some(5i32), None, None]),
        ctx
    );
    Ok(())
}

#[test]
fn satisfies_max_requests() {
    let stored = MaxBound.bind(options(5));

    assert_eq!(
        stored.can_satisfy(&MaxBound.bind(options(5))),
        AggregateFnSatisfaction::Exact
    );
    assert_eq!(
        stored.can_satisfy(&MaxBound.bind(options(4))),
        AggregateFnSatisfaction::Approximate
    );
    assert_eq!(
        stored.can_satisfy(&MaxBound.bind(options(6))),
        AggregateFnSatisfaction::Approximate
    );
    assert_eq!(
        stored.can_satisfy(&Max.bind(NumericalAggregateOpts::skip_nans())),
        AggregateFnSatisfaction::Approximate
    );
    assert_eq!(
        stored.can_satisfy(&Max.bind(NumericalAggregateOpts::include_nans())),
        AggregateFnSatisfaction::No
    );
    assert_eq!(
        Max.bind(NumericalAggregateOpts::skip_nans())
            .can_satisfy(&stored),
        AggregateFnSatisfaction::Approximate
    );
    assert_eq!(
        Max.bind(NumericalAggregateOpts::include_nans())
            .can_satisfy(&stored),
        AggregateFnSatisfaction::No
    );
    assert_eq!(
        stored.can_satisfy(&Min.bind(NumericalAggregateOpts::skip_nans())),
        AggregateFnSatisfaction::No
    );
    assert_eq!(
        stored.can_satisfy(&MinBound.bind(options(5))),
        AggregateFnSatisfaction::No
    );
}

#[test]
fn bounded_max_and_max_bound_satisfy_each_other() {
    let legacy = BoundedMax.bind(BoundedMaxOptions {
        max_bytes: options(5).max_bytes,
    });
    let stored = MaxBound.bind(options(5));

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
    let metadata = MaxBound.serialize(&options)?.expect("serializable options");
    let roundtrip = MaxBound.deserialize(&metadata, &VortexSession::empty())?;

    assert_eq!(roundtrip, options);
    assert!(
        MaxBound
            .deserialize(&[0; 8], &VortexSession::empty())
            .is_err()
    );
    assert!(
        MaxBound
            .deserialize(&[1; 3], &VortexSession::empty())
            .is_err()
    );
    Ok(())
}

/// `vortex.bounded_max` partials map onto the v2 states: a null partial is empty, an unknown
/// bound is unrepresentable, and a string value can only be read as an upper bound.
#[rstest]
#[case::empty(None, MaxBoundState::Empty)]
#[case::value(Some((Some("abd"), false)), MaxBoundState::UpperBound(utf8("abd")))]
#[case::unknown(Some((None, true)), MaxBoundState::Unrepresentable)]
fn maps_bounded_max_partials(
    #[case] legacy: Option<(Option<&str>, bool)>,
    #[case] expected: MaxBoundState,
) -> VortexResult<()> {
    let element_dtype = DType::Utf8(Nullability::NonNullable);
    let legacy_dtype = make_bounded_max_partial_dtype(&element_dtype);
    let scalar = match legacy {
        None => Scalar::null(legacy_dtype),
        Some((bound, unknown)) => Scalar::struct_(
            legacy_dtype,
            vec![
                bound.map_or_else(|| Scalar::null(element_dtype.as_nullable()), utf8),
                Scalar::bool(unknown, Nullability::NonNullable),
            ],
        ),
    };

    assert_eq!(
        MaxBound.partial_from_bounded_max(&scalar)?.into_state(),
        expected
    );
    Ok(())
}

#[test]
fn maps_fixed_width_bounded_max_partials_as_exact() -> VortexResult<()> {
    let element_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let scalar = Scalar::struct_(
        make_bounded_max_partial_dtype(&element_dtype),
        vec![
            Scalar::primitive(7i32, Nullability::Nullable),
            Scalar::bool(false, Nullability::NonNullable),
        ],
    );

    assert_eq!(
        MaxBound.partial_from_bounded_max(&scalar)?.into_state(),
        MaxBoundState::Exact(Scalar::primitive(7i32, Nullability::Nullable))
    );
    Ok(())
}

/// A v2 partial scalar has the same shape the bounded max promised, only with exactness.
#[test]
fn partial_dtype_matches_helper() {
    let dtype = DType::Binary(Nullability::Nullable);
    assert_eq!(
        MaxBound.partial_dtype(&options(8), &dtype),
        Some(bound_partial_dtype(&dtype, Nullability::Nullable))
    );
}
