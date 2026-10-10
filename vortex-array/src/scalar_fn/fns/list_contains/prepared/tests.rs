// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::BitBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::PreparedSet;
use super::PreparedSetArray;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::ScalarFn;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::assert_arrays_eq;
use crate::builders::builder_with_capacity_in;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::expr::list_contains;
use crate::expr::lit;
use crate::expr::root;
use crate::match_each_decimal_value_type;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::list_contains::PreparedSetData;
use crate::validity::Validity;

/// Prepares the non-null list scalar `list`.
fn prepare(list: &Scalar) -> VortexResult<PreparedSetData> {
    PreparedSetData::try_new(list.clone())
}

/// Prepares the non-null list scalar `list` as a set repeated `len` times.
fn prepare_array(list: &Scalar, len: usize) -> VortexResult<ArrayRef> {
    Ok(PreparedSetArray::try_new(list.clone(), len)?.into_array())
}

/// A non-null list scalar of the rows of `elements`.
fn list_scalar(elements: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    let values = (0..elements.len())
        .map(|idx| elements.execute_scalar(idx, ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(Scalar::list(
        elements.dtype().clone(),
        values,
        Nullability::NonNullable,
    ))
}

fn nested_needles() -> ArrayRef {
    ListArray::try_new(
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(2), None, Some(9)]).into_array(),
        buffer![0u32, 2, 2, 4, 5].into_array(),
        Validity::from(BitBuffer::from_iter([true, false, true, true])),
    )
    .unwrap()
    .into_array()
}

fn map_needles() -> ArrayRef {
    let ctx = array_session().create_execution_ctx();
    let dtype = DType::map(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable),
        false,
        Nullability::Nullable,
    )
    .unwrap();
    let mut builder = builder_with_capacity_in(&dtype, 4, ctx.allocator());
    for key in [Some(1i32), None, Some(2), Some(9)] {
        let scalar = match key {
            Some(key) => Scalar::map(
                dtype.clone(),
                [(key.into(), Scalar::null(DType::Utf8(Nullability::Nullable)))],
            ),
            None => Scalar::null(dtype.clone()),
        };
        builder.append_scalar(&scalar).unwrap();
    }
    builder.finish()
}

fn struct_needles() -> ArrayRef {
    StructArray::from_fields(&[("list", nested_needles())])
        .unwrap()
        .into_array()
}

#[rstest]
#[case::list(nested_needles())]
#[case::map(map_needles())]
#[case::struct_of_lists(struct_needles())]
#[case::fixed_size_list(FixedSizeListArray::new(
    PrimitiveArray::from_option_iter([
        Some(1i32), None, Some(2), None, Some(9), None, Some(8), None,
    ]).into_array(),
    2, Validity::NonNullable, 4,
).into_array())]
fn test_row_set_returns_membership_bits(
    #[case] needles: ArrayRef,
    #[values(false, true)] sql_null_semantics: bool,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = needles.dtype().as_nullable();
    // Unsorted, repeated members and a top-level null exercise normalization and SQL's
    // unknown non-match independently of any nulls nested inside the members.
    let members = [2, 0, 2]
        .map(|idx| needles.execute_scalar(idx, &mut ctx)?.cast(&dtype))
        .into_iter()
        .collect::<VortexResult<Vec<_>>>()?;
    let mut elements = members.clone();
    elements.push(Scalar::null(dtype.clone()));
    let list = Scalar::list(dtype, elements, Nullability::NonNullable);
    let options = ListContainsOptions { sql_null_semantics };
    let set = prepare(&list)?;
    let result = set.contains(&needles, &options, &mut ctx)?;
    // The old fallback returned a lazy OR tree here.
    assert!(result.is::<Bool>());
    let expected = (0..needles.len())
        .map(|idx| {
            let value = needles.execute_scalar(idx, &mut ctx)?;
            Ok(if value.is_null() {
                None
            } else if members.contains(&value) {
                Some(true)
            } else if sql_null_semantics {
                None
            } else {
                Some(false)
            })
        })
        .collect::<VortexResult<Vec<_>>>()?;

    // The list is not null, so only a null needle can make the result null. Under SQL null
    // semantics the null element also can, by leaving a non-match unknown.
    let nullability = needles.dtype().nullability() | Nullability::from(sql_null_semantics);
    assert_eq!(result.dtype(), &DType::Bool(nullability));

    let validity = match nullability {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => Validity::from_iter(expected.iter().map(Option::is_some)),
    };
    let expected = BoolArray::new(
        BitBuffer::from_iter(expected.into_iter().map(Option::unwrap_or_default)),
        validity,
    );
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[rstest]
fn test_decimal_bitmap_across_storage_widths(
    #[values(2, 4, 9, 18, 38, 76)] precision: u8,
    #[values(
        DecimalType::I8,
        DecimalType::I16,
        DecimalType::I32,
        DecimalType::I64,
        DecimalType::I128,
        DecimalType::I256
    )]
    needle_width: DecimalType,
    #[values(false, true)] sql_null_semantics: bool,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let decimal = DecimalDType::new(precision, 1);
    let dtype = DType::Decimal(decimal, Nullability::Nullable);
    let mut elements: Vec<_> = [1i8, -2, 1]
        .map(|value| Scalar::decimal(value.into(), decimal, Nullability::Nullable))
        .into();
    elements.push(Scalar::null(dtype.clone()));
    let list = Scalar::list(dtype, elements, Nullability::NonNullable);
    let needles = match_each_decimal_value_type!(needle_width, |T| {
        DecimalArray::from_option_iter::<T, _>(
            [Some(-2i8), Some(0), Some(1), None]
                .map(|value| value.map(|value| DecimalValue::from(value).cast::<T>().unwrap())),
            decimal,
        )
        .into_array()
    });
    let options = ListContainsOptions { sql_null_semantics };
    let set = prepare(&list)?;
    assert!(set.element_set(&mut ctx)?.probe.is_bitmap());
    let non_match = (!sql_null_semantics).then_some(false);
    assert_arrays_eq!(
        set.contains(&needles, &options, &mut ctx)?,
        BoolArray::from_iter([Some(true), non_match, Some(true), None]),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::dense_i128(38, i256::from_i128(1i128 << 100), true)]
#[case::sparse_i128(38, i256::from_i128(1i128 << 100), false)]
#[case::dense_i256(76, i256::from_parts(0, 1i128 << 72), true)]
#[case::sparse_i256(76, i256::from_parts(0, 1i128 << 72), false)]
fn test_decimal_wide_values(
    #[case] precision: u8,
    #[case] base: i256,
    #[case] dense: bool,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let decimal = DecimalDType::new(precision, 2);
    let second = if dense { base + i256::ONE } else { -base };
    let list = Scalar::list(
        DType::Decimal(decimal, Nullability::NonNullable),
        [second, base, second]
            .map(|v| Scalar::decimal(v.into(), decimal, Nullability::NonNullable))
            .into(),
        Nullability::NonNullable,
    );
    // The last non-null value shares the low 64 bits of a member but must not match it.
    let needles = DecimalArray::from_option_iter::<i256, _>(
        [
            Some(base),
            Some(second),
            Some(base - i256::ONE),
            Some(i256::ZERO),
            Some(base + i256::from_i128(1i128 << 64)),
            None,
        ],
        decimal,
    )
    .into_array();
    let set = prepare(&list)?;
    assert_eq!(set.element_set(&mut ctx)?.probe.is_bitmap(), dense);
    assert_arrays_eq!(
        set.contains(&needles, &ListContainsOptions::default(), &mut ctx)?,
        BoolArray::from_iter([
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            None,
        ]),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::bitmap(
    PrimitiveArray::from_option_iter([Some(5i64), Some(-3), Some(5)]).into_array(),
    PrimitiveArray::from_option_iter([Some(-3i64), Some(4), Some(i64::MIN), None]).into_array(),
    true,
    [Some(true), Some(false), Some(false), None],
)]
#[case::unsigned_in_signed_order(
    PrimitiveArray::from_option_iter([Some(u64::MAX), Some(0), Some(1 << 40)]).into_array(),
    PrimitiveArray::from_option_iter([Some(1u64 << 40), Some(1), Some(u64::MAX), Some(0)])
        .into_array(),
    false,
    [Some(true), Some(false), Some(true), Some(true)],
)]
#[case::float_bits(
    PrimitiveArray::from_option_iter([Some(0.0f64), Some(f64::NAN)]).into_array(),
    PrimitiveArray::from_option_iter([Some(-0.0f64), Some(0.0), Some(f64::NAN), Some(1.5)])
        .into_array(),
    false,
    [Some(false), Some(true), Some(true), Some(false)],
)]
fn test_primitive_integers(
    #[case] elements: ArrayRef,
    #[case] needles: ArrayRef,
    #[case] dense: bool,
    #[case] expected: [Option<bool>; 4],
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let set = PreparedSetData::try_new(list_scalar(&elements, &mut ctx)?)?;
    assert_eq!(set.element_set(&mut ctx)?.probe.is_bitmap(), dense);
    assert_arrays_eq!(
        set.contains(&needles, &ListContainsOptions::default(), &mut ctx)?,
        BoolArray::from_iter(expected),
        &mut ctx
    );
    Ok(())
}

/// The set `{2, null}` of nullable `i32`.
fn set_with_null() -> Scalar {
    let element = DType::Primitive(PType::I32, Nullability::Nullable);
    Scalar::list(
        element.clone(),
        vec![
            Scalar::primitive(2i32, Nullability::Nullable),
            Scalar::null(element),
        ],
        Nullability::NonNullable,
    )
}

#[test]
fn test_prepared_set_rows_are_the_constant_list() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let set = prepare_array(&set_with_null(), 4)?;

    assert_arrays_eq!(set, ConstantArray::new(set_with_null(), 4), &mut ctx);

    // A slice keeps the probe instead of building it again.
    let sliced = set.slice(1..3)?;
    assert!(sliced.is::<PreparedSet>());
    assert_eq!(sliced.len(), 2);
    Ok(())
}

#[rstest]
#[case::default(ListContainsOptions::default(), [Some(false), Some(true), None, Some(false)])]
#[case::sql(
    ListContainsOptions { sql_null_semantics: true },
    [None, Some(true), None, None]
)]
fn test_list_contains_probes_a_prepared_set_list(
    #[case] options: ListContainsOptions,
    #[case] expected: [Option<bool>; 4],
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let set = prepare_array(&set_with_null(), 4)?;
    let needles =
        PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, Some(3)]).into_array();

    let result = ListContains::try_new_opts(set, needles, options)?.into_array();
    assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);
    Ok(())
}

#[rstest]
#[case::default(ListContainsOptions::default(), Some(false))]
#[case::sql(ListContainsOptions { sql_null_semantics: true }, None)]
fn test_constant_needle_gives_a_constant(
    #[case] options: ListContainsOptions,
    #[case] expected: Option<bool>,
) -> VortexResult<()> {
    // `3 IN (2, NULL)` against the prepared set is looked up once, and gives a constant.
    let mut ctx = array_session().create_execution_ctx();
    let set = prepare_array(&set_with_null(), 4)?;
    let needle = ConstantArray::new(Scalar::primitive(3i32, Nullability::Nullable), 4).into_array();

    let result = ListContains::try_new_opts(set, needle, options)?
        .into_array()
        .execute::<ArrayRef>(&mut ctx)?;
    let expected = match expected {
        Some(value) => Scalar::bool(value, Nullability::Nullable),
        None => Scalar::null(DType::Bool(Nullability::Nullable)),
    };
    assert_eq!(result.as_constant(), Some(expected));
    Ok(())
}

#[test]
fn test_constant_row_needle_probes_one_row() -> VortexResult<()> {
    // A probe of rows cannot look a scalar needle up, so execution probes one row.
    let mut ctx = array_session().create_execution_ctx();
    let needles = nested_needles();
    let dtype = needles.dtype().as_nullable();
    let member = needles.execute_scalar(2, &mut ctx)?.cast(&dtype)?;
    let list = Scalar::list(dtype, vec![member.clone()], Nullability::NonNullable);

    let set = prepare_array(&list, 4)?;
    let needle = ConstantArray::new(member, 4).into_array();
    let result = ListContains::try_new_opts(set, needle, ListContainsOptions::default())?
        .into_array()
        .execute::<ArrayRef>(&mut ctx)?;
    assert_eq!(
        result.as_constant(),
        Some(Scalar::bool(true, Nullability::Nullable))
    );
    Ok(())
}

#[rstest]
#[case::default(ListContainsOptions::default(), [Some(true), Some(false), None])]
#[case::sql(ListContainsOptions { sql_null_semantics: true }, [Some(true), None, None])]
fn test_result_from_bits_applies_null_semantics(
    #[case] options: ListContainsOptions,
    #[case] expected: [Option<bool>; 3],
) -> VortexResult<()> {
    // Against `{2, null}`: a match, a non-match, and a null needle whose bit has no effect.
    let mut ctx = array_session().create_execution_ctx();
    let set = prepare(&set_with_null())?;
    let needle_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
    let bits = BitBuffer::from_iter([true, false, true]);
    let validity = Validity::from_iter([true, true, false]);

    let result = set.result_from_bits(bits, validity, &needle_dtype, &options)?;
    assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);
    Ok(())
}

#[test]
fn test_result_from_bits_rejects_mismatched_needles() -> VortexResult<()> {
    let set = prepare(&set_with_null())?;
    let options = ListContainsOptions::default();
    let bits = BitBuffer::from_iter([true, false]);

    // A validity with a null keeps its row count; an all-valid one folds to `AllValid`, which
    // has none to check.
    let long_validity = Validity::from_iter([true, false, true]);
    let needle_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
    assert!(
        set.result_from_bits(bits.clone(), long_validity, &needle_dtype, &options)
            .is_err()
    );

    let wrong_dtype = DType::Primitive(PType::I64, Nullability::Nullable);
    assert!(
        set.result_from_bits(bits, Validity::AllValid, &wrong_dtype, &options)
            .is_err()
    );
    Ok(())
}

#[test]
fn test_bytes_set_short_and_long_views() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements = VarBinViewArray::from_iter_nullable_str([
        Some("ab"),
        Some(""),
        Some("a long string with suffix A"),
        Some("another long string"),
        None,
        Some("ab"),
    ])
    .into_array();
    let set = PreparedSetData::try_new(list_scalar(&elements, &mut ctx)?)?;

    // "abc" shares the prefix of a short element, and "... suffix B" shares the head of a long
    // one: length and first 4 bytes. Neither is an element.
    let needles = VarBinViewArray::from_iter_nullable_str([
        Some("ab"),
        Some("abc"),
        Some(""),
        Some("a long string with suffix B"),
        Some("another long string"),
        Some("a string never in the set"),
        None,
    ])
    .into_array();

    assert_arrays_eq!(
        set.contains(&needles, &ListContainsOptions::default(), &mut ctx)?,
        BoolArray::from_iter([
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            None,
        ]),
        &mut ctx
    );
    Ok(())
}

fn chunked_needles() -> VortexResult<ArrayRef> {
    Ok(ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_option_iter([Some(1i32), Some(2)]).into_array(),
            PrimitiveArray::from_option_iter::<i32, _>([]).into_array(),
            PrimitiveArray::from_option_iter([None, Some(3), Some(2)]).into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::Nullable),
    )?
    .into_array())
}

/// The set that the list of a `list_contains` node holds.
fn list_set(node: &ArrayRef) -> PreparedSetData {
    node.as_::<ScalarFn>()
        .get_child(0)
        .as_::<PreparedSet>()
        .data()
        .clone()
}

/// Asserts that `array` was pushed into its chunks, and that every non-empty chunk probes one set.
fn assert_chunks_share_one_set(array: &ArrayRef) {
    let chunked = array.as_::<Chunked>();
    let sets: Vec<_> = chunked
        .iter_chunks()
        .filter(|chunk| !chunk.is_empty())
        .map(list_set)
        .collect();
    assert!(sets.len() > 1);
    assert!(sets.iter().all(|set| set.shares_set_with(&sets[0])));
}

#[rstest]
#[case::default(
    ListContainsOptions::default(),
    [Some(false), Some(true), None, Some(false), Some(true)],
)]
#[case::sql(
    ListContainsOptions { sql_null_semantics: true },
    [None, Some(true), None, None, Some(true)],
)]
fn test_constant_list_over_chunked_needles(
    #[case] options: ListContainsOptions,
    #[case] expected: [Option<bool>; 5],
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let needles = chunked_needles()?;
    let list = ConstantArray::new(set_with_null(), needles.len()).into_array();

    // The list is prepared before the push-down, so every chunk gets a slice of the one set.
    let array = ListContains::try_new_opts(list, needles, options)?
        .into_array()
        .optimize()?;
    assert_chunks_share_one_set(&array);

    assert_arrays_eq!(array, BoolArray::from_iter(expected), &mut ctx);
    Ok(())
}

#[test]
fn test_optimized_expression_over_chunked_needles() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let needles = chunked_needles()?;
    let expr = list_contains(lit(set_with_null()), root())
        .bind(needles.dtype())?
        .optimize()?;

    let array = needles.apply_bound(&expr)?;
    assert_chunks_share_one_set(&array);

    assert_arrays_eq!(
        array,
        BoolArray::from_iter([Some(false), Some(true), None, Some(false), Some(true)]),
        &mut ctx
    );
    Ok(())
}

/// Dictionary needles with a null code, against a set that is empty or not.
#[rstest]
#[case::default(ListContainsOptions::default(), set_with_null(), [Some(true), Some(false), None, Some(true)])]
#[case::sql(ListContainsOptions { sql_null_semantics: true }, set_with_null(), [Some(true), None, None, Some(true)])]
// Off SQL null semantics an empty list answers `false` even for the null code.
#[case::empty(ListContainsOptions::default(), empty_i32_set(), [Some(false); 4])]
#[case::empty_sql(ListContainsOptions { sql_null_semantics: true }, empty_i32_set(), [Some(false), Some(false), None, Some(false)])]
fn test_dictionary_needles_with_null_codes(
    #[case] options: ListContainsOptions,
    #[case] set: Scalar,
    #[case] expected: [Option<bool>; 4],
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let needles = DictArray::try_new(
        PrimitiveArray::from_option_iter([Some(0u8), Some(1), None, Some(0)]).into_array(),
        buffer![2i32, 3].into_array(),
    )?
    .into_array();
    let list = ConstantArray::new(set, needles.len()).into_array();

    let array = ListContains::try_new_opts(list, needles, options)?
        .into_array()
        .optimize()?;
    // The rule took the node: it is a dictionary of answers, or a constant for the empty list.
    assert!(!array.is::<ScalarFn>());

    assert_arrays_eq!(array, BoolArray::from_iter(expected), &mut ctx);
    Ok(())
}

/// The empty set of nullable `i32`.
fn empty_i32_set() -> Scalar {
    Scalar::list(
        DType::Primitive(PType::I32, Nullability::Nullable),
        vec![],
        Nullability::NonNullable,
    )
}

#[test]
fn test_constant_list_over_dictionary_needles() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let needles = DictArray::try_new(
        buffer![0u8, 1, 0, 2, 1].into_array(),
        PrimitiveArray::from_option_iter([Some(2i32), Some(3), None]).into_array(),
    )?
    .into_array();
    let list = ConstantArray::new(set_with_null(), needles.len()).into_array();

    // The prepared list still lets the function into the values of the dictionary.
    let array = ListContains::try_new(list, needles)?
        .into_array()
        .optimize()?;
    let values = array.as_::<Dict>().values().clone();
    assert!(values.as_::<ScalarFn>().get_child(0).is::<PreparedSet>());

    assert_arrays_eq!(
        array,
        BoolArray::from_iter([Some(true), Some(false), Some(true), None, Some(false)]),
        &mut ctx
    );
    Ok(())
}
