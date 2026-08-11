// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_buffer::BitBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::assert_arrays_eq;
use crate::builders::ArrayBuilder;
use crate::builders::MapBuilder;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::extension::datetime::TimestampOptions;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::scalar_cmp;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;
use crate::test_harness::to_int_indices;
use crate::validity::Validity;

#[test]
fn test_bool_basic_comparisons() {
    let ctx = &mut array_session().create_execution_ctx();
    let arr = BoolArray::new(
        BitBuffer::from_iter([true, true, false, true, false]),
        Validity::from_iter([false, true, true, true, true]),
    );

    let matches = arr
        .clone()
        .into_array()
        .binary(arr.clone().into_array(), Operator::Eq)
        .unwrap()
        .execute::<BoolArray>(ctx)
        .vortex_expect("must be a bool array");
    assert_eq!(to_int_indices(matches, ctx).unwrap(), [1u64, 2, 3, 4]);

    let matches = arr
        .clone()
        .into_array()
        .binary(arr.clone().into_array(), Operator::NotEq)
        .unwrap()
        .execute::<BoolArray>(ctx)
        .vortex_expect("must be a bool array");
    let empty: [u64; 0] = [];
    assert_eq!(to_int_indices(matches, ctx).unwrap(), empty);

    let other = BoolArray::new(
        BitBuffer::from_iter([false, false, false, true, true]),
        Validity::from_iter([false, true, true, true, true]),
    );

    let matches = arr
        .clone()
        .into_array()
        .binary(other.clone().into_array(), Operator::Lte)
        .unwrap()
        .execute::<BoolArray>(ctx)
        .vortex_expect("must be a bool array");
    assert_eq!(to_int_indices(matches, ctx).unwrap(), [2u64, 3, 4]);

    let matches = arr
        .clone()
        .into_array()
        .binary(other.clone().into_array(), Operator::Lt)
        .unwrap()
        .execute::<BoolArray>(ctx)
        .vortex_expect("must be a bool array");
    assert_eq!(to_int_indices(matches, ctx).unwrap(), [4u64]);

    let matches = other
        .clone()
        .into_array()
        .binary(arr.clone().into_array(), Operator::Gte)
        .unwrap()
        .execute::<BoolArray>(ctx)
        .vortex_expect("must be a bool array");
    assert_eq!(to_int_indices(matches, ctx).unwrap(), [2u64, 3, 4]);

    let matches = other
        .into_array()
        .binary(arr.into_array(), Operator::Gt)
        .unwrap()
        .execute::<BoolArray>(ctx)
        .vortex_expect("must be a bool array");
    assert_eq!(to_int_indices(matches, ctx).unwrap(), [4u64]);
}

#[test]
fn constant_compare() {
    let left = ConstantArray::new(Scalar::from(2u32), 10);
    let right = ConstantArray::new(Scalar::from(10u32), 10);

    let result = left
        .into_array()
        .binary(right.into_array(), Operator::Gt)
        .unwrap();
    assert_eq!(result.len(), 10);
    let scalar = result
        .execute_scalar(0, &mut array_session().create_execution_ctx())
        .unwrap();
    assert_eq!(scalar.as_bool().value(), Some(false));
}

#[rstest]
#[case(VarBinArray::from(vec!["a", "b"]).into_array(), VarBinViewArray::from_iter_str(["a", "b"]).into_array())]
#[case(VarBinViewArray::from_iter_str(["a", "b"]).into_array(), VarBinArray::from(vec!["a", "b"]).into_array())]
#[case(VarBinArray::from(vec!["a".as_bytes(), "b".as_bytes()]).into_array(), VarBinViewArray::from_iter_bin(["a".as_bytes(), "b".as_bytes()]).into_array())]
#[case(VarBinViewArray::from_iter_bin(["a".as_bytes(), "b".as_bytes()]).into_array(), VarBinArray::from(vec!["a".as_bytes(), "b".as_bytes()]).into_array())]
fn compare_different_encodings(#[case] left: ArrayRef, #[case] right: ArrayRef) {
    let mut ctx = array_session().create_execution_ctx();
    let res = left.binary(right, Operator::Eq).unwrap();
    let expected = BoolArray::from_iter([true, true]);
    assert_arrays_eq!(res, expected, &mut ctx);
}

#[test]
fn test_list_array_comparison() {
    let mut ctx = array_session().create_execution_ctx();
    let values1 = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]);
    let offsets1 = PrimitiveArray::from_iter([0i32, 2, 4, 6]);
    let list1 = ListArray::try_new(
        values1.into_array(),
        offsets1.into_array(),
        Validity::NonNullable,
    )
    .unwrap();

    let values2 = PrimitiveArray::from_iter([1i32, 2, 3, 4, 7, 8]);
    let offsets2 = PrimitiveArray::from_iter([0i32, 2, 4, 6]);
    let list2 = ListArray::try_new(
        values2.into_array(),
        offsets2.into_array(),
        Validity::NonNullable,
    )
    .unwrap();

    let result = list1
        .clone()
        .into_array()
        .binary(list2.clone().into_array(), Operator::Eq)
        .unwrap();
    let expected = BoolArray::from_iter([true, true, false]);
    assert_arrays_eq!(result, expected, &mut ctx);

    let result = list1
        .clone()
        .into_array()
        .binary(list2.clone().into_array(), Operator::NotEq)
        .unwrap();
    let expected = BoolArray::from_iter([false, false, true]);
    assert_arrays_eq!(result, expected, &mut ctx);

    let result = list1
        .into_array()
        .binary(list2.into_array(), Operator::Lt)
        .unwrap();
    let expected = BoolArray::from_iter([false, false, true]);
    assert_arrays_eq!(result, expected, &mut ctx);
}

#[test]
fn test_list_array_constant_comparison() {
    let mut ctx = array_session().create_execution_ctx();
    let values = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]);
    let offsets = PrimitiveArray::from_iter([0i32, 2, 4, 6]);
    let list = ListArray::try_new(
        values.into_array(),
        offsets.into_array(),
        Validity::NonNullable,
    )
    .unwrap();

    let list_scalar = Scalar::list(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        vec![3i32.into(), 4i32.into()],
        Nullability::NonNullable,
    );
    let constant = ConstantArray::new(list_scalar, 3);

    let result = list
        .into_array()
        .binary(constant.into_array(), Operator::Eq)
        .unwrap();
    let expected = BoolArray::from_iter([false, true, false]);
    assert_arrays_eq!(result, expected, &mut ctx);
}

#[test]
fn test_struct_array_comparison() {
    let mut ctx = array_session().create_execution_ctx();
    let bool_field1 = BoolArray::from_iter([Some(true), Some(false), Some(true)]);
    let int_field1 = PrimitiveArray::from_iter([1i32, 2, 3]);

    let bool_field2 = BoolArray::from_iter([Some(true), Some(false), Some(false)]);
    let int_field2 = PrimitiveArray::from_iter([1i32, 2, 4]);

    let struct1 = StructArray::from_fields(&[
        ("bool_col", bool_field1.into_array()),
        ("int_col", int_field1.into_array()),
    ])
    .unwrap();

    let struct2 = StructArray::from_fields(&[
        ("bool_col", bool_field2.into_array()),
        ("int_col", int_field2.into_array()),
    ])
    .unwrap();

    let result = struct1
        .clone()
        .into_array()
        .binary(struct2.clone().into_array(), Operator::Eq)
        .unwrap();
    let expected = BoolArray::from_iter([true, true, false]);
    assert_arrays_eq!(result, expected, &mut ctx);

    let result = struct1
        .into_array()
        .binary(struct2.into_array(), Operator::Gt)
        .unwrap();
    let expected = BoolArray::from_iter([false, false, true]);
    assert_arrays_eq!(result, expected, &mut ctx);
}

#[test]
fn test_empty_struct_compare() {
    let mut ctx = array_session().create_execution_ctx();
    let empty1 = StructArray::try_new(
        FieldNames::from(Vec::<FieldName>::new()),
        Vec::new(),
        5,
        Validity::NonNullable,
    )
    .unwrap();

    let empty2 = StructArray::try_new(
        FieldNames::from(Vec::<FieldName>::new()),
        Vec::new(),
        5,
        Validity::NonNullable,
    )
    .unwrap();

    let result = empty1
        .into_array()
        .binary(empty2.into_array(), Operator::Eq)
        .unwrap();
    let expected = BoolArray::from_iter([true, true, true, true, true]);
    assert_arrays_eq!(result, expected, &mut ctx);
}

/// Regression test: comparing struct arrays where the same logical field is backed by
/// different Vortex encodings (VarBinArray vs VarBinViewArray) must not panic.
#[test]
fn struct_compare_mixed_binary_encodings() {
    let mut ctx = array_session().create_execution_ctx();
    // LHS: struct with a VarBinArray (offset-based) binary field
    let bin_field1 = VarBinArray::from(vec![
        "apple".as_bytes(),
        "banana".as_bytes(),
        "cherry".as_bytes(),
    ]);
    let struct1 = StructArray::from_fields(&[("data", bin_field1.into_array())]).unwrap();

    // RHS: struct with a VarBinViewArray (view-based) binary field — same logical DType
    let bin_field2 = VarBinViewArray::from_iter_bin([
        "apple".as_bytes(),
        "banana".as_bytes(),
        "durian".as_bytes(),
    ]);
    let struct2 = StructArray::from_fields(&[("data", bin_field2.into_array())]).unwrap();

    let result = struct1
        .into_array()
        .binary(struct2.into_array(), Operator::Eq)
        .unwrap();
    let expected = BoolArray::from_iter([true, true, false]);
    assert_arrays_eq!(result, expected, &mut ctx);
}

/// Regression test: `scalar_cmp` must error when comparing scalars with incompatible
/// extension types (e.g., timestamps with different time units) rather than silently
/// returning a wrong result.
#[test]
fn scalar_cmp_incompatible_extension_types_errors() {
    let ms_scalar = Scalar::extension::<Timestamp>(
        TimestampOptions {
            unit: TimeUnit::Milliseconds,
            tz: None,
        },
        Scalar::from(1704067200000i64),
    );
    let s_scalar = Scalar::extension::<Timestamp>(
        TimestampOptions {
            unit: TimeUnit::Seconds,
            tz: None,
        },
        Scalar::from(1704067200i64),
    );

    // Ordering comparisons must error on incompatible types.
    assert!(scalar_cmp(&ms_scalar, &s_scalar, CompareOperator::Gt).is_err());
    assert!(scalar_cmp(&ms_scalar, &s_scalar, CompareOperator::Lt).is_err());
    assert!(scalar_cmp(&ms_scalar, &s_scalar, CompareOperator::Gte).is_err());
    assert!(scalar_cmp(&ms_scalar, &s_scalar, CompareOperator::Lte).is_err());
    assert!(scalar_cmp(&ms_scalar, &s_scalar, CompareOperator::Eq).is_err());
    assert!(scalar_cmp(&ms_scalar, &s_scalar, CompareOperator::NotEq).is_err());
}

#[test]
fn test_empty_list() {
    let ctx = &mut array_session().create_execution_ctx();
    let list = ListViewArray::new(
        BoolArray::from_iter(Vec::<bool>::new()).into_array(),
        buffer![0i32, 0i32, 0i32].into_array(),
        buffer![0i32, 0i32, 0i32].into_array(),
        Validity::AllValid,
    );

    let result = list
        .clone()
        .into_array()
        .binary(list.into_array(), Operator::Eq)
        .unwrap();
    assert!(result.execute_scalar(0, ctx).unwrap().is_valid());
    assert!(result.execute_scalar(1, ctx).unwrap().is_valid());
    assert!(result.execute_scalar(2, ctx).unwrap().is_valid());
}

fn execute_compare_test(lhs: ArrayRef, rhs: ArrayRef, op: Operator) -> ArrayRef {
    lhs.binary(rhs, op).unwrap()
}

#[rstest]
#[case(Operator::Eq, [false, true, false, false])]
#[case(Operator::NotEq, [true, false, true, true])]
#[case(Operator::Lt, [true, false, false, true])]
#[case(Operator::Lte, [true, true, false, true])]
#[case(Operator::Gt, [false, false, true, false])]
#[case(Operator::Gte, [false, true, true, false])]
fn int_all_operators(#[case] op: Operator, #[case] expected: [bool; 4]) {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = buffer![1i32, 5, 9, 2].into_array();
    let rhs = buffer![3i32, 5, 7, 4].into_array();
    let result = execute_compare_test(lhs, rhs, op);
    assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);
}

#[rstest]
#[case(Operator::Lt, [Some(true), None, Some(false), None])]
#[case(Operator::Eq, [Some(false), None, Some(false), None])]
fn int_nullable(#[case] op: Operator, #[case] expected: [Option<bool>; 4]) {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = PrimitiveArray::from_option_iter([Some(1i64), None, Some(9), Some(2)]).into_array();
    let rhs = PrimitiveArray::from_option_iter([Some(3i64), Some(5), Some(7), None]).into_array();
    let result = execute_compare_test(lhs, rhs, op);
    assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);
}

#[rstest]
#[case(Operator::Gt, [false, false, true, false])]
#[case(Operator::Lte, [true, true, false, true])]
fn int_constant_lhs_and_rhs(#[case] op: Operator, #[case] expected: [bool; 4]) {
    let mut ctx = array_session().create_execution_ctx();
    let array = buffer![1i32, 5, 9, 2].into_array();
    let constant = ConstantArray::new(5i32, 4).into_array();

    let result = execute_compare_test(array.clone(), constant.clone(), op);
    assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);

    // The swapped form must produce the swapped result.
    let swapped = execute_compare_test(constant, array, op);
    let expected_swapped: Vec<bool> = match op {
        Operator::Gt => vec![true, false, false, true],
        Operator::Lte => vec![false, true, true, false],
        _ => unreachable!(),
    };
    assert_arrays_eq!(swapped, BoolArray::from_iter(expected_swapped), &mut ctx);
}

/// Floats compare with Vortex's total ordering: NaN is the largest value, equality is bitwise,
/// and -0.0 < +0.0. This matches `Scalar` comparison semantics.
#[test]
fn float_total_order() {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = buffer![f64::NAN, f64::NAN, -0.0f64, 1.0].into_array();
    let rhs = buffer![f64::NAN, f64::INFINITY, 0.0f64, f64::NAN].into_array();

    let result = execute_compare_test(lhs.clone(), rhs.clone(), Operator::Eq);
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([true, false, false, false]),
        &mut ctx
    );

    let result = execute_compare_test(lhs, rhs, Operator::Lt);
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([false, false, true, true]),
        &mut ctx
    );
}

#[rstest]
#[case(Operator::Eq, [true, false, true, true])]
#[case(Operator::Lt, [false, true, false, false])]
#[case(Operator::Gte, [true, false, true, true])]
fn bool_constant(#[case] op: Operator, #[case] expected: [bool; 4]) {
    let mut ctx = array_session().create_execution_ctx();
    let array = BoolArray::from_iter([true, false, true, true]).into_array();
    let constant = ConstantArray::new(true, 4).into_array();
    let result = execute_compare_test(array, constant, op);
    assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);
}

#[rstest]
// Inlined vs inlined, decided by prefix.
#[case("bad", "bat", Operator::Lt, true)]
// Inlined prefix tie decided by the tail bytes.
#[case("abcdefgh", "abcdefgi", Operator::Lt, true)]
// Inlined prefix and tail tie decided by length.
#[case("abc", "abcd", Operator::Lt, true)]
// Out-of-line values with equal prefixes.
#[case("aaaaaaaaaaaaaaaaaaaab", "aaaaaaaaaaaaaaaaaaaac", Operator::Lt, true)]
// Inlined vs out-of-line where one is a prefix of the other.
#[case("aaaa", "aaaaaaaaaaaaaaaaaaaa", Operator::Lt, true)]
// Equality across the inlined/out-of-line boundary.
#[case("aaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaa", Operator::Eq, true)]
#[case("aaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaab", Operator::Eq, false)]
// Embedded NUL: "a\0" > "a" even though the padded prefixes tie.
#[case("a\0", "a", Operator::Gt, true)]
fn string_compare_cases(
    #[case] lhs: &str,
    #[case] rhs: &str,
    #[case] op: Operator,
    #[case] expected: bool,
) {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = VarBinViewArray::from_iter_str([lhs]).into_array();
    let rhs = VarBinViewArray::from_iter_str([rhs]).into_array();
    let result = execute_compare_test(lhs, rhs, op);
    assert_arrays_eq!(result, BoolArray::from_iter([expected]), &mut ctx);
}

#[test]
fn string_constant_compare() {
    let mut ctx = array_session().create_execution_ctx();
    let array = VarBinViewArray::from_iter_str([
        "apple",
        "banana",
        "banan",
        "bananarama-bananarama",
        "cherry",
    ])
    .into_array();
    let constant = ConstantArray::new(Scalar::from("banana"), 5).into_array();

    let result = execute_compare_test(array.clone(), constant.clone(), Operator::Eq);
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([false, true, false, false, false]),
        &mut ctx
    );

    let result = execute_compare_test(array.clone(), constant.clone(), Operator::Lt);
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([true, false, true, false, false]),
        &mut ctx
    );

    let result = execute_compare_test(constant, array, Operator::Lt);
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([false, false, false, true, true]),
        &mut ctx
    );
}

#[test]
fn string_constant_longer_than_inline() {
    let mut ctx = array_session().create_execution_ctx();
    let array = VarBinViewArray::from_iter_str(["short", "averyveryverylongstring", "averyvery"])
        .into_array();
    let constant = ConstantArray::new(Scalar::from("averyveryverylongstring"), 3).into_array();

    let result = execute_compare_test(array.clone(), constant.clone(), Operator::Eq);
    assert_arrays_eq!(result, BoolArray::from_iter([false, true, false]), &mut ctx);

    let result = execute_compare_test(array, constant, Operator::Lte);
    assert_arrays_eq!(result, BoolArray::from_iter([false, true, true]), &mut ctx);
}

#[test]
fn decimal_compare() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs = DecimalArray::from_iter::<i64, _>([100, 250, 300], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i64, _>([200, 250, 100], dtype).into_array();

    let result = execute_compare_test(lhs, rhs, Operator::Lt);
    assert_arrays_eq!(result, BoolArray::from_iter([true, false, false]), &mut ctx);
}

/// Two decimal arrays with the same logical dtype but different storage widths compare through
/// the widened common storage type.
#[test]
fn decimal_compare_mixed_storage_widths() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs = DecimalArray::from_iter::<i32, _>([100, 250, 300], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i128, _>([200, 250, 100], dtype).into_array();

    let result = execute_compare_test(lhs, rhs, Operator::Lte);
    assert_arrays_eq!(result, BoolArray::from_iter([true, true, false]), &mut ctx);
}

/// A decimal constant that does not fit the array's narrow storage type still compares
/// correctly: it is greater than every representable array value.
#[test]
fn decimal_constant_out_of_storage_range() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(20, 2);
    let array = DecimalArray::from_iter::<i8, _>([1, 50, 100], dtype).into_array();
    let constant = ConstantArray::new(
        Scalar::decimal(
            DecimalValue::from(10_000_000i64),
            dtype,
            Nullability::NonNullable,
        ),
        3,
    )
    .into_array();

    let result = execute_compare_test(array.clone(), constant.clone(), Operator::Lt);
    assert_arrays_eq!(result, BoolArray::from_iter([true, true, true]), &mut ctx);

    let result = execute_compare_test(array, constant, Operator::Gte);
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([false, false, false]),
        &mut ctx
    );
}

/// Extension arrays compare through their storage values.
#[test]
fn extension_timestamp_compare() {
    let mut ctx = array_session().create_execution_ctx();
    let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
    let lhs = ExtensionArray::new(ext_dtype.clone(), buffer![1000i64, 2000, 3000].into_array())
        .into_array();
    let rhs =
        ExtensionArray::new(ext_dtype, buffer![1500i64, 2000, 2500].into_array()).into_array();

    let result = execute_compare_test(lhs.clone(), rhs.clone(), Operator::Lt);
    assert_arrays_eq!(result, BoolArray::from_iter([true, false, false]), &mut ctx);

    let result = execute_compare_test(lhs, rhs, Operator::Eq);
    assert_arrays_eq!(result, BoolArray::from_iter([false, true, false]), &mut ctx);
}

/// Comparing extension arrays with different extension dtypes (e.g. timestamps in different
/// units) must error rather than silently comparing raw storage values.
#[test]
fn extension_mismatched_units_errors() {
    let mut ctx = array_session().create_execution_ctx();
    let ms = ExtensionArray::new(
        Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased(),
        buffer![1000i64, 2000].into_array(),
    )
    .into_array();
    let secs = ExtensionArray::new(
        Timestamp::new(TimeUnit::Seconds, Nullability::NonNullable).erased(),
        buffer![1i64, 3].into_array(),
    )
    .into_array();

    let result = ms
        .binary(secs, Operator::Eq)
        .and_then(|a| a.execute::<BoolArray>(&mut ctx));
    assert!(result.is_err());
}

/// Struct fields containing nulls order null-first, matching `Scalar` comparison semantics;
/// only top-level nulls make the result null.
#[test]
fn struct_compare_null_fields_order_first() {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = StructArray::from_fields(&[(
        "a",
        PrimitiveArray::from_option_iter([None, Some(5i32), None]).into_array(),
    )])
    .unwrap()
    .into_array();
    let rhs = StructArray::from_fields(&[(
        "a",
        PrimitiveArray::from_option_iter([Some(1i32), Some(5), None]).into_array(),
    )])
    .unwrap()
    .into_array();

    // null field < non-null field; null == null.
    let result = execute_compare_test(lhs.clone(), rhs.clone(), Operator::Lt);
    assert_arrays_eq!(result, BoolArray::from_iter([true, false, false]), &mut ctx);

    let result = execute_compare_test(lhs, rhs, Operator::Eq);
    assert_arrays_eq!(result, BoolArray::from_iter([false, true, true]), &mut ctx);
}

#[test]
fn fixed_size_list_compare() {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = FixedSizeListArray::new(
        buffer![1i32, 2, 3, 4, 5, 6].into_array(),
        2,
        Validity::NonNullable,
        3,
    )
    .into_array();
    let rhs = FixedSizeListArray::new(
        buffer![1i32, 2, 3, 5, 4, 6].into_array(),
        2,
        Validity::NonNullable,
        3,
    )
    .into_array();

    let result = execute_compare_test(lhs.clone(), rhs.clone(), Operator::Eq);
    assert_arrays_eq!(result, BoolArray::from_iter([true, false, false]), &mut ctx);

    let result = execute_compare_test(lhs, rhs, Operator::Lt);
    assert_arrays_eq!(result, BoolArray::from_iter([false, true, false]), &mut ctx);
}

/// Binary (non-utf8) comparison over VarBinView arrays.
#[test]
fn binary_compare() {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = VarBinViewArray::from_iter_bin([b"bad".as_slice(), b"\xff\x00", b""]).into_array();
    let rhs = VarBinViewArray::from_iter_bin([b"bat".as_slice(), b"\xff", b""]).into_array();

    let result = execute_compare_test(lhs.clone(), rhs.clone(), Operator::Lt);
    assert_arrays_eq!(result, BoolArray::from_iter([true, false, false]), &mut ctx);

    let result = execute_compare_test(lhs, rhs, Operator::Eq);
    assert_arrays_eq!(result, BoolArray::from_iter([false, false, true]), &mut ctx);
}

/// A `map(i32, utf8?)` dtype that makes no sortedness assertion.
fn map_dtype(nullability: Nullability) -> VortexResult<DType> {
    DType::map(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable),
        false,
        nullability,
    )
}

type MapRow = Vec<(i32, Option<&'static str>)>;

fn map_scalar(nullability: Nullability, entries: MapRow) -> VortexResult<Scalar> {
    Scalar::try_map(
        map_dtype(nullability)?,
        entries.into_iter().map(|(key, value)| {
            (
                Scalar::primitive(key, Nullability::NonNullable),
                match value {
                    Some(value) => Scalar::utf8(value, Nullability::Nullable),
                    None => Scalar::null(DType::Utf8(Nullability::Nullable)),
                },
            )
        }),
    )
}

fn map_array(
    nullability: Nullability,
    rows: impl IntoIterator<Item = Option<MapRow>>,
) -> VortexResult<ArrayRef> {
    let rows = rows.into_iter().collect::<Vec<_>>();
    let dtype = map_dtype(nullability)?;
    let map_dtype = dtype.as_map_opt().vortex_expect("map dtype").clone();
    let mut builder = MapBuilder::<u64, u64>::with_capacity(map_dtype, nullability, rows.len());
    for row in rows {
        let scalar = match row {
            Some(entries) => map_scalar(nullability, entries)?,
            None => Scalar::null(dtype.clone()),
        };
        builder.append_scalar(&scalar)?;
    }
    Ok(builder.finish_into_map().into_array())
}

/// Maps compare as the ordered sequence of their `{key, value}` entries: entry-wise first, then
/// by entry count. A null map value orders before every non-null one.
#[rstest]
#[case(Operator::Eq, [true, false, false, false, true])]
#[case(Operator::NotEq, [false, true, true, true, false])]
#[case(Operator::Lt, [false, true, false, false, false])]
#[case(Operator::Lte, [true, true, false, false, true])]
#[case(Operator::Gt, [false, false, true, true, false])]
#[case(Operator::Gte, [true, false, true, true, true])]
fn map_compare_all_operators(
    #[case] op: Operator,
    #[case] expected: [bool; 5],
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = map_array(
        Nullability::NonNullable,
        [
            // Identical entries.
            Some(vec![(1, Some("a")), (2, Some("b"))]),
            // A strict prefix of the right-hand row, so it orders first.
            Some(vec![(1, Some("a"))]),
            // Keys break the tie before values do.
            Some(vec![(2, Some("a"))]),
            // A null map value orders before a non-null one.
            Some(vec![(1, Some("a"))]),
            // Two empty maps.
            Some(vec![]),
        ],
    )?;
    let rhs = map_array(
        Nullability::NonNullable,
        [
            Some(vec![(1, Some("a")), (2, Some("b"))]),
            Some(vec![(1, Some("a")), (2, Some("b"))]),
            Some(vec![(1, Some("z"))]),
            Some(vec![(1, None)]),
            Some(vec![]),
        ],
    )?;

    let result = execute_compare_test(lhs, rhs, op);
    assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);

    Ok(())
}

/// Only top-level nulls make a map comparison null; an empty map is not a null map.
#[test]
fn map_compare_nulls() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = map_array(
        Nullability::Nullable,
        [None, None, Some(vec![]), Some(vec![(1, Some("a"))])],
    )?;
    let rhs = map_array(
        Nullability::Nullable,
        [None, Some(vec![]), Some(vec![]), Some(vec![(1, Some("a"))])],
    )?;

    let result = execute_compare_test(lhs, rhs, Operator::Eq)
        .execute::<BoolArray>(&mut ctx)
        .vortex_expect("bool array");
    let expected = BoolArray::from_iter([None, None, Some(true), Some(true)]);
    assert_arrays_eq!(result, expected, &mut ctx);

    Ok(())
}

/// A map array compared against a constant map, and two constant maps compared through
/// [`scalar_cmp`], agree with the row-wise kernel.
#[test]
fn map_constant_compare() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = map_array(
        Nullability::NonNullable,
        [
            Some(vec![(1, Some("a"))]),
            Some(vec![(1, Some("a")), (2, Some("b"))]),
            Some(vec![(9, Some("a"))]),
        ],
    )?;
    let needle = map_scalar(Nullability::NonNullable, vec![(1, Some("a"))])?;
    let constant = ConstantArray::new(needle.clone(), array.len()).into_array();

    let result = execute_compare_test(array.clone(), constant.clone(), Operator::Eq);
    assert_arrays_eq!(result, BoolArray::from_iter([true, false, false]), &mut ctx);

    let result = execute_compare_test(array, constant, Operator::Lt);
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([false, false, false]),
        &mut ctx
    );

    // Constant-vs-constant folds through `scalar_cmp`.
    let bigger = map_scalar(
        Nullability::NonNullable,
        vec![(1, Some("a")), (2, Some("b"))],
    )?;
    assert_eq!(
        scalar_cmp(&needle, &bigger, CompareOperator::Lt)?,
        Scalar::bool(true, Nullability::NonNullable)
    );
    assert_eq!(
        scalar_cmp(&needle, &needle, CompareOperator::Eq)?,
        Scalar::bool(true, Nullability::NonNullable)
    );

    Ok(())
}

/// Maps nested inside another nested type compare through the same row comparator tree.
#[test]
fn struct_of_map_compare() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = StructArray::from_fields(&[(
        "m",
        map_array(
            Nullability::NonNullable,
            [Some(vec![(1, Some("a"))]), Some(vec![(1, Some("a"))])],
        )?,
    )])?
    .into_array();
    let rhs = StructArray::from_fields(&[(
        "m",
        map_array(
            Nullability::NonNullable,
            [Some(vec![(1, Some("a"))]), Some(vec![(1, Some("b"))])],
        )?,
    )])?
    .into_array();

    let result = execute_compare_test(lhs.clone(), rhs.clone(), Operator::Eq);
    assert_arrays_eq!(result, BoolArray::from_iter([true, false]), &mut ctx);

    let result = execute_compare_test(lhs, rhs, Operator::Lt);
    assert_arrays_eq!(result, BoolArray::from_iter([false, true]), &mut ctx);

    Ok(())
}
