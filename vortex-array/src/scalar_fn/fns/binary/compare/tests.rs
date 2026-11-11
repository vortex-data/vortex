// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::Ordering;

use prost::Message;
use rstest::rstest;
use vortex_buffer::BitBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::DictArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::ScalarFnArray;
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
use crate::expr::compare_with_mode;
use crate::expr::lit;
use crate::expr::root;
use crate::extension::datetime::Time;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::extension::datetime::TimestampOptions;
use crate::memory::MemorySessionExt;
use crate::memory::test_allocator::counting_allocator;
use crate::proto::expr::BinaryOpts;
use crate::proto::expr::binary_opts::SignedZeros as ProtoSignedZeros;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::BinaryOptions;
use crate::scalar_fn::fns::binary::scalar_cmp;
use crate::scalar_fn::fns::operators::CompareMode;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::NanOrder;
use crate::scalar_fn::fns::operators::NestedNulls;
use crate::scalar_fn::fns::operators::Operator;
use crate::scalar_fn::fns::operators::SignedZeros;
use crate::test_harness::ExpectedOutcome;
use crate::test_harness::assert_execution_paths;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn test_bool_basic_comparisons() {
    let ctx = &mut SESSION.create_execution_ctx();
    let arr = BoolArray::new(
        BitBuffer::from_iter([true, true, false, true, false]),
        Validity::from_iter([false, true, true, true, true]),
    );

    let matches = arr
        .clone()
        .into_array()
        .binary(arr.clone().into_array(), Operator::Eq)
        .unwrap();
    assert_arrays_eq!(
        matches,
        BoolArray::from_iter([None, Some(true), Some(true), Some(true), Some(true)]),
        ctx
    );

    let matches = arr
        .clone()
        .into_array()
        .binary(arr.clone().into_array(), Operator::NotEq)
        .unwrap();
    assert_arrays_eq!(
        matches,
        BoolArray::from_iter([None, Some(false), Some(false), Some(false), Some(false)]),
        ctx
    );

    let other = BoolArray::new(
        BitBuffer::from_iter([false, false, false, true, true]),
        Validity::from_iter([false, true, true, true, true]),
    );

    let matches = arr
        .clone()
        .into_array()
        .binary(other.clone().into_array(), Operator::Lte)
        .unwrap();
    assert_arrays_eq!(
        matches,
        BoolArray::from_iter([None, Some(false), Some(true), Some(true), Some(true)]),
        ctx
    );

    let matches = arr
        .clone()
        .into_array()
        .binary(other.clone().into_array(), Operator::Lt)
        .unwrap();
    assert_arrays_eq!(
        matches,
        BoolArray::from_iter([None, Some(false), Some(false), Some(false), Some(true)]),
        ctx
    );

    let matches = other
        .clone()
        .into_array()
        .binary(arr.clone().into_array(), Operator::Gte)
        .unwrap();
    assert_arrays_eq!(
        matches,
        BoolArray::from_iter([None, Some(false), Some(true), Some(true), Some(true)]),
        ctx
    );

    let matches = other
        .into_array()
        .binary(arr.into_array(), Operator::Gt)
        .unwrap();
    assert_arrays_eq!(
        matches,
        BoolArray::from_iter([None, Some(false), Some(false), Some(false), Some(true)]),
        ctx
    );
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
        .execute_scalar(0, &mut SESSION.create_execution_ctx())
        .unwrap();
    assert_eq!(scalar.as_bool().value(), Some(false));
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([false; 10]),
        &mut SESSION.create_execution_ctx()
    );
}

#[rstest]
#[case(VarBinArray::from(vec!["a", "b"]).into_array(), VarBinViewArray::from_iter_str(["a", "b"]).into_array())]
#[case(VarBinViewArray::from_iter_str(["a", "b"]).into_array(), VarBinArray::from(vec!["a", "b"]).into_array())]
#[case(VarBinArray::from(vec!["a".as_bytes(), "b".as_bytes()]).into_array(), VarBinViewArray::from_iter_bin(["a".as_bytes(), "b".as_bytes()]).into_array())]
#[case(VarBinViewArray::from_iter_bin(["a".as_bytes(), "b".as_bytes()]).into_array(), VarBinArray::from(vec!["a".as_bytes(), "b".as_bytes()]).into_array())]
fn compare_different_encodings(#[case] left: ArrayRef, #[case] right: ArrayRef) {
    let mut ctx = SESSION.create_execution_ctx();
    let res = left.binary(right, Operator::Eq).unwrap();
    let expected = BoolArray::from_iter([true, true]);
    assert_arrays_eq!(res, expected, &mut ctx);
}

#[test]
fn test_list_array_comparison() {
    let mut ctx = SESSION.create_execution_ctx();
    let values1 = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]);
    let offsets1 = PrimitiveArray::from_iter([0i32, 2, 4, 6]);
    let list1 = ListArray::try_new(
        values1.into_array(),
        offsets1.into_array(),
        Validity::NonNullable,
        &mut ctx,
    )
    .unwrap();

    let values2 = PrimitiveArray::from_iter([1i32, 2, 3, 4, 7, 8]);
    let offsets2 = PrimitiveArray::from_iter([0i32, 2, 4, 6]);
    let list2 = ListArray::try_new(
        values2.into_array(),
        offsets2.into_array(),
        Validity::NonNullable,
        &mut ctx,
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
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]);
    let offsets = PrimitiveArray::from_iter([0i32, 2, 4, 6]);
    let list = ListArray::try_new(
        values.into_array(),
        offsets.into_array(),
        Validity::NonNullable,
        &mut ctx,
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
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut ctx = SESSION.create_execution_ctx();
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
    let ctx = &mut SESSION.create_execution_ctx();
    let list = ListViewArray::new(
        BoolArray::from_iter(Vec::<bool>::new()).into_array(),
        buffer![0i32, 0i32, 0i32].into_array(),
        buffer![0i32, 0i32, 0i32].into_array(),
        Validity::AllValid,
        ctx,
    );

    let result = list
        .clone()
        .into_array()
        .binary(list.into_array(), Operator::Eq)
        .unwrap();
    assert!(result.execute_scalar(0, ctx).unwrap().is_valid());
    assert!(result.execute_scalar(1, ctx).unwrap().is_valid());
    assert!(result.execute_scalar(2, ctx).unwrap().is_valid());
    assert_arrays_eq!(result, BoolArray::from_iter([Some(true); 3]), ctx);
}

fn execute_compare_test(lhs: ArrayRef, rhs: ArrayRef, op: Operator) -> ArrayRef {
    lhs.binary(rhs, op).unwrap()
}

#[test]
fn comparison_uses_execution_allocator() -> VortexResult<()> {
    let (allocator, allocations) = counting_allocator();
    let mut ctx = VortexSession::empty()
        .with_allocator(allocator)
        .create_execution_ctx();
    let result = buffer![1i32, 2, 3]
        .into_array()
        .binary(buffer![1i32, 0, 3].into_array(), Operator::Eq)?
        .execute::<BoolArray>(&mut ctx)?;
    drop(result);
    assert_ne!(allocations.load(Ordering::Relaxed), 0);
    Ok(())
}

#[rstest]
#[case(Operator::Eq, [false, true, false, false])]
#[case(Operator::NotEq, [true, false, true, true])]
#[case(Operator::Lt, [true, false, false, true])]
#[case(Operator::Lte, [true, true, false, true])]
#[case(Operator::Gt, [false, false, true, false])]
#[case(Operator::Gte, [false, true, true, false])]
fn int_all_operators(#[case] op: Operator, #[case] expected: [bool; 4]) {
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = buffer![1i32, 5, 9, 2].into_array();
    let rhs = buffer![3i32, 5, 7, 4].into_array();
    let result = execute_compare_test(lhs, rhs, op);
    assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);
}

#[rstest]
#[case(Operator::Lt, [Some(true), None, Some(false), None])]
#[case(Operator::Eq, [Some(false), None, Some(false), None])]
fn int_nullable(#[case] op: Operator, #[case] expected: [Option<bool>; 4]) {
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = PrimitiveArray::from_option_iter([Some(1i64), None, Some(9), Some(2)]).into_array();
    let rhs = PrimitiveArray::from_option_iter([Some(3i64), Some(5), Some(7), None]).into_array();
    let result = execute_compare_test(lhs, rhs, op);
    assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);
}

#[rstest]
#[case(Operator::Gt, [false, false, true, false])]
#[case(Operator::Lte, [true, true, false, true])]
fn int_constant_lhs_and_rhs(#[case] op: Operator, #[case] expected: [bool; 4]) {
    let mut ctx = SESSION.create_execution_ctx();
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
#[rstest]
fn float_total_order(#[values(4, 129)] len: usize) {
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = PrimitiveArray::from_iter(
        [f64::NAN, f64::NAN, -0.0, 1.0]
            .into_iter()
            .cycle()
            .take(len),
    )
    .into_array();
    let rhs = PrimitiveArray::from_iter(
        [f64::NAN, f64::INFINITY, 0.0, f64::NAN]
            .into_iter()
            .cycle()
            .take(len),
    )
    .into_array();

    let result = execute_compare_test(lhs.clone(), rhs.clone(), Operator::Eq);
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([true, false, false, false].into_iter().cycle().take(len)),
        &mut ctx
    );

    let result = execute_compare_test(lhs, rhs, Operator::Lt);
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([false, false, true, true].into_iter().cycle().take(len)),
        &mut ctx
    );
}

#[rstest]
#[case(Operator::Eq, [true, false, true, true])]
#[case(Operator::Lt, [false, true, false, false])]
#[case(Operator::Gte, [true, false, true, true])]
fn bool_constant(#[case] op: Operator, #[case] expected: [bool; 4]) {
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = VarBinViewArray::from_iter_str([lhs]).into_array();
    let rhs = VarBinViewArray::from_iter_str([rhs]).into_array();
    let result = execute_compare_test(lhs, rhs, op);
    assert_arrays_eq!(result, BoolArray::from_iter([expected]), &mut ctx);
}

#[test]
fn string_constant_compare() {
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut ctx = SESSION.create_execution_ctx();
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

#[rstest]
#[case::storage(DType::Primitive(PType::I64, Nullability::Nullable))]
#[case::different_unit(DType::Extension(
    Timestamp::new(TimeUnit::Nanoseconds, Nullability::Nullable).erased()
))]
#[case::different_extension(DType::Extension(
    Time::new(TimeUnit::Microseconds, Nullability::Nullable).erased()
))]
#[case::different_time_zone(DType::Extension(
    Timestamp::new_with_tz(TimeUnit::Milliseconds, Some("UTC".into()), Nullability::Nullable).erased()
))]
fn test_extension_comparison_rejects_mismatched_dtypes_at_binding(
    #[case] rhs_dtype: DType,
    #[values(false, true)] swapped: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = ExtensionArray::new(
        Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased(),
        buffer![1600i64, 1400].into_array(),
    )
    .into_array();
    let scalar = Scalar::try_new(rhs_dtype, Some(1500i64.into()))?;
    let rhs = ConstantArray::new(scalar, lhs.len()).into_array();
    let inputs = if swapped { [rhs, lhs] } else { [lhs, rhs] };

    assert_execution_paths(
        &inputs,
        |inputs| {
            let error = inputs[0]
                .binary(inputs[1].clone(), Operator::Lt)
                .expect_err("mismatched comparison dtypes must be rejected while binding");

            Err(error)
        },
        ExpectedOutcome::Error,
        &mut ctx,
    );

    Ok(())
}

#[rstest]
#[case::equal(Operator::Eq, [Some(false), Some(true), Some(false), None])]
#[case::not_equal(Operator::NotEq, [Some(true), Some(false), Some(true), None])]
#[case::less(Operator::Lt, [Some(false), Some(false), Some(true), None])]
#[case::less_equal(Operator::Lte, [Some(false), Some(true), Some(true), None])]
#[case::greater(Operator::Gt, [Some(true), Some(false), Some(false), None])]
#[case::greater_equal(Operator::Gte, [Some(true), Some(true), Some(false), None])]
fn test_extension_comparison_preserves_paths_and_nulls(
    #[case] operator: Operator,
    #[case] expected: [Option<bool>; 4],
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
    let lhs = ExtensionArray::new(
        dtype,
        PrimitiveArray::from_option_iter([Some(1600i64), Some(1500), Some(1400), None])
            .into_array(),
    )
    .into_array();
    let rhs = ConstantArray::new(
        Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit: TimeUnit::Milliseconds,
                tz: None,
            },
            Scalar::from(1500i64),
        ),
        lhs.len(),
    )
    .into_array();

    assert_execution_paths(
        &[lhs, rhs],
        |inputs| inputs[0].binary(inputs[1].clone(), operator),
        ExpectedOutcome::Array(BoolArray::from_iter(expected).into_array()),
        &mut ctx,
    );

    Ok(())
}

#[rstest]
#[case::empty(0)]
#[case::null(2)]
fn test_extension_comparison_checks_dtypes_before_fast_paths(
    #[case] len: usize,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
    let lhs = ExtensionArray::new(
        dtype,
        ConstantArray::new(Scalar::primitive(1000i64, Nullability::Nullable), len).into_array(),
    )
    .into_array();
    let rhs = ConstantArray::new(
        Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
        len,
    )
    .into_array();

    assert!(
        super::execute_compare(
            &lhs,
            &rhs,
            CompareOperator::Eq,
            CompareMode::TOTAL_ORDER,
            &mut ctx
        )
        .is_err()
    );

    Ok(())
}

#[test]
fn test_null_scalar_comparison_requires_matching_dtypes() {
    let dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();

    assert!(
        scalar_cmp(
            &Scalar::null(DType::Extension(dtype)),
            &Scalar::from(1000i64),
            CompareOperator::Eq
        )
        .is_err()
    );
}

/// Struct fields containing nulls order null-first, matching `Scalar` comparison semantics;
/// only top-level nulls make the result null.
#[test]
fn struct_compare_null_fields_order_first() {
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut builder = MapBuilder::<u64, u64>::with_capacity_in(
        map_dtype,
        nullability,
        rows.len(),
        vortex_buffer::BufferAllocatorRef::static_ref(),
    );
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
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = map_array(
        Nullability::Nullable,
        [None, None, Some(vec![]), Some(vec![(1, Some("a"))])],
    )?;
    let rhs = map_array(
        Nullability::Nullable,
        [None, Some(vec![]), Some(vec![]), Some(vec![(1, Some("a"))])],
    )?;

    let result = execute_compare_test(lhs, rhs, Operator::Eq);
    let expected = BoolArray::from_iter([None, None, Some(true), Some(true)]);
    assert_arrays_eq!(result, expected, &mut ctx);

    Ok(())
}

/// A map array compared against a constant map, and two constant maps compared through
/// [`scalar_cmp`], agree with the row-wise kernel.
#[test]
fn map_constant_compare() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut ctx = SESSION.create_execution_ctx();
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

#[rstest]
fn primitive_comparisons_across_bitmap_words(
    #[values(PType::I8, PType::U8, PType::I64, PType::U64, PType::F64)] ptype: PType,
    #[values(0, 1, 63, 64, 65, 129)] len: usize,
    #[values(
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Lt,
        CompareOperator::Lte,
        CompareOperator::Gt,
        CompareOperator::Gte
    )]
    op: CompareOperator,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let value_for_type = |value: Option<i16>| {
        value.map(|value| {
            if ptype.is_unsigned_int() {
                value.abs()
            } else {
                value
            }
        })
    };
    let left: Vec<_> = [Some(-128i16), Some(-1), None, Some(127), Some(42), Some(2)]
        .into_iter()
        .cycle()
        .take(len)
        .map(value_for_type)
        .collect();
    let right: Vec<_> = [
        Some(-1i16),
        Some(-128),
        Some(127),
        None,
        Some(42),
        Some(3),
        Some(100),
    ]
    .into_iter()
    .cycle()
    .take(len)
    .map(value_for_type)
    .collect();
    let dtype = DType::Primitive(ptype, Nullability::Nullable);
    let lhs = PrimitiveArray::from_option_iter([None].into_iter().chain(left.iter().copied()))
        .into_array()
        .cast(dtype.clone())?
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array()
        .slice(1..len + 1)?;
    let rhs =
        PrimitiveArray::from_option_iter([None, None].into_iter().chain(right.iter().copied()))
            .into_array()
            .cast(dtype.clone())?
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array()
            .slice(2..len + 2)?;
    let predicate = |a: i16, b: i16| match op {
        CompareOperator::Eq => a == b,
        CompareOperator::NotEq => a != b,
        CompareOperator::Lt => a < b,
        CompareOperator::Lte => a <= b,
        CompareOperator::Gt => a > b,
        CompareOperator::Gte => a >= b,
    };
    let expected = BoolArray::from_iter(
        left.iter()
            .zip(&right)
            .map(|(a, b)| a.zip(*b).map(|(a, b)| predicate(a, b))),
    );
    assert_arrays_eq!(lhs.binary(rhs, op.into())?, expected, &mut ctx);

    let constant = ConstantArray::new(
        Scalar::primitive(42u8, Nullability::Nullable).cast(&dtype)?,
        len,
    )
    .into_array();
    for swapped in [false, true] {
        let actual = if swapped {
            constant.binary(lhs.clone(), op.into())?
        } else {
            lhs.binary(constant.clone(), op.into())?
        };
        let expected = BoolArray::from_iter(left.iter().map(|value| {
            value.map(|value| {
                if swapped {
                    predicate(42, value)
                } else {
                    predicate(value, 42)
                }
            })
        }));
        assert_arrays_eq!(actual, expected, &mut ctx);
    }
    Ok(())
}

/// Floats that each compare mode orders differently: both NaN signs, both zeros, and both
/// infinities.
const MODE_VALUES: [f64; 8] = [
    -f64::NAN,
    f64::NEG_INFINITY,
    -1.0,
    -0.0,
    0.0,
    1.0,
    f64::INFINITY,
    f64::NAN,
];

const DATAFUSION: CompareMode = CompareMode::new(SignedZeros::Equal, NanOrder::TotalOrder);
const SPARK: CompareMode = CompareMode::new(SignedZeros::Equal, NanOrder::Canonical);
const ICEBERG: CompareMode = CompareMode::new(SignedZeros::Distinct, NanOrder::Canonical);
const CLICKHOUSE: CompareMode = CompareMode::new(SignedZeros::Equal, NanOrder::Ieee754);
const DUCKDB: CompareMode = SPARK.with_nested_nulls(NestedNulls::Last);

fn compare_with(
    lhs: ArrayRef,
    rhs: ArrayRef,
    op: CompareOperator,
    mode: CompareMode,
) -> VortexResult<ArrayRef> {
    Ok(
        ScalarFnArray::try_new(Binary.bind(BinaryOptions::compare(op, mode)), [lhs, rhs])?
            .into_array(),
    )
}

#[rstest]
#[case::total_eq_zero(CompareMode::TOTAL_ORDER, CompareOperator::Eq, 0.0, [false, false, false, false, true, false, false, false])]
#[case::total_lt_zero(CompareMode::TOTAL_ORDER, CompareOperator::Lt, 0.0, [true, true, true, true, false, false, false, false])]
#[case::datafusion_eq_zero(DATAFUSION, CompareOperator::Eq, 0.0, [false, false, false, true, true, false, false, false])]
#[case::datafusion_eq_neg_zero(DATAFUSION, CompareOperator::Eq, -0.0, [false, false, false, true, true, false, false, false])]
#[case::datafusion_lt_zero(DATAFUSION, CompareOperator::Lt, 0.0, [true, true, true, false, false, false, false, false])]
#[case::datafusion_gte_neg_zero(DATAFUSION, CompareOperator::Gte, -0.0, [false, false, false, true, true, true, true, true])]
#[case::datafusion_eq_nan(DATAFUSION, CompareOperator::Eq, f64::NAN, [false, false, false, false, false, false, false, true])]
#[case::spark_eq_nan(SPARK, CompareOperator::Eq, f64::NAN, [true, false, false, false, false, false, false, true])]
#[case::spark_eq_neg_nan(SPARK, CompareOperator::Eq, -f64::NAN, [true, false, false, false, false, false, false, true])]
#[case::spark_lt_one(SPARK, CompareOperator::Lt, 1.0, [false, true, true, true, true, false, false, false])]
#[case::spark_gt_inf(SPARK, CompareOperator::Gt, f64::INFINITY, [true, false, false, false, false, false, false, true])]
#[case::spark_lte_zero(SPARK, CompareOperator::Lte, 0.0, [false, true, true, true, true, false, false, false])]
#[case::iceberg_eq_zero(ICEBERG, CompareOperator::Eq, 0.0, [false, false, false, false, true, false, false, false])]
#[case::iceberg_gt_one(ICEBERG, CompareOperator::Gt, 1.0, [true, false, false, false, false, false, true, true])]
#[case::clickhouse_eq_nan(CLICKHOUSE, CompareOperator::Eq, f64::NAN, [false; 8])]
#[case::clickhouse_not_eq_nan(CLICKHOUSE, CompareOperator::NotEq, f64::NAN, [true; 8])]
#[case::clickhouse_gte_neg_zero(CLICKHOUSE, CompareOperator::Gte, -0.0, [false, false, false, true, true, true, true, false])]
#[case::clickhouse_not_eq_one(CLICKHOUSE, CompareOperator::NotEq, 1.0, [true, true, true, true, true, false, true, true])]
fn float_compare_modes(
    #[case] mode: CompareMode,
    #[case] op: CompareOperator,
    #[case] constant: f64,
    #[case] expected: [bool; 8],
) -> VortexResult<()> {
    let ctx = &mut SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter(MODE_VALUES).into_array();
    let expected = BoolArray::from_iter(expected);
    let len = MODE_VALUES.len();

    let with_constant = compare_with(
        values.clone(),
        ConstantArray::new(constant, len).into_array(),
        op,
        mode,
    )?;
    assert_arrays_eq!(with_constant, expected, ctx);

    let with_array = compare_with(
        values.clone(),
        PrimitiveArray::from_iter([constant; 8]).into_array(),
        op,
        mode,
    )?;
    assert_arrays_eq!(with_array, expected, ctx);

    let constant_on_left = compare_with(
        ConstantArray::new(constant, len).into_array(),
        values,
        op.swap(),
        mode,
    )?;
    assert_arrays_eq!(constant_on_left, expected, ctx);

    Ok(())
}

#[rstest]
#[case::f16(PrimitiveArray::from_iter([half::f16::NEG_ZERO, half::f16::ZERO, half::f16::NAN]).into_array(), Scalar::from(half::f16::ZERO))]
#[case::f32(PrimitiveArray::from_iter([-0.0f32, 0.0, f32::NAN]).into_array(), Scalar::from(0.0f32))]
fn signed_zeros_equal_for_each_float_width(
    #[case] values: ArrayRef,
    #[case] zero: Scalar,
) -> VortexResult<()> {
    let ctx = &mut SESSION.create_execution_ctx();
    let zero = ConstantArray::new(zero, values.len()).into_array();

    let result = compare_with(values, zero, CompareOperator::Eq, DATAFUSION)?;

    assert_arrays_eq!(result, BoolArray::from_iter([true, true, false]), ctx);
    Ok(())
}

#[test]
fn compare_mode_ignores_integers() -> VortexResult<()> {
    let ctx = &mut SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter([-1i32, 0, 1]).into_array();
    let zero = ConstantArray::new(0i32, 3).into_array();

    let result = compare_with(values, zero, CompareOperator::Lte, CLICKHOUSE)?;

    assert_arrays_eq!(result, BoolArray::from_iter([true, true, false]), ctx);
    Ok(())
}

/// A total-order encoding kernel must not answer a comparison that its mode orders differently.
#[test]
fn compare_mode_declines_encoding_kernel_for_zero() -> VortexResult<()> {
    let ctx = &mut SESSION.create_execution_ctx();
    let values = DictArray::try_new(
        buffer![0u8, 1, 0, 1].into_array(),
        PrimitiveArray::from_iter([-0.0f64, 0.0]).into_array(),
    )?
    .into_array();
    let zero = ConstantArray::new(0.0f64, 4).into_array();

    let result = compare_with(values, zero, CompareOperator::Eq, DATAFUSION)?;

    assert_arrays_eq!(result, BoolArray::from_iter([true; 4]), ctx);
    Ok(())
}

/// Structs compare field by field, and each float field compares in the mode. The rows are
/// `(-0.0, 1) ? (0.0, 1)`, `(NaN, 1) ? (NaN, 2)`, `(-NaN, 1) ? (NaN, 1)`, `(NaN, 1) ? (inf, 1)`
/// and `(1.0, 1) ? (1.0, 2)`. An unordered float field makes the rows unordered.
#[rstest]
#[case::total_eq(CompareMode::TOTAL_ORDER, CompareOperator::Eq, [false, false, false, false, false])]
#[case::total_lt(CompareMode::TOTAL_ORDER, CompareOperator::Lt, [true, true, true, false, true])]
#[case::datafusion_eq(DATAFUSION, CompareOperator::Eq, [true, false, false, false, false])]
#[case::spark_eq(SPARK, CompareOperator::Eq, [true, false, true, false, false])]
#[case::spark_lt(SPARK, CompareOperator::Lt, [false, true, false, false, true])]
#[case::spark_gt(SPARK, CompareOperator::Gt, [false, false, false, true, false])]
#[case::iceberg_eq(ICEBERG, CompareOperator::Eq, [false, false, true, false, false])]
#[case::clickhouse_eq(CLICKHOUSE, CompareOperator::Eq, [true, false, false, false, false])]
#[case::clickhouse_not_eq(CLICKHOUSE, CompareOperator::NotEq, [false, true, true, true, true])]
#[case::clickhouse_lt(CLICKHOUSE, CompareOperator::Lt, [false, false, false, false, true])]
#[case::clickhouse_gte(CLICKHOUSE, CompareOperator::Gte, [true, false, false, false, false])]
fn compare_mode_struct_floats(
    #[case] mode: CompareMode,
    #[case] op: CompareOperator,
    #[case] expected: [bool; 5],
) -> VortexResult<()> {
    let ctx = &mut SESSION.create_execution_ctx();
    let lhs = StructArray::from_fields(&[
        (
            "a",
            PrimitiveArray::from_iter([-0.0f64, f64::NAN, -f64::NAN, f64::NAN, 1.0]).into_array(),
        ),
        (
            "b",
            PrimitiveArray::from_iter([1i32, 1, 1, 1, 1]).into_array(),
        ),
    ])?
    .into_array();
    let rhs = StructArray::from_fields(&[
        (
            "a",
            PrimitiveArray::from_iter([0.0f64, f64::NAN, f64::NAN, f64::INFINITY, 1.0])
                .into_array(),
        ),
        (
            "b",
            PrimitiveArray::from_iter([1i32, 2, 1, 1, 2]).into_array(),
        ),
    ])?
    .into_array();

    let result = compare_with(lhs, rhs, op, mode)?;

    assert_arrays_eq!(result, BoolArray::from_iter(expected), ctx);
    Ok(())
}

/// Lists compare element by element in the mode. The rows are `[-0.0, NaN] ? [0.0, NaN]` and
/// `[NaN] ? [NaN, 1.0]`.
#[rstest]
#[case::total_eq(CompareMode::TOTAL_ORDER, CompareOperator::Eq, [false, false])]
#[case::spark_eq(SPARK, CompareOperator::Eq, [true, false])]
#[case::spark_lt(SPARK, CompareOperator::Lt, [false, true])]
#[case::clickhouse_eq(CLICKHOUSE, CompareOperator::Eq, [false, false])]
#[case::clickhouse_lt(CLICKHOUSE, CompareOperator::Lt, [false, false])]
#[case::clickhouse_not_eq(CLICKHOUSE, CompareOperator::NotEq, [true, true])]
fn compare_mode_list_floats(
    #[case] mode: CompareMode,
    #[case] op: CompareOperator,
    #[case] expected: [bool; 2],
) -> VortexResult<()> {
    let ctx = &mut SESSION.create_execution_ctx();
    let lhs = ListArray::try_new(
        PrimitiveArray::from_iter([-0.0f64, f64::NAN, f64::NAN]).into_array(),
        PrimitiveArray::from_iter([0i32, 2, 3]).into_array(),
        Validity::NonNullable,
        ctx,
    )?
    .into_array();
    let rhs = ListArray::try_new(
        PrimitiveArray::from_iter([0.0f64, f64::NAN, f64::NAN, 1.0]).into_array(),
        PrimitiveArray::from_iter([0i32, 2, 4]).into_array(),
        Validity::NonNullable,
        ctx,
    )?
    .into_array();

    let result = compare_with(lhs, rhs, op, mode)?;

    assert_arrays_eq!(result, BoolArray::from_iter(expected), ctx);
    Ok(())
}

#[rstest]
#[case::total_order(CompareMode::TOTAL_ORDER, CompareOperator::Lt, 0.0, true)]
#[case::zeros_equal_nonzero(DATAFUSION, CompareOperator::Lt, 1.0, true)]
#[case::zeros_equal_zero(DATAFUSION, CompareOperator::Eq, 0.0, false)]
#[case::zeros_equal_nan(DATAFUSION, CompareOperator::Gt, f64::NAN, true)]
#[case::canonical_eq(SPARK, CompareOperator::Eq, 1.0, true)]
#[case::canonical_order(SPARK, CompareOperator::Lt, 1.0, false)]
#[case::canonical_nan(SPARK, CompareOperator::Eq, f64::NAN, false)]
#[case::ieee_not_eq(CLICKHOUSE, CompareOperator::NotEq, 1.0, true)]
#[case::ieee_order(CLICKHOUSE, CompareOperator::Gte, 1.0, false)]
fn compare_mode_agrees_with_total_order(
    #[case] mode: CompareMode,
    #[case] op: CompareOperator,
    #[case] constant: f64,
    #[case] expected: bool,
) {
    let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);

    assert_eq!(
        mode.agrees_with_total_order(op, &dtype, Some(&Scalar::from(constant))),
        expected
    );
    assert!(mode.agrees_with_total_order(
        op,
        &DType::Primitive(PType::I64, Nullability::NonNullable),
        None
    ));
}

#[rstest]
#[case::total_order(BinaryOptions::from(Operator::Eq))]
#[case::datafusion(BinaryOptions::compare(CompareOperator::Lt, DATAFUSION))]
#[case::clickhouse(BinaryOptions::compare(CompareOperator::NotEq, CLICKHOUSE))]
#[case::duckdb(BinaryOptions::compare(CompareOperator::Gt, DUCKDB))]
fn binary_options_round_trip(#[case] options: BinaryOptions) -> VortexResult<()> {
    let bytes = Binary
        .serialize(&options)?
        .vortex_expect("binary options serialize");

    assert_eq!(Binary.deserialize(&bytes, &SESSION)?, options);
    Ok(())
}

#[test]
fn compare_mode_requires_comparison_operator() -> VortexResult<()> {
    let bytes = BinaryOpts {
        op: Operator::Add.into(),
        signed_zeros: ProtoSignedZeros::Equal.into(),
        nan_order: 0,
        nested_nulls: 0,
    }
    .encode_to_vec();

    assert!(Binary.deserialize(&bytes, &SESSION).is_err());
    Ok(())
}

/// A comparison with a literal that has no floats keeps the total order, so total-order rewrites
/// still apply to it. A float literal keeps the requested mode.
#[rstest]
#[case::integer(Scalar::from(1i32), CompareMode::TOTAL_ORDER)]
#[case::string(Scalar::from("a"), CompareMode::TOTAL_ORDER)]
#[case::float(Scalar::from(1.0f64), SPARK)]
#[case::integer_list(int_list_scalar(), CompareMode::TOTAL_ORDER.with_nested_nulls(NestedNulls::Last))]
fn compare_with_mode_keeps_total_order_without_floats(
    #[case] value: Scalar,
    #[case] expected: CompareMode,
) {
    let expr = compare_with_mode(CompareOperator::Eq, DUCKDB, root(), lit(value));

    let options = expr.as_::<Binary>();
    assert_eq!(options.compare_mode(), expected);
}

fn int_list_scalar() -> Scalar {
    Scalar::list(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        vec![1i32.into()],
        Nullability::NonNullable,
    )
}

/// Nested nulls order first or last as the mode requires, at each level. Nested nulls are equal
/// to each other. The rows are `{a: null} ? {a: 1}`, `{a: 1} ? {a: null}` and
/// `{a: null} ? {a: null}`.
#[rstest]
#[case::first_lt(CompareMode::TOTAL_ORDER, CompareOperator::Lt, [true, false, false])]
#[case::first_gt(CompareMode::TOTAL_ORDER, CompareOperator::Gt, [false, true, false])]
#[case::last_lt(DUCKDB, CompareOperator::Lt, [false, true, false])]
#[case::last_gt(DUCKDB, CompareOperator::Gt, [true, false, false])]
#[case::last_eq(DUCKDB, CompareOperator::Eq, [false, false, true])]
fn compare_mode_struct_nested_nulls(
    #[case] mode: CompareMode,
    #[case] op: CompareOperator,
    #[case] expected: [bool; 3],
) -> VortexResult<()> {
    let ctx = &mut SESSION.create_execution_ctx();
    let lhs = StructArray::from_fields(&[(
        "a",
        PrimitiveArray::from_option_iter([None, Some(1i32), None]).into_array(),
    )])?
    .into_array();
    let rhs = StructArray::from_fields(&[(
        "a",
        PrimitiveArray::from_option_iter([Some(1i32), None, None]).into_array(),
    )])?
    .into_array();

    let result = compare_with(lhs, rhs, op, mode)?;

    assert_arrays_eq!(result, BoolArray::from_iter(expected), ctx);
    Ok(())
}

/// A list element that is null orders after all values when nested nulls are last. The rows are
/// `[1, null] ? [1, 2]` and `[null] ? [5, 6]`.
#[rstest]
#[case::first_lt(CompareMode::TOTAL_ORDER, CompareOperator::Lt, [true, true])]
#[case::last_lt(DUCKDB, CompareOperator::Lt, [false, false])]
#[case::last_gt(DUCKDB, CompareOperator::Gt, [true, true])]
fn compare_mode_list_nested_nulls(
    #[case] mode: CompareMode,
    #[case] op: CompareOperator,
    #[case] expected: [bool; 2],
) -> VortexResult<()> {
    let ctx = &mut SESSION.create_execution_ctx();
    let lhs = ListArray::try_new(
        PrimitiveArray::from_option_iter([Some(1i32), None, None]).into_array(),
        PrimitiveArray::from_iter([0i32, 2, 3]).into_array(),
        Validity::NonNullable,
        ctx,
    )?
    .into_array();
    let rhs = ListArray::try_new(
        PrimitiveArray::from_iter([1i32, 2, 5, 6]).into_array(),
        PrimitiveArray::from_iter([0i32, 2, 4]).into_array(),
        Validity::NonNullable,
        ctx,
    )?
    .into_array();

    let result = compare_with(lhs, rhs, op, mode)?;

    assert_arrays_eq!(result, BoolArray::from_iter(expected), ctx);
    Ok(())
}

/// Nested nulls that order last disagree with the total order only for nested values.
#[test]
fn nested_nulls_agree_with_total_order_only_when_flat() {
    let mode = CompareMode::TOTAL_ORDER.with_nested_nulls(NestedNulls::Last);
    let list = int_list_scalar();

    assert!(!mode.agrees_with_total_order(CompareOperator::Lt, list.dtype(), Some(&list)));
    assert!(mode.agrees_with_total_order(
        CompareOperator::Lt,
        &DType::Primitive(PType::I32, Nullability::NonNullable),
        None
    ));
}
