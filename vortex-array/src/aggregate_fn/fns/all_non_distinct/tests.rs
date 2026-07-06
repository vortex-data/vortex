// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::all_non_distinct;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::DecimalArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::NullArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::validity::Validity;

/// Baseline oracle: compare two arrays element-wise using `execute_scalar`.
/// Returns true iff every position has the same scalar (null == null is true).
fn scalar_baseline(a: &ArrayRef, b: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        let sa = a.execute_scalar(i, ctx)?;
        let sb = b.execute_scalar(i, ctx)?;
        if sa != sb {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Assert that `all_non_distinct` agrees with the scalar baseline.
fn assert_matches_baseline(a: &ArrayRef, b: &ArrayRef) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let expected = scalar_baseline(a, b, &mut ctx)?;
    let actual = all_non_distinct(a, b, &mut ctx)?;
    assert_eq!(
        actual, expected,
        "all_non_distinct disagrees with scalar baseline for arrays:\n  a: {:?}\n  b: {:?}",
        a, b
    );
    Ok(())
}

// ─── Null arrays ─────────────────────────────────────────────────────────────

#[test]
fn null_arrays_identical() -> VortexResult<()> {
    let a = NullArray::new(3).into_array();
    let b = NullArray::new(3).into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn null_arrays_empty() -> VortexResult<()> {
    let a = NullArray::new(0).into_array();
    let b = NullArray::new(0).into_array();
    assert_matches_baseline(&a, &b)
}

// ─── Bool ────────────────────────────────────────────────────────────────────

#[rstest]
#[case::identical(&[true, false, true], &[true, false, true], true)]
#[case::different(&[true, false, true], &[true, true, true], false)]
#[case::empty(&[], &[], true)]
#[case::single_true(&[true], &[true], true)]
#[case::single_false(&[false], &[false], true)]
#[case::single_mismatch(&[true], &[false], false)]
#[case::all_true(&[true, true, true], &[true, true, true], true)]
#[case::all_false(&[false, false, false], &[false, false, false], true)]
fn bool_non_nullable(
    #[case] a: &[bool],
    #[case] b: &[bool],
    #[case] expected: bool,
) -> VortexResult<()> {
    let a = BoolArray::from_iter(a.iter().copied()).into_array();
    let b = BoolArray::from_iter(b.iter().copied()).into_array();
    let mut ctx = array_session().create_execution_ctx();
    assert_eq!(all_non_distinct(&a, &b, &mut ctx)?, expected);
    assert_matches_baseline(&a, &b)
}

#[rstest]
#[case::same_nulls(
    &[Some(true), None, Some(false)],
    &[Some(true), None, Some(false)],
    true
)]
#[case::different_under_null(
    &[Some(true), None, Some(false)],
    &[Some(true), None, Some(true)],
    false
)]
#[case::null_vs_value(
    &[Some(true), None],
    &[Some(true), Some(false)],
    false
)]
#[case::all_null(
    &[None, None, None],
    &[None, None, None],
    true
)]
fn bool_nullable(
    #[case] a: &[Option<bool>],
    #[case] b: &[Option<bool>],
    #[case] expected: bool,
) -> VortexResult<()> {
    let a = BoolArray::from_iter(a.iter().copied()).into_array();
    let b = BoolArray::from_iter(b.iter().copied()).into_array();
    let mut ctx = array_session().create_execution_ctx();
    assert_eq!(all_non_distinct(&a, &b, &mut ctx)?, expected);
    assert_matches_baseline(&a, &b)
}

// ─── Primitive (multiple ptypes) ─────────────────────────────────────────────

#[rstest]
#[case::i32_identical(vec![1i32, 2, 3], vec![1i32, 2, 3], true)]
#[case::i32_different(vec![1i32, 2, 3], vec![1i32, 2, 4], false)]
#[case::i32_empty(vec![], vec![], true)]
#[case::i32_single(vec![42i32], vec![42i32], true)]
fn primitive_i32(
    #[case] a: Vec<i32>,
    #[case] b: Vec<i32>,
    #[case] expected: bool,
) -> VortexResult<()> {
    let a = PrimitiveArray::from_iter(a).into_array();
    let b = PrimitiveArray::from_iter(b).into_array();
    let mut ctx = array_session().create_execution_ctx();
    assert_eq!(all_non_distinct(&a, &b, &mut ctx)?, expected);
    assert_matches_baseline(&a, &b)
}

#[test]
fn primitive_f64() -> VortexResult<()> {
    let a = buffer![1.0f64, 2.5, 3.0].into_array();
    let b = buffer![1.0f64, 2.5, 3.0].into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn primitive_f64_different() -> VortexResult<()> {
    let a = buffer![1.0f64, 2.5, 3.0].into_array();
    let b = buffer![1.0f64, 2.5, 3.1].into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn primitive_u8() -> VortexResult<()> {
    let a = buffer![0u8, 128, 255].into_array();
    let b = buffer![0u8, 128, 255].into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn primitive_i64() -> VortexResult<()> {
    let a = buffer![i64::MIN, 0i64, i64::MAX].into_array();
    let b = buffer![i64::MIN, 0i64, i64::MAX].into_array();
    assert_matches_baseline(&a, &b)
}

#[rstest]
#[case::same_nulls(
    &[Some(1i32), None, Some(3)],
    &[Some(1i32), None, Some(3)],
    true
)]
#[case::null_vs_value(
    &[Some(1i32), None, Some(3)],
    &[Some(1i32), Some(2), Some(3)],
    false
)]
#[case::different_values_under_valid(
    &[Some(1i32), None, Some(3)],
    &[Some(1i32), None, Some(99)],
    false
)]
#[case::all_null(
    &[None, None],
    &[None, None],
    true
)]
#[case::single_null(
    &[None],
    &[None],
    true
)]
fn primitive_nullable(
    #[case] a: &[Option<i32>],
    #[case] b: &[Option<i32>],
    #[case] expected: bool,
) -> VortexResult<()> {
    let a = PrimitiveArray::from_option_iter(a.iter().copied()).into_array();
    let b = PrimitiveArray::from_option_iter(b.iter().copied()).into_array();
    let mut ctx = array_session().create_execution_ctx();
    assert_eq!(all_non_distinct(&a, &b, &mut ctx)?, expected);
    assert_matches_baseline(&a, &b)
}

// ─── VarBinView (strings / binary) ──────────────────────────────────────────

#[rstest]
#[case::identical(&["hello", "world"], &["hello", "world"], true)]
#[case::different(&["hello", "world"], &["hello", "earth"], false)]
#[case::empty_strings(&["", ""], &["", ""], true)]
#[case::single(&["abc"], &["abc"], true)]
#[case::long_strings(
    &["a]long string that exceeds inline"],
    &["a]long string that exceeds inline"],
    true
)]
#[case::long_strings_different(
    &["a long string that exceeds inline"],
    &["a long string that exceeds OOPS!"],
    false
)]
fn strings_non_nullable(
    #[case] a: &[&str],
    #[case] b: &[&str],
    #[case] expected: bool,
) -> VortexResult<()> {
    let a = VarBinViewArray::from_iter_str(a.iter().copied()).into_array();
    let b = VarBinViewArray::from_iter_str(b.iter().copied()).into_array();
    let mut ctx = array_session().create_execution_ctx();
    assert_eq!(all_non_distinct(&a, &b, &mut ctx)?, expected);
    assert_matches_baseline(&a, &b)
}

#[rstest]
#[case::same_nulls(
    &[Some("hi"), None, Some("lo")],
    &[Some("hi"), None, Some("lo")],
    true
)]
#[case::null_vs_value(
    &[Some("hi"), None],
    &[Some("hi"), Some("lo")],
    false
)]
fn strings_nullable(
    #[case] a: &[Option<&str>],
    #[case] b: &[Option<&str>],
    #[case] expected: bool,
) -> VortexResult<()> {
    let a = VarBinViewArray::from_iter_nullable_str(a.iter().copied()).into_array();
    let b = VarBinViewArray::from_iter_nullable_str(b.iter().copied()).into_array();
    let mut ctx = array_session().create_execution_ctx();
    assert_eq!(all_non_distinct(&a, &b, &mut ctx)?, expected);
    assert_matches_baseline(&a, &b)
}

// ─── Decimal ─────────────────────────────────────────────────────────────────

#[test]
fn decimal_identical() -> VortexResult<()> {
    let dtype = DecimalDType::new(5, 2);
    let a = DecimalArray::new(buffer![100i32, 200, 300], dtype, Validity::NonNullable).into_array();
    let b = DecimalArray::new(buffer![100i32, 200, 300], dtype, Validity::NonNullable).into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn decimal_different() -> VortexResult<()> {
    let dtype = DecimalDType::new(5, 2);
    let a = DecimalArray::new(buffer![100i32, 200, 300], dtype, Validity::NonNullable).into_array();
    let b = DecimalArray::new(buffer![100i32, 200, 999], dtype, Validity::NonNullable).into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn decimal_different_value_types() -> VortexResult<()> {
    let dtype = DecimalDType::new(3, 0);
    let a = DecimalArray::new(buffer![1i8, 2, 3], dtype, Validity::NonNullable).into_array();
    let b = DecimalArray::new(buffer![1i16, 2, 3], dtype, Validity::NonNullable).into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn decimal_nullable() -> VortexResult<()> {
    let dtype = DecimalDType::new(5, 2);
    let validity = Validity::from_iter([true, false, true]);
    let a = DecimalArray::new(buffer![100i32, 200, 300], dtype, validity.clone()).into_array();
    let b = DecimalArray::new(buffer![100i32, 999, 300], dtype, validity).into_array();
    assert_matches_baseline(&a, &b)
}

// ─── Struct ──────────────────────────────────────────────────────────────────

#[test]
fn struct_identical() -> VortexResult<()> {
    let a = StructArray::try_new(
        FieldNames::from(["x", "y"]),
        vec![
            buffer![1i32, 2].into_array(),
            buffer![10i32, 20].into_array(),
        ],
        2,
        Validity::NonNullable,
    )?
    .into_array();
    let b = StructArray::try_new(
        FieldNames::from(["x", "y"]),
        vec![
            buffer![1i32, 2].into_array(),
            buffer![10i32, 20].into_array(),
        ],
        2,
        Validity::NonNullable,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn struct_different() -> VortexResult<()> {
    let a = StructArray::try_new(
        FieldNames::from(["x", "y"]),
        vec![
            buffer![1i32, 2].into_array(),
            buffer![10i32, 20].into_array(),
        ],
        2,
        Validity::NonNullable,
    )?
    .into_array();
    let b = StructArray::try_new(
        FieldNames::from(["x", "y"]),
        vec![
            buffer![1i32, 2].into_array(),
            buffer![10i32, 99].into_array(),
        ],
        2,
        Validity::NonNullable,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn struct_nullable_ignores_garbage_under_nulls() -> VortexResult<()> {
    let validity = Validity::from_iter([true, false]);
    let a = StructArray::try_new(
        FieldNames::from(["x", "y"]),
        vec![
            buffer![1i32, 2].into_array(),
            buffer![10i32, 20].into_array(),
        ],
        2,
        validity.clone(),
    )?
    .into_array();
    let b = StructArray::try_new(
        FieldNames::from(["x", "y"]),
        vec![
            buffer![1i32, 99].into_array(),
            buffer![10i32, 999].into_array(),
        ],
        2,
        validity,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn struct_nested_nullable_fields() -> VortexResult<()> {
    let a = StructArray::try_new(
        FieldNames::from(["x"]),
        vec![PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array()],
        3,
        Validity::NonNullable,
    )?
    .into_array();
    let b = StructArray::try_new(
        FieldNames::from(["x"]),
        vec![PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array()],
        3,
        Validity::NonNullable,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

// ─── List ────────────────────────────────────────────────────────────────────

#[test]
fn list_identical() -> VortexResult<()> {
    let elements = buffer![1i32, 2, 3, 4].into_array();
    let a = ListViewArray::try_new(
        elements.clone(),
        buffer![0u32, 2].into_array(),
        buffer![2u32, 2].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let b = ListViewArray::try_new(
        elements,
        buffer![0u32, 2].into_array(),
        buffer![2u32, 2].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn list_different_elements() -> VortexResult<()> {
    let a = ListViewArray::try_new(
        buffer![1i32, 2, 3, 4].into_array(),
        buffer![0u32, 2].into_array(),
        buffer![2u32, 2].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let b = ListViewArray::try_new(
        buffer![1i32, 2, 3, 99].into_array(),
        buffer![0u32, 2].into_array(),
        buffer![2u32, 2].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn list_different_lengths() -> VortexResult<()> {
    let a = ListViewArray::try_new(
        buffer![1i32, 2, 3].into_array(),
        buffer![0u32, 2].into_array(),
        buffer![2u32, 1].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let b = ListViewArray::try_new(
        buffer![1i32, 2, 3].into_array(),
        buffer![0u32, 2].into_array(),
        buffer![2u32, 1].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn list_nullable_ignores_garbage() -> VortexResult<()> {
    let validity = Validity::from_iter([true, false]);
    let a = ListViewArray::try_new(
        buffer![1i32, 2, 3, 4].into_array(),
        buffer![0u8, 2].into_array(),
        buffer![2u8, 2].into_array(),
        validity.clone(),
    )?
    .into_array();
    let b = ListViewArray::try_new(
        buffer![1i32, 2, 9, 8].into_array(),
        buffer![0u8, 2].into_array(),
        buffer![2u8, 2].into_array(),
        validity,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn list_different_offset_dtypes() -> VortexResult<()> {
    let elements = buffer![1i32, 2, 3, 4].into_array();
    let a = ListViewArray::try_new(
        elements.clone(),
        buffer![0u8, 2].into_array(),
        buffer![2u8, 2].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let b = ListViewArray::try_new(
        elements,
        buffer![0i16, 2].into_array(),
        buffer![2i16, 2].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

// ─── FixedSizeList ───────────────────────────────────────────────────────────

#[test]
fn fixed_size_list_identical() -> VortexResult<()> {
    let a = FixedSizeListArray::try_new(
        buffer![1i32, 2, 3, 4].into_array(),
        2,
        Validity::NonNullable,
        2,
    )?
    .into_array();
    let b = FixedSizeListArray::try_new(
        buffer![1i32, 2, 3, 4].into_array(),
        2,
        Validity::NonNullable,
        2,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn fixed_size_list_different() -> VortexResult<()> {
    let a = FixedSizeListArray::try_new(
        buffer![1i32, 2, 3, 4].into_array(),
        2,
        Validity::NonNullable,
        2,
    )?
    .into_array();
    let b = FixedSizeListArray::try_new(
        buffer![1i32, 2, 3, 99].into_array(),
        2,
        Validity::NonNullable,
        2,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn fixed_size_list_nullable_ignores_garbage() -> VortexResult<()> {
    let validity = Validity::from_iter([true, false]);
    let a =
        FixedSizeListArray::try_new(buffer![1i32, 2, 3, 4].into_array(), 2, validity.clone(), 2)?
            .into_array();
    let b = FixedSizeListArray::try_new(buffer![1i32, 2, 9, 8].into_array(), 2, validity, 2)?
        .into_array();
    assert_matches_baseline(&a, &b)
}

// ─── Chunked ─────────────────────────────────────────────────────────────────

#[test]
fn chunked_identical() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let a = ChunkedArray::try_new(
        vec![buffer![1i32, 2].into_array(), buffer![3i32, 4].into_array()],
        dtype.clone(),
    )?
    .into_array();
    let b = ChunkedArray::try_new(
        vec![buffer![1i32, 2].into_array(), buffer![3i32, 4].into_array()],
        dtype,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn chunked_different() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let a = ChunkedArray::try_new(
        vec![buffer![1i32, 2].into_array(), buffer![3i32, 4].into_array()],
        dtype.clone(),
    )?
    .into_array();
    let b = ChunkedArray::try_new(
        vec![
            buffer![1i32, 2].into_array(),
            buffer![3i32, 99].into_array(),
        ],
        dtype,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn chunked_different_chunk_boundaries() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let a = ChunkedArray::try_new(
        vec![buffer![1i32, 2, 3].into_array(), buffer![4i32].into_array()],
        dtype.clone(),
    )?
    .into_array();
    let b = ChunkedArray::try_new(
        vec![buffer![1i32].into_array(), buffer![2i32, 3, 4].into_array()],
        dtype,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn chunked_empty() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let a = ChunkedArray::try_new(vec![], dtype.clone())?.into_array();
    let b = ChunkedArray::try_new(vec![], dtype)?.into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn chunked_with_nullable() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
    let a = ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_option_iter([Some(1i32), None]).into_array(),
            PrimitiveArray::from_option_iter([Some(3i32)]).into_array(),
        ],
        dtype.clone(),
    )?
    .into_array();
    let b = ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_option_iter([Some(1i32), None]).into_array(),
            PrimitiveArray::from_option_iter([Some(3i32)]).into_array(),
        ],
        dtype,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

// ─── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn single_element_identical() -> VortexResult<()> {
    let a = buffer![42i32].into_array();
    let b = buffer![42i32].into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn single_element_different() -> VortexResult<()> {
    let a = buffer![42i32].into_array();
    let b = buffer![43i32].into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn all_nulls_both_sides() -> VortexResult<()> {
    let a = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
    let b = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn first_element_different() -> VortexResult<()> {
    let a = buffer![99i32, 2, 3, 4, 5].into_array();
    let b = buffer![1i32, 2, 3, 4, 5].into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn last_element_different() -> VortexResult<()> {
    let a = buffer![1i32, 2, 3, 4, 5].into_array();
    let b = buffer![1i32, 2, 3, 4, 99].into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn middle_element_different() -> VortexResult<()> {
    let a = buffer![1i32, 2, 3, 4, 5].into_array();
    let b = buffer![1i32, 2, 99, 4, 5].into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn alternating_nulls_identical() -> VortexResult<()> {
    let a =
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)]).into_array();
    let b =
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)]).into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn alternating_nulls_different_value() -> VortexResult<()> {
    let a =
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)]).into_array();
    let b =
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(99), None, Some(5)]).into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn leading_nulls() -> VortexResult<()> {
    let a = PrimitiveArray::from_option_iter([None, None, Some(3i32)]).into_array();
    let b = PrimitiveArray::from_option_iter([None, None, Some(3i32)]).into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn trailing_nulls() -> VortexResult<()> {
    let a = PrimitiveArray::from_option_iter([Some(1i32), None, None]).into_array();
    let b = PrimitiveArray::from_option_iter([Some(1i32), None, None]).into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn large_array_identical() -> VortexResult<()> {
    let values: Vec<i32> = (0..1000).collect();
    let a = PrimitiveArray::from_iter(values.clone()).into_array();
    let b = PrimitiveArray::from_iter(values).into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn large_array_last_differs() -> VortexResult<()> {
    let mut values_a: Vec<i32> = (0..1000).collect();
    let mut values_b = values_a.clone();
    values_a[999] = 0;
    values_b[999] = 1;
    let a = PrimitiveArray::from_iter(values_a).into_array();
    let b = PrimitiveArray::from_iter(values_b).into_array();
    assert_matches_baseline(&a, &b)
}

// ─── Cross-encoding (mixed non-canonical arrays) ────────────────────────────

#[test]
fn chunked_vs_primitive() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let a = ChunkedArray::try_new(
        vec![buffer![1i32, 2].into_array(), buffer![3i32, 4].into_array()],
        dtype,
    )?
    .into_array();
    let b = buffer![1i32, 2, 3, 4].into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn chunked_vs_primitive_different() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let a = ChunkedArray::try_new(
        vec![buffer![1i32, 2].into_array(), buffer![3i32, 4].into_array()],
        dtype,
    )?
    .into_array();
    let b = buffer![1i32, 2, 3, 99].into_array();
    assert_matches_baseline(&a, &b)
}

// ─── Nested structs with nullable inner fields ──────────────────────────────

#[test]
fn struct_with_nullable_string_field() -> VortexResult<()> {
    let a = StructArray::try_new(
        FieldNames::from(["name", "age"]),
        vec![
            VarBinViewArray::from_iter_nullable_str([Some("alice"), None, Some("charlie")])
                .into_array(),
            PrimitiveArray::from_option_iter([Some(30i32), Some(25), None]).into_array(),
        ],
        3,
        Validity::NonNullable,
    )?
    .into_array();
    let b = StructArray::try_new(
        FieldNames::from(["name", "age"]),
        vec![
            VarBinViewArray::from_iter_nullable_str([Some("alice"), None, Some("charlie")])
                .into_array(),
            PrimitiveArray::from_option_iter([Some(30i32), Some(25), None]).into_array(),
        ],
        3,
        Validity::NonNullable,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}

#[test]
fn struct_with_nullable_string_field_different() -> VortexResult<()> {
    let a = StructArray::try_new(
        FieldNames::from(["name"]),
        vec![
            VarBinViewArray::from_iter_nullable_str([Some("alice"), None, Some("charlie")])
                .into_array(),
        ],
        3,
        Validity::NonNullable,
    )?
    .into_array();
    let b = StructArray::try_new(
        FieldNames::from(["name"]),
        vec![
            VarBinViewArray::from_iter_nullable_str([Some("alice"), None, Some("david")])
                .into_array(),
        ],
        3,
        Validity::NonNullable,
    )?
    .into_array();
    assert_matches_baseline(&a, &b)
}
