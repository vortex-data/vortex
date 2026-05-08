// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::all_non_distinct;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::DecimalArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::validity::Validity;

#[test]
fn identical_primitives() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let a = buffer![1i32, 2, 3].into_array();
    let b = buffer![1i32, 2, 3].into_array();
    assert!(all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn different_primitives() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let a = buffer![1i32, 2, 3].into_array();
    let b = buffer![1i32, 2, 4].into_array();
    assert!(!all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn identical_with_nulls() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let a = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
    let b = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
    assert!(all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn different_nulls() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let a = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
    let b = PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3)]).into_array();
    assert!(!all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn identical_empty() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let a = PrimitiveArray::from_iter(Vec::<i32>::new()).into_array();
    let b = PrimitiveArray::from_iter(Vec::<i32>::new()).into_array();
    assert!(all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn identical_bools() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let a = BoolArray::from_iter([true, false, true]).into_array();
    let b = BoolArray::from_iter([true, false, true]).into_array();
    assert!(all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn different_bools() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let a = BoolArray::from_iter([true, false, true]).into_array();
    let b = BoolArray::from_iter([true, true, true]).into_array();
    assert!(!all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn identical_strings() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let a = VarBinViewArray::from_iter_str(["hello", "world"]).into_array();
    let b = VarBinViewArray::from_iter_str(["hello", "world"]).into_array();
    assert!(all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn different_strings() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let a = VarBinViewArray::from_iter_str(["hello", "world"]).into_array();
    let b = VarBinViewArray::from_iter_str(["hello", "earth"]).into_array();
    assert!(!all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn identical_structs() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
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
    assert!(all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn different_structs() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
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
    assert!(!all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn identical_structs_ignore_values_under_null_rows() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
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

    assert!(all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[rstest]
#[case(vec![1i32, 2, 3], vec![1i32, 2, 3], true)]
#[case(vec![1i32, 2, 3], vec![1i32, 2, 4], false)]
fn parameterized_primitive(
    #[case] a: Vec<i32>,
    #[case] b: Vec<i32>,
    #[case] expected: bool,
) -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let a = PrimitiveArray::from_iter(a).into_array();
    let b = PrimitiveArray::from_iter(b).into_array();
    assert_eq!(all_non_distinct(&a, &b, &mut ctx)?, expected);
    Ok(())
}

#[test]
fn identical_chunked() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let a = ChunkedArray::try_new(
        vec![buffer![1i32, 2].into_array(), buffer![3i32, 4].into_array()],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )?
    .into_array();
    let b = ChunkedArray::try_new(
        vec![buffer![1i32, 2].into_array(), buffer![3i32, 4].into_array()],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )?
    .into_array();
    assert!(all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn identical_lists_with_different_offset_dtypes() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
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

    assert!(all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn identical_decimals_with_different_value_types() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let decimal_dtype = DecimalDType::new(3, 0);
    let a =
        DecimalArray::new(buffer![1i8, 2, 3], decimal_dtype, Validity::NonNullable).into_array();
    let b =
        DecimalArray::new(buffer![1i16, 2, 3], decimal_dtype, Validity::NonNullable).into_array();

    assert!(all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn identical_lists_ignore_null_row_garbage() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
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
        buffer![0i16, 2].into_array(),
        buffer![2i16, 2].into_array(),
        validity,
    )?
    .into_array();

    assert!(all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}

#[test]
fn identical_fixed_size_lists_ignore_null_row_garbage() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let validity = Validity::from_iter([true, false]);
    let a =
        FixedSizeListArray::try_new(buffer![1i32, 2, 3, 4].into_array(), 2, validity.clone(), 2)?
            .into_array();
    let b = FixedSizeListArray::try_new(buffer![1i32, 2, 9, 8].into_array(), 2, validity, 2)?
        .into_array();

    assert!(all_non_distinct(&a, &b, &mut ctx)?);
    Ok(())
}
