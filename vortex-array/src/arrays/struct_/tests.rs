// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinArray;
use crate::arrays::struct_::StructArrayExt;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::validity::Validity;

#[test]
fn test_project() {
    let mut ctx = array_session().create_execution_ctx();
    let xs = PrimitiveArray::new(buffer![0i64, 1, 2, 3, 4], Validity::NonNullable);
    let ys = VarBinArray::from_vec(
        vec!["a", "b", "c", "d", "e"],
        DType::Utf8(Nullability::NonNullable),
    );
    let zs = BoolArray::from_iter([true, true, true, false, false]);

    let struct_a = StructArray::try_new(
        FieldNames::from(["xs", "ys", "zs"]),
        vec![xs.into_array(), ys.into_array(), zs.into_array()],
        5,
        Validity::NonNullable,
    )
    .unwrap();

    let struct_b = struct_a
        .project(&[FieldName::from("zs"), FieldName::from("xs")])
        .unwrap();
    assert_eq!(
        struct_b.names().as_ref(),
        [FieldName::from("zs"), FieldName::from("xs")],
    );

    assert_eq!(struct_b.len(), 5);

    let bools = struct_b.unmasked_field(0);
    assert_arrays_eq!(
        bools,
        BoolArray::from_iter([true, true, true, false, false]),
        &mut ctx
    );

    let prims = struct_b.unmasked_field(1);
    assert_arrays_eq!(
        prims,
        PrimitiveArray::from_iter([0i64, 1, 2, 3, 4]),
        &mut ctx
    );
}

#[test]
fn test_remove_column() {
    let mut ctx = array_session().create_execution_ctx();
    let xs = PrimitiveArray::new(buffer![0i64, 1, 2, 3, 4], Validity::NonNullable);
    let ys = PrimitiveArray::new(buffer![4u64, 5, 6, 7, 8], Validity::NonNullable);

    let struct_a = StructArray::try_new(
        FieldNames::from(["xs", "ys"]),
        vec![xs.into_array(), ys.into_array()],
        5,
        Validity::NonNullable,
    )
    .unwrap();

    let (data, removed) = struct_a.remove_column("xs").unwrap();
    assert_eq!(
        removed.dtype(),
        &DType::Primitive(PType::I64, Nullability::NonNullable)
    );
    assert_arrays_eq!(
        removed,
        PrimitiveArray::from_iter([0i64, 1, 2, 3, 4]),
        &mut ctx
    );

    assert_eq!(data.names(), &["ys"]);
    assert_eq!(data.struct_fields().nfields(), 1);
    assert_eq!(data.len(), 5);
    assert_eq!(
        data.unmasked_field(0).dtype(),
        &DType::Primitive(PType::U64, Nullability::NonNullable)
    );
    assert_arrays_eq!(
        data.unmasked_field(0),
        PrimitiveArray::from_iter([4u64, 5, 6, 7, 8]),
        &mut ctx
    );

    let empty = data.remove_column("non_existent");
    assert!(
        empty.is_none(),
        "Expected None when removing non-existent column"
    );
    assert_eq!(data.names(), &["ys"]);
}

#[test]
fn test_duplicate_field_names() {
    let mut ctx = array_session().create_execution_ctx();
    // Test that StructArray allows duplicate field names and returns the first match
    let field1 = buffer![1i32, 2, 3].into_array();
    let field2 = buffer![10i32, 20, 30].into_array();
    let field3 = buffer![100i32, 200, 300].into_array();

    // Create struct with duplicate field names - "value" appears twice
    let struct_array = StructArray::try_new(
        FieldNames::from(["value", "other", "value"]),
        vec![field1, field2, field3],
        3,
        Validity::NonNullable,
    )
    .unwrap();

    // field_by_name should return the first field with the matching name
    let first_value_field = struct_array.unmasked_field_by_name("value").unwrap();
    assert_arrays_eq!(
        first_value_field,
        PrimitiveArray::from_iter([1i32, 2, 3]),
        &mut ctx
    );

    // Verify field_by_name_opt also returns the first match
    let opt_field = struct_array.unmasked_field_by_name_opt("value").unwrap();
    assert_arrays_eq!(opt_field, PrimitiveArray::from_iter([1i32, 2, 3]), &mut ctx);

    // Verify the third field (second "value") can be accessed by index
    let third_field = struct_array.unmasked_field(2);
    assert_arrays_eq!(
        third_field,
        PrimitiveArray::from_iter([100i32, 200, 300]),
        &mut ctx
    );
}

#[test]
fn test_uncompressed_size_in_bytes() -> VortexResult<()> {
    let struct_array = StructArray::new(
        FieldNames::from(["integers"]),
        vec![ConstantArray::new(5, 1000).into_array()],
        1000,
        Validity::NonNullable,
    );

    let canonical_size = struct_array
        .clone()
        .into_array()
        .execute::<Canonical>(&mut array_session().create_execution_ctx())?
        .into_array()
        .nbytes();
    let uncompressed_size = struct_array
        .statistics()
        .compute_uncompressed_size_in_bytes(&mut array_session().create_execution_ctx());

    assert_eq!(canonical_size, 2);
    assert_eq!(uncompressed_size, Some(4000));
    Ok(())
}

#[test]
fn test_push_validity_into_children_drops_struct_validity() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let struct_array = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![
            buffer![1i32, 2, 3].into_array(),
            buffer![10i32, 20, 30].into_array(),
        ],
        3,
        Validity::from_iter([true, false, true]),
    )?;

    let pushed = struct_array.push_validity_into_children(true)?;

    // The struct is now non-nullable; the row-1 null lives in every field instead.
    let expected = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![
            PrimitiveArray::new(
                buffer![1i32, 2, 3],
                Validity::from_iter([true, false, true]),
            )
            .into_array(),
            PrimitiveArray::new(
                buffer![10i32, 20, 30],
                Validity::from_iter([true, false, true]),
            )
            .into_array(),
        ],
        3,
        Validity::NonNullable,
    )?;

    assert!(!pushed.dtype().is_nullable());
    assert_arrays_eq!(pushed, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_push_validity_into_children_preserves_struct_validity() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let struct_array = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![
            buffer![1i32, 2, 3].into_array(),
            buffer![10i32, 20, 30].into_array(),
        ],
        3,
        Validity::from_iter([true, false, true]),
    )?;

    let pushed = struct_array.push_validity_into_children(false)?;

    // The null now exists both at the struct level and in every field.
    let expected = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![
            PrimitiveArray::new(
                buffer![1i32, 2, 3],
                Validity::from_iter([true, false, true]),
            )
            .into_array(),
            PrimitiveArray::new(
                buffer![10i32, 20, 30],
                Validity::from_iter([true, false, true]),
            )
            .into_array(),
        ],
        3,
        Validity::from_iter([true, false, true]),
    )?;

    assert!(pushed.dtype().is_nullable());
    assert_arrays_eq!(pushed, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_push_validity_into_children_intersects_field_validity() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();

    // Fields carry their own nulls (a at row 1, b at row 2) and the struct is null at row 1,
    // so pushing intersects both levels rather than overwriting the fields.
    let struct_array = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
            PrimitiveArray::from_option_iter([Some(10i64), Some(20), None]).into_array(),
        ],
        3,
        Validity::from_iter([true, false, true]),
    )?;

    let pushed = struct_array.push_validity_into_children(true)?;

    // a: null at row 1; b: null at rows 1 and 2.
    let expected = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
            PrimitiveArray::from_option_iter([Some(10i64), None, None]).into_array(),
        ],
        3,
        Validity::NonNullable,
    )?;

    assert_arrays_eq!(pushed, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_push_validity_into_children_all_invalid() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let struct_array = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![
            buffer![1i32, 2, 3].into_array(),
            buffer![10i32, 20, 30].into_array(),
        ],
        3,
        Validity::AllInvalid,
    )?;

    let pushed = struct_array.push_validity_into_children(true)?;

    // Every row is null at the struct level, so every field becomes all-null.
    let expected = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![
            PrimitiveArray::new(buffer![1i32, 2, 3], Validity::AllInvalid).into_array(),
            PrimitiveArray::new(buffer![10i32, 20, 30], Validity::AllInvalid).into_array(),
        ],
        3,
        Validity::NonNullable,
    )?;

    assert_arrays_eq!(pushed, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_push_validity_into_children_no_nulls() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();

    // No nulls: the fields are untouched, only the top-level nullability changes.
    let struct_array = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![
            buffer![1i32, 2, 3].into_array(),
            buffer![10i32, 20, 30].into_array(),
        ],
        3,
        Validity::AllValid,
    )?;

    let dropped = struct_array.push_validity_into_children(true)?;
    let expected = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![
            buffer![1i32, 2, 3].into_array(),
            buffer![10i32, 20, 30].into_array(),
        ],
        3,
        Validity::NonNullable,
    )?;
    assert!(!dropped.dtype().is_nullable());
    assert_arrays_eq!(dropped, expected, &mut ctx);

    let preserved = struct_array.push_validity_into_children(false)?;
    assert!(preserved.dtype().is_nullable());
    assert_arrays_eq!(preserved, struct_array, &mut ctx);
    Ok(())
}
