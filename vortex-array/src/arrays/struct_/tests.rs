// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_dtype::DType;
use vortex_dtype::FieldName;
use vortex_dtype::FieldNames;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::primitive::PrimitiveArray;
use crate::arrays::struct_::StructArray;
use crate::arrays::varbin::VarBinArray;
use crate::assert_arrays_eq;
use crate::validity::Validity;

#[test]
fn test_project() {
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

    let bools = &struct_b.fields[0];
    assert_eq!(
        bools.to_bool().bit_buffer().iter().collect::<Vec<_>>(),
        vec![true, true, true, false, false]
    );

    let prims = &struct_b.fields[1];
    assert_arrays_eq!(prims, PrimitiveArray::from_iter([0i64, 1, 2, 3, 4]));
}

#[test]
fn test_remove_column() {
    let xs = PrimitiveArray::new(buffer![0i64, 1, 2, 3, 4], Validity::NonNullable);
    let ys = PrimitiveArray::new(buffer![4u64, 5, 6, 7, 8], Validity::NonNullable);

    let mut struct_a = StructArray::try_new(
        FieldNames::from(["xs", "ys"]),
        vec![xs.into_array(), ys.into_array()],
        5,
        Validity::NonNullable,
    )
    .unwrap();

    let removed = struct_a.remove_column("xs").unwrap();
    assert_eq!(
        removed.dtype(),
        &DType::Primitive(PType::I64, Nullability::NonNullable)
    );
    assert_arrays_eq!(removed, PrimitiveArray::from_iter([0i64, 1, 2, 3, 4]));

    assert_eq!(struct_a.names(), &["ys"]);
    assert_eq!(struct_a.fields.len(), 1);
    assert_eq!(struct_a.len(), 5);
    assert_eq!(
        struct_a.fields[0].dtype(),
        &DType::Primitive(PType::U64, Nullability::NonNullable)
    );
    assert_arrays_eq!(
        struct_a.fields[0],
        PrimitiveArray::from_iter([4u64, 5, 6, 7, 8])
    );

    let empty = struct_a.remove_column("non_existent");
    assert!(
        empty.is_none(),
        "Expected None when removing non-existent column"
    );
    assert_eq!(struct_a.names(), &["ys"]);
}

#[test]
fn test_duplicate_field_names() {
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
        PrimitiveArray::from_iter([1i32, 2, 3]) // This is field1, not field3
    );

    // Verify field_by_name_opt also returns the first match
    let opt_field = struct_array.unmasked_field_by_name_opt("value").unwrap();
    assert_arrays_eq!(
        opt_field,
        PrimitiveArray::from_iter([1i32, 2, 3]) // First "value" field
    );

    // Verify the third field (second "value") can be accessed by index
    let third_field = &struct_array.unmasked_fields()[2];
    assert_arrays_eq!(third_field, PrimitiveArray::from_iter([100i32, 200, 300]));
}

#[test]
fn test_uncompressed_size_in_bytes() -> VortexResult<()> {
    let struct_array = StructArray::new(
        FieldNames::from(["integers"]),
        vec![ConstantArray::new(5, 1000).into_array()],
        1000,
        Validity::NonNullable,
    );

    let canonical_size = struct_array.to_canonical()?.into_array().nbytes();
    let uncompressed_size = struct_array
        .statistics()
        .compute_uncompressed_size_in_bytes();

    assert_eq!(canonical_size, 2);
    assert_eq!(uncompressed_size, Some(4000));
    Ok(())
}

#[test]
fn test_masked_fields_without_validity_pushed_down() -> VortexResult<()> {
    // Create a nullable struct with struct-level nulls at positions 1 and 3
    // Fields are non-nullable, so struct validity should be applied to all fields
    let field_a = PrimitiveArray::new(buffer![10i32, 20, 30, 40], Validity::NonNullable);
    let field_b = PrimitiveArray::new(buffer![100i32, 200, 300, 400], Validity::NonNullable);

    let struct_validity = Validity::Array(
        BoolArray::from_iter([true, false, true, false]).into_array(), // positions 1, 3 are null
    );
    let expected_mask = Mask::from_iter([true, false, true, false]);

    let struct_array = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![field_a.into_array(), field_b.into_array()],
        4,
        struct_validity,
    )?;

    // Verify validity is NOT pushed down by default
    assert!(!struct_array.has_validity_pushed_down());

    // === Test masked_fields ===
    let masked = struct_array.masked_fields()?;
    assert_eq!(masked.len(), 2);

    // Both fields should have struct validity applied
    assert!(masked[0].dtype().is_nullable());
    assert_eq!(masked[0].validity_mask()?, expected_mask);
    assert!(masked[1].dtype().is_nullable());
    assert_eq!(masked[1].validity_mask()?, expected_mask);

    // === Test field_by_name ===
    let field_a_by_name = struct_array.field_by_name("a")?;
    assert!(field_a_by_name.dtype().is_nullable());
    assert_eq!(field_a_by_name.validity_mask()?, expected_mask);

    let field_b_by_name = struct_array.field_by_name("b")?;
    assert!(field_b_by_name.dtype().is_nullable());
    assert_eq!(field_b_by_name.validity_mask()?, expected_mask);

    Ok(())
}

#[test]
fn test_masked_fields_with_validity_pushed_down() -> VortexResult<()> {
    // Create nullable fields where validity has already been pushed down
    // (child nulls include struct nulls at positions 1 and 3)
    let field_a = PrimitiveArray::new(
        buffer![10i32, 20, 30, 40],
        Validity::Array(BoolArray::from_iter([true, false, true, false]).into_array()),
    );
    let field_b = PrimitiveArray::new(
        buffer![100i32, 200, 300, 400],
        Validity::Array(BoolArray::from_iter([true, false, true, false]).into_array()),
    );

    let struct_validity =
        Validity::Array(BoolArray::from_iter([true, false, true, false]).into_array());
    let expected_mask = Mask::from_iter([true, false, true, false]);

    let struct_array = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![field_a.into_array(), field_b.into_array()],
        4,
        struct_validity,
    )?
    .with_validity_pushed_down(true);

    // Verify validity IS pushed down
    assert!(struct_array.has_validity_pushed_down());

    // === Test masked_fields ===
    let masked = struct_array.masked_fields()?;
    assert_eq!(masked.len(), 2);

    // Nullable fields returned as-is (validity already pushed)
    assert!(masked[0].dtype().is_nullable());
    assert_eq!(masked[0].validity_mask()?, expected_mask);
    assert!(masked[1].dtype().is_nullable());
    assert_eq!(masked[1].validity_mask()?, expected_mask);

    // === Test field_by_name ===
    let field_a_by_name = struct_array.field_by_name("a")?;
    assert!(field_a_by_name.dtype().is_nullable());
    assert_eq!(field_a_by_name.validity_mask()?, expected_mask);

    let field_b_by_name = struct_array.field_by_name("b")?;
    assert!(field_b_by_name.dtype().is_nullable());
    assert_eq!(field_b_by_name.validity_mask()?, expected_mask);

    Ok(())
}

#[test]
fn test_masked_fields_mixed_nullability_with_pushed_down() -> VortexResult<()> {
    // Create a struct with mixed nullable and non-nullable fields
    // Nullable field has validity pushed down, non-nullable field needs masking
    let nullable_field = PrimitiveArray::new(
        buffer![10i32, 20, 30, 40],
        Validity::Array(BoolArray::from_iter([true, false, true, false]).into_array()),
    );
    let non_nullable_field =
        PrimitiveArray::new(buffer![100i32, 200, 300, 400], Validity::NonNullable);

    let struct_validity =
        Validity::Array(BoolArray::from_iter([true, false, true, false]).into_array());
    let expected_mask = Mask::from_iter([true, false, true, false]);

    let struct_array = StructArray::try_new(
        FieldNames::from(["nullable", "non_nullable"]),
        vec![nullable_field.into_array(), non_nullable_field.into_array()],
        4,
        struct_validity,
    )?
    .with_validity_pushed_down(true);

    assert!(struct_array.has_validity_pushed_down());

    // === Test masked_fields ===
    let masked = struct_array.masked_fields()?;

    // Nullable field: returned as-is (validity already pushed down)
    assert!(masked[0].dtype().is_nullable());
    assert_eq!(masked[0].validity_mask()?, expected_mask);

    // Non-nullable field: struct validity applied (becomes nullable)
    assert!(masked[1].dtype().is_nullable());
    assert_eq!(masked[1].validity_mask()?, expected_mask);

    // === Test field_by_name ===
    let nullable_by_name = struct_array.field_by_name("nullable")?;
    assert!(nullable_by_name.dtype().is_nullable());
    assert_eq!(nullable_by_name.validity_mask()?, expected_mask);

    let non_nullable_by_name = struct_array.field_by_name("non_nullable")?;
    assert!(non_nullable_by_name.dtype().is_nullable());
    assert_eq!(non_nullable_by_name.validity_mask()?, expected_mask);

    Ok(())
}

#[test]
fn test_masked_fields_non_nullable_struct() -> VortexResult<()> {
    // Non-nullable struct: fields should be returned unchanged
    let field_a = PrimitiveArray::new(buffer![10i32, 20, 30, 40], Validity::NonNullable);
    let field_b = PrimitiveArray::new(
        buffer![100i32, 200, 300, 400],
        Validity::Array(BoolArray::from_iter([true, true, false, true]).into_array()),
    );
    let expected_mask_all_valid = Mask::new_true(4);
    let expected_mask_b = Mask::from_iter([true, true, false, true]);

    let struct_array = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![field_a.into_array(), field_b.into_array()],
        4,
        Validity::NonNullable,
    )?;

    // Non-nullable struct cannot have validity pushed down
    assert!(!struct_array.has_validity_pushed_down());

    // with_validity_pushed_down should be a no-op for non-nullable structs
    let struct_array = struct_array.with_validity_pushed_down(true);
    assert!(!struct_array.has_validity_pushed_down());

    // === Test masked_fields ===
    let masked = struct_array.masked_fields()?;

    // Non-nullable field: stays non-nullable (no struct validity to apply)
    assert!(!masked[0].dtype().is_nullable());
    assert_eq!(masked[0].validity_mask()?, expected_mask_all_valid);

    // Nullable field: keeps its own validity
    assert!(masked[1].dtype().is_nullable());
    assert_eq!(masked[1].validity_mask()?, expected_mask_b);

    // === Test field_by_name ===
    let field_a_by_name = struct_array.field_by_name("a")?;
    assert!(!field_a_by_name.dtype().is_nullable());
    assert_eq!(field_a_by_name.validity_mask()?, expected_mask_all_valid);

    let field_b_by_name = struct_array.field_by_name("b")?;
    assert!(field_b_by_name.dtype().is_nullable());
    assert_eq!(field_b_by_name.validity_mask()?, expected_mask_b);

    Ok(())
}
