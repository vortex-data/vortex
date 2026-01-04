// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_dtype::DType;
use vortex_dtype::FieldName;
use vortex_dtype::FieldNames;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_scalar::Scalar;

use crate::Array;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::primitive::PrimitiveArray;
use crate::arrays::struct_::StructArray;
use crate::arrays::varbin::VarBinArray;
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
    assert_eq!(prims.to_primitive().as_slice::<i64>(), [0i64, 1, 2, 3, 4]);
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
    assert_eq!(removed.to_primitive().as_slice::<i64>(), [0i64, 1, 2, 3, 4]);

    assert_eq!(struct_a.names(), &["ys"]);
    assert_eq!(struct_a.fields.len(), 1);
    assert_eq!(struct_a.len(), 5);
    assert_eq!(
        struct_a.fields[0].dtype(),
        &DType::Primitive(PType::U64, Nullability::NonNullable)
    );
    assert_eq!(
        struct_a.fields[0].to_primitive().as_slice::<u64>(),
        [4u64, 5, 6, 7, 8]
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
    let first_value_field = struct_array.field_by_name("value").unwrap();
    assert_eq!(
        first_value_field.to_primitive().as_slice::<i32>(),
        [1i32, 2, 3] // This is field1, not field3
    );

    // Verify field_by_name_opt also returns the first match
    let opt_field = struct_array.field_by_name_opt("value").unwrap();
    assert_eq!(
        opt_field.to_primitive().as_slice::<i32>(),
        [1i32, 2, 3] // First "value" field
    );

    // Verify the third field (second "value") can be accessed by index
    let third_field = &struct_array.fields()[2];
    assert_eq!(
        third_field.to_primitive().as_slice::<i32>(),
        [100i32, 200, 300]
    );
}

#[test]
fn test_uncompressed_size_in_bytes() {
    let struct_array = StructArray::new(
        FieldNames::from(["integers"]),
        vec![ConstantArray::new(5, 1000).into_array()],
        1000,
        Validity::NonNullable,
    );

    let canonical_size = struct_array.to_canonical().into_array().nbytes();
    let uncompressed_size = struct_array
        .statistics()
        .compute_uncompressed_size_in_bytes();

    assert_eq!(canonical_size, 2);
    assert_eq!(uncompressed_size, Some(4000));
}

#[test]
fn test_push_validity_into_children_preserve_struct() {
    // Create struct with top-level nulls
    // structArray : [a, b]
    // fields: [1, 2, 3] (a), [10, 20, 30] (b)
    // validity: [true, false, true]
    // row 1 is null at struct level
    let struct_array = StructArray::try_new(
        ["a", "b"].into(),
        vec![
            buffer![1i32, 2i32, 3i32].into_array(),
            buffer![10i32, 20i32, 30i32].into_array(),
        ],
        3,
        Validity::from_iter([true, false, true]), // row 1 is null at struct level
    )
    .unwrap();

    // Push validity into children, preserving struct validity
    let pushed = struct_array.push_validity_into_children(true).unwrap();

    // Check that struct validity is preserved
    assert_eq!(pushed.validity_mask(), struct_array.validity_mask());

    // Check that children now have nulls where struct was null
    let field_a = pushed.fields()[0].as_ref();
    let field_b = pushed.fields()[1].as_ref();


    assert!(field_a.is_valid(0));
    assert!(!field_a.is_valid(1)); // Should be null due to struct null
    assert!(field_a.is_valid(2));

    assert!(field_b.is_valid(0));
    assert!(!field_b.is_valid(1)); // Should be null due to struct null
    assert!(field_b.is_valid(2));


    // Original values should be preserved where valid
    assert_eq!(field_a.scalar_at(0), 1i32.into());
    assert_eq!(field_a.scalar_at(2), 3i32.into());
    assert_eq!(field_b.scalar_at(0), 10i32.into());
    assert_eq!(field_b.scalar_at(2), 30i32.into());


    // Verify pushed struct array values (preserve_struct_validity = true)
    assert!(pushed.is_valid(0));  // Row 0 should be valid
    assert!(!pushed.is_valid(1)); // Row 1 should be null (preserved)
    assert!(pushed.is_valid(2));  // Row 2 should be valid

    // Row 0: {a: 1, b: 10} - should be valid struct with valid fields
    let row0 = pushed.scalar_at(0);
    assert!(row0.is_valid());

    // Row 1: null - should be null struct (preserved from original)
    let row1 = pushed.scalar_at(1);
    assert!(!row1.is_valid());

    // Row 2: {a: 3, b: 30} - should be valid struct with valid fields
    let row2 = pushed.scalar_at(2);
    assert!(row2.is_valid());

}

#[test]
fn test_push_validity_into_children_remove_struct() {

    // Create struct with top-level nulls
    let struct_array = StructArray::try_new(
        ["a", "b"].into(),
        vec![
            buffer![1i32, 2i32, 3i32].into_array(),
            buffer![10i32, 20i32, 30i32].into_array(),
        ],
        3,
        Validity::from_iter([true, false, true]), // row 1 is null at struct level
    )
    .unwrap();


    // Push validity into children, removing struct validity (preserve_struct_validity = false)
    let pushed = struct_array.push_validity_into_children(false).unwrap();


    // Check that struct validity is now NonNullable (struct itself cannot be null)
    // NonNullable means the struct instances themselves cannot be null
    assert!(pushed.validity_mask().all_true());

    // Check that children still have nulls where struct was null
    let field_a = pushed.fields()[0].as_ref();
    let field_b = pushed.fields()[1].as_ref();


    assert!(field_a.is_valid(0));
    assert!(!field_a.is_valid(1)); // Should be null    due to struct null
    assert!(field_a.is_valid(2));

    assert!(field_b.is_valid(0));
    assert!(!field_b.is_valid(1)); // Should be null due to struct null
    assert!(field_b.is_valid(2));


    // Original values should be preserved where valid
    assert_eq!(field_a.scalar_at(0), 1i32.into());
    assert_eq!(field_a.scalar_at(2), 3i32.into());
    assert_eq!(field_b.scalar_at(0), 10i32.into());
    assert_eq!(field_b.scalar_at(2), 30i32.into());

    // Verify null values using proper null scalar comparison
    use vortex_dtype::{DType, Nullability, PType};
    let null_i32_scalar = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
    assert_eq!(field_a.scalar_at(1), null_i32_scalar);
    assert_eq!(field_b.scalar_at(1), null_i32_scalar);

    // Alternative: check if the scalar is null
    assert!(!field_a.scalar_at(1).is_valid());
    assert!(!field_b.scalar_at(1).is_valid());

    // Verify pushed struct array values (preserve_struct_validity = false)
    assert!(pushed.is_valid(0)); // Row 0 should be valid
    assert!(pushed.is_valid(1)); // Row 1 should be valid (validity removed)
    assert!(pushed.is_valid(2)); // Row 2 should be valid

    // Row 0: {a: 1, b: 10} - should be valid struct with valid fields
    let row0 = pushed.scalar_at(0);
    assert!(row0.is_valid());

    // Row 1: {a: null, b: null} - should be valid struct but with null fields
    let row1 = pushed.scalar_at(1);
    assert!(row1.is_valid()); // Struct is valid, but fields are null

    // Row 2: {a: 3, b: 30} - should be valid struct with valid fields
    let row2 = pushed.scalar_at(2);
    assert!(row2.is_valid());

}

#[test]
fn test_push_validity_into_children_no_nulls() {
    // Create struct without any nulls
    let struct_array = StructArray::try_new(
        ["a", "b"].into(),
        vec![
            buffer![1i32, 2i32, 3i32].into_array(),
            buffer![10i32, 20i32, 30i32].into_array(),
        ],
        3,
        Validity::AllValid,
    )
    .unwrap();


    // Push validity into children (should be no-op when preserve=true)
    let pushed_preserve = struct_array.push_validity_into_children(true).unwrap();
    assert_eq!(pushed_preserve.validity_mask(), struct_array.validity_mask());

    // Push validity into children (should change validity to NonNullable when preserve=false)
    let pushed_remove = struct_array.push_validity_into_children(false).unwrap();
    assert!(pushed_remove.validity_mask().all_true());

    // Fields should remain unchanged
    for i in 0..struct_array.fields().len() {
        assert_eq!(
            pushed_preserve.fields()[i].scalar_at(0),
            struct_array.fields()[i].scalar_at(0)
        );
        assert_eq!(
            pushed_preserve.fields()[i].scalar_at(1),
            struct_array.fields()[i].scalar_at(1)
        );
        assert_eq!(
            pushed_preserve.fields()[i].scalar_at(2),
            struct_array.fields()[i].scalar_at(2)
        );
    }

}
