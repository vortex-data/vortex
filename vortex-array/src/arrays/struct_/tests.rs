// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
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

    let bools = struct_b.unmasked_field(0);
    assert_arrays_eq!(
        bools,
        BoolArray::from_iter([true, true, true, false, false])
    );

    let prims = struct_b.unmasked_field(1);
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
    assert_eq!(struct_a.struct_fields().nfields(), 1);
    assert_eq!(struct_a.len(), 5);
    assert_eq!(
        struct_a.unmasked_field(0).dtype(),
        &DType::Primitive(PType::U64, Nullability::NonNullable)
    );
    assert_arrays_eq!(
        struct_a.unmasked_field(0),
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
    let third_field = struct_array.unmasked_field(2);
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
