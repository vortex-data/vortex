// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_dtype::DType;
use vortex_dtype::FieldName;
use vortex_dtype::FieldNames;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexResult;

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

// Field validity must include struct null positions for pushed_down invariant.
// STRUCT: nulls at positions 1, 3
// FIELD:  nulls at positions 1, 2, 3 (superset of struct nulls)
const STRUCT_VALIDITY: [bool; 4] = [true, false, true, false];
const FIELD_VALIDITY: [bool; 4] = [true, false, false, false];

fn field_validity(nullable: bool) -> Validity {
    if nullable {
        Validity::Array(BoolArray::from_iter(FIELD_VALIDITY).into_array())
    } else {
        Validity::NonNullable
    }
}

fn struct_validity(nullable: bool) -> Validity {
    if nullable {
        Validity::Array(BoolArray::from_iter(STRUCT_VALIDITY).into_array())
    } else {
        Validity::NonNullable
    }
}

#[rstest]
#[case::both_non_nullable_struct_non_nullable(false, false, false)]
#[case::both_non_nullable_struct_nullable(false, false, true)]
#[case::field_a_nullable_struct_non_nullable(true, false, false)]
#[case::field_a_nullable_struct_nullable(true, false, true)]
#[case::field_b_nullable_struct_non_nullable(false, true, false)]
#[case::field_b_nullable_struct_nullable(false, true, true)]
#[case::both_nullable_struct_non_nullable(true, true, false)]
#[case::both_nullable_struct_nullable(true, true, true)]
fn test_masked_fields(
    #[case] field_a_nullable: bool,
    #[case] field_b_nullable: bool,
    #[case] struct_nullable: bool,
    #[values(false, true)] should_compact: bool,
) -> VortexResult<()> {
    let field_a_val = field_validity(field_a_nullable);
    let field_b_val = field_validity(field_b_nullable);
    let struct_val = struct_validity(struct_nullable);
    let field_a = PrimitiveArray::new(buffer![10i32, 20, 30, 40], field_a_val.clone());
    let field_b = PrimitiveArray::new(buffer![100i32, 200, 300, 400], field_b_val.clone());

    let mut struct_array = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![field_a.into_array(), field_b.into_array()],
        4,
        struct_val.clone(),
    )?;

    let before_dtype = struct_array.dtype().clone();

    if should_compact {
        struct_array = struct_array.compact()?;
    }

    assert_eq!(struct_array.dtype(), &before_dtype);
    if struct_val.nullability().is_nullable() {
        assert_eq!(struct_array.has_validity_pushed_down(), should_compact);
    }

    assert_eq!(struct_array.validity()?, struct_val);

    let combined_a =
        if field_a_val.nullability().is_nullable() || struct_val.nullability().is_nullable() {
            field_a_val.mask(&struct_val.to_mask(struct_array.len()).not())
        } else {
            Validity::NonNullable
        };

    let combined_b =
        if field_b_val.nullability().is_nullable() || struct_val.nullability().is_nullable() {
            field_b_val.mask(&struct_val.to_mask(struct_array.len()).not())
        } else {
            Validity::NonNullable
        };

    // Test masked_fields
    let masked = struct_array.masked_fields()?;
    assert_eq!(masked.len(), 2);
    assert_eq!(masked[0].validity()?, combined_a);
    assert_eq!(masked[1].validity()?, combined_b);

    // Test field_by_name
    let field_a_by_name = struct_array.field_by_name("a")?;
    assert_eq!(field_a_by_name.validity()?, combined_a);

    let field_b_by_name = struct_array.field_by_name("b")?;
    assert_eq!(field_b_by_name.validity()?, combined_b);

    Ok(())
}
