// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::buffer;

use crate::IntoArray;
use crate::ToCanonical;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::builders::ArrayBuilder;
use crate::builders::ListBuilder;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::scalar::Scalar;
use crate::validity::Validity;

////////////////////////////////////////////////////////////////////////////////////////////////////
// FSL of FSL tests
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_fsl_of_fsl_basic() {
    let outer_len = 2;
    let outer_list_size = 3;
    let inner_list_size = 2;

    // Create inner FSLs: [[1,2], [3,4], [5,6]], [[7,8], [9,10], [11,12]].
    // This needs 12 primitive elements total.
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_array();

    // First create the inner FSL array containing all inner lists.
    let inner_fsl = FixedSizeListArray::new(
        elements.into_array(),
        inner_list_size,
        Validity::NonNullable,
        outer_len * outer_list_size as usize,
    );

    // Now create the outer FSL.
    let outer_fsl = FixedSizeListArray::new(
        inner_fsl.into_array(),
        outer_list_size,
        Validity::NonNullable,
        outer_len,
    );

    assert_eq!(outer_fsl.len(), outer_len);
    assert_eq!(outer_fsl.list_size(), outer_list_size);

    // Check the dtype - should be FSL of FSL.
    assert!(matches!(
        outer_fsl.dtype(),
        DType::FixedSizeList(inner_dtype, 3, Nullability::NonNullable)
            if matches!(
                inner_dtype.as_ref(),
                DType::FixedSizeList(elem_dtype, 2, Nullability::NonNullable)
                    if matches!(elem_dtype.as_ref(),
                        DType::Primitive(PType::I32, Nullability::NonNullable))
            )
    ));

    // Get the first outer list.
    let first_outer = outer_fsl.fixed_size_list_elements_at(0).unwrap();
    assert_eq!(first_outer.len(), outer_list_size as usize);

    // The first outer list should contain 3 inner lists.
    // We can check by slicing and examining scalars.
    let first_scalar = outer_fsl.scalar_at(0).unwrap();
    assert!(!first_scalar.is_null());

    // Check the actual values in the nested structure.
    // First outer list contains: [[1,2], [3,4], [5,6]].
    let first_outer_list = outer_fsl.fixed_size_list_elements_at(0).unwrap();

    // Check first inner list [1,2].
    let inner_list_0 = first_outer_list
        .to_fixed_size_list()
        .fixed_size_list_elements_at(0)
        .unwrap();
    assert_eq!(inner_list_0.scalar_at(0).unwrap(), 1i32.into());
    assert_eq!(inner_list_0.scalar_at(1).unwrap(), 2i32.into());

    // Check second inner list [3,4].
    let inner_list_1 = first_outer_list
        .to_fixed_size_list()
        .fixed_size_list_elements_at(1)
        .unwrap();
    assert_eq!(inner_list_1.scalar_at(0).unwrap(), 3i32.into());
    assert_eq!(inner_list_1.scalar_at(1).unwrap(), 4i32.into());

    // Check third inner list [5,6].
    let inner_list_2 = first_outer_list
        .to_fixed_size_list()
        .fixed_size_list_elements_at(2)
        .unwrap();
    assert_eq!(inner_list_2.scalar_at(0).unwrap(), 5i32.into());
    assert_eq!(inner_list_2.scalar_at(1).unwrap(), 6i32.into());

    // Second outer list contains: [[7,8], [9,10], [11,12]].
    let second_outer_list = outer_fsl.fixed_size_list_elements_at(1).unwrap();

    // Check first inner list [7,8].
    let inner_list_0 = second_outer_list
        .to_fixed_size_list()
        .fixed_size_list_elements_at(0)
        .unwrap();
    assert_eq!(inner_list_0.scalar_at(0).unwrap(), 7i32.into());
    assert_eq!(inner_list_0.scalar_at(1).unwrap(), 8i32.into());

    // Check second inner list [9,10].
    let inner_list_1 = second_outer_list
        .to_fixed_size_list()
        .fixed_size_list_elements_at(1)
        .unwrap();
    assert_eq!(inner_list_1.scalar_at(0).unwrap(), 9i32.into());
    assert_eq!(inner_list_1.scalar_at(1).unwrap(), 10i32.into());

    // Check third inner list [11,12].
    let inner_list_2 = second_outer_list
        .to_fixed_size_list()
        .fixed_size_list_elements_at(2)
        .unwrap();
    assert_eq!(inner_list_2.scalar_at(0).unwrap(), 11i32.into());
    assert_eq!(inner_list_2.scalar_at(1).unwrap(), 12i32.into());
}

#[test]
fn test_fsl_of_fsl_with_nulls() {
    let outer_len = 3;
    let outer_list_size = 2;
    let inner_list_size = 2;

    // Create elements with some nulls.
    let elements = PrimitiveArray::from_option_iter(vec![
        Some(1i32),
        None,
        Some(3),
        Some(4),
        Some(5),
        Some(6),
        Some(7),
        None,
        Some(9),
        Some(10),
        None,
        Some(12),
    ]);

    // Inner FSL with nullable elements but non-null lists.
    let inner_fsl = FixedSizeListArray::new(
        elements.into_array(),
        inner_list_size,
        Validity::NonNullable,
        outer_len * outer_list_size as usize,
    );

    // Outer FSL with some null lists.
    let outer_validity = Validity::from_iter([true, false, true]);
    let outer_fsl = FixedSizeListArray::new(
        inner_fsl.into_array(),
        outer_list_size,
        outer_validity,
        outer_len,
    );

    assert_eq!(outer_fsl.len(), outer_len);

    // First outer list is valid.
    assert!(!outer_fsl.scalar_at(0).unwrap().is_null());

    // Second outer list is null.
    assert!(outer_fsl.scalar_at(1).unwrap().is_null());

    // Third outer list is valid.
    assert!(!outer_fsl.scalar_at(2).unwrap().is_null());
}

#[test]
fn test_deeply_nested_fsl() {
    let _len = 2;
    let list_size = 2;

    // Create a 3-level nested FSL: FSL[FSL[FSL[i32]]].
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array();

    // Level 1: FSL of i32.
    let level1 =
        FixedSizeListArray::new(elements.into_array(), list_size, Validity::NonNullable, 4);

    // Level 2: FSL of FSL.
    let level2 = FixedSizeListArray::new(level1.into_array(), list_size, Validity::NonNullable, 2);

    // Level 3: FSL of FSL of FSL.
    let level3 = FixedSizeListArray::new(level2.into_array(), list_size, Validity::NonNullable, 1);

    assert_eq!(level3.len(), 1);
    assert_eq!(level3.list_size(), list_size);

    // Verify the dtype is correct.
    assert!(matches!(
        level3.dtype(),
        DType::FixedSizeList(l2_dtype, 2, Nullability::NonNullable)
            if matches!(
                l2_dtype.as_ref(),
                DType::FixedSizeList(l1_dtype, 2, Nullability::NonNullable)
                    if matches!(
                        l1_dtype.as_ref(),
                        DType::FixedSizeList(elem_dtype, 2, Nullability::NonNullable)
                            if matches!(elem_dtype.as_ref(),
                                DType::Primitive(PType::I32, Nullability::NonNullable))
                    )
            )
    ));

    // Check the actual deeply nested values.
    // Structure: [[[1,2],[3,4]],[[5,6],[7,8]]].
    let top_level = level3.fixed_size_list_elements_at(0).unwrap();
    let level2_0 = top_level
        .to_fixed_size_list()
        .fixed_size_list_elements_at(0)
        .unwrap();
    let level2_1 = top_level
        .to_fixed_size_list()
        .fixed_size_list_elements_at(1)
        .unwrap();

    // First level-2 list: [[1,2],[3,4]].
    let level1_0_0 = level2_0
        .to_fixed_size_list()
        .fixed_size_list_elements_at(0)
        .unwrap();
    assert_eq!(level1_0_0.scalar_at(0).unwrap(), 1i32.into());
    assert_eq!(level1_0_0.scalar_at(1).unwrap(), 2i32.into());

    let level1_0_1 = level2_0
        .to_fixed_size_list()
        .fixed_size_list_elements_at(1)
        .unwrap();
    assert_eq!(level1_0_1.scalar_at(0).unwrap(), 3i32.into());
    assert_eq!(level1_0_1.scalar_at(1).unwrap(), 4i32.into());

    // Second level-2 list: [[5,6],[7,8]].
    let level1_1_0 = level2_1
        .to_fixed_size_list()
        .fixed_size_list_elements_at(0)
        .unwrap();
    assert_eq!(level1_1_0.scalar_at(0).unwrap(), 5i32.into());
    assert_eq!(level1_1_0.scalar_at(1).unwrap(), 6i32.into());

    let level1_1_1 = level2_1
        .to_fixed_size_list()
        .fixed_size_list_elements_at(1)
        .unwrap();
    assert_eq!(level1_1_1.scalar_at(0).unwrap(), 7i32.into());
    assert_eq!(level1_1_1.scalar_at(1).unwrap(), 8i32.into());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FSL of List tests
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_fsl_of_list() {
    let fsl_len = 2;
    let fsl_size = 3;

    // Create a ListBuilder for i32 values.
    let mut list_builder = ListBuilder::<u64>::with_capacity(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        Nullability::NonNullable,
        12,
        6,
    );

    // Add 6 lists (2 FSL * 3 lists each).
    // First FSL: [[1,2], [3], [4,5,6]].
    list_builder
        .append_scalar(&Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![1i32.into(), 2i32.into()],
            Nullability::NonNullable,
        ))
        .unwrap();
    list_builder
        .append_scalar(&Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![3i32.into()],
            Nullability::NonNullable,
        ))
        .unwrap();
    list_builder
        .append_scalar(&Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![4i32.into(), 5i32.into(), 6i32.into()],
            Nullability::NonNullable,
        ))
        .unwrap();

    // Second FSL: [[7], [8,9], []].
    list_builder
        .append_scalar(&Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![7i32.into()],
            Nullability::NonNullable,
        ))
        .unwrap();
    list_builder
        .append_scalar(&Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![8i32.into(), 9i32.into()],
            Nullability::NonNullable,
        ))
        .unwrap();
    list_builder
        .append_scalar(&Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![],
            Nullability::NonNullable,
        ))
        .unwrap();

    let list_array = list_builder.finish();

    // Create FSL of List.
    let fsl = FixedSizeListArray::new(list_array, fsl_size, Validity::NonNullable, fsl_len);

    assert_eq!(fsl.len(), fsl_len);
    assert_eq!(fsl.list_size(), fsl_size);

    // Check dtype.
    assert!(matches!(
        fsl.dtype(),
        DType::FixedSizeList(list_dtype, 3, Nullability::NonNullable)
            if matches!(
                list_dtype.as_ref(),
                DType::List(elem_dtype, Nullability::NonNullable)
                    if matches!(elem_dtype.as_ref(),
                        DType::Primitive(PType::I32, Nullability::NonNullable))
            )
    ));
}

#[test]
fn test_fsl_of_nullable_list() {
    let fsl_len = 2;
    let fsl_size = 2;

    // Create a ListBuilder with nullable lists.
    let mut list_builder = ListBuilder::<u64>::with_capacity(
        Arc::new(DType::Primitive(PType::U16, Nullability::NonNullable)),
        Nullability::Nullable,
        8,
        4,
    );

    // Add 4 lists (2 FSL * 2 lists each).
    // First FSL: [[1,2], null].
    list_builder
        .append_scalar(&Scalar::list(
            Arc::new(DType::Primitive(PType::U16, Nullability::NonNullable)),
            vec![1u16.into(), 2u16.into()],
            Nullability::Nullable,
        ))
        .unwrap();
    list_builder.append_null();

    // Second FSL: [[3], [4,5]].
    list_builder
        .append_scalar(&Scalar::list(
            Arc::new(DType::Primitive(PType::U16, Nullability::NonNullable)),
            vec![3u16.into()],
            Nullability::Nullable,
        ))
        .unwrap();
    list_builder
        .append_scalar(&Scalar::list(
            Arc::new(DType::Primitive(PType::U16, Nullability::NonNullable)),
            vec![4u16.into(), 5u16.into()],
            Nullability::Nullable,
        ))
        .unwrap();

    let list_array = list_builder.finish();

    // Create FSL of nullable List.
    let fsl = FixedSizeListArray::new(list_array, fsl_size, Validity::NonNullable, fsl_len);

    assert_eq!(fsl.len(), fsl_len);
    assert_eq!(fsl.list_size(), fsl_size);

    // Check that the FSL itself is non-nullable but contains nullable lists.
    assert!(matches!(
        fsl.dtype(),
        DType::FixedSizeList(list_dtype, 2, Nullability::NonNullable)
            if matches!(
                list_dtype.as_ref(),
                DType::List(_, Nullability::Nullable)
            )
    ));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FSL with complex types tests (Struct, etc.)
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_fsl_of_struct() {
    let fsl_len = 2;
    let fsl_size = 3u32;

    // Create a struct with two fields: a: i32, b: f64.
    let struct_fields = StructFields::new(
        FieldNames::from(["a", "b"].as_slice()),
        vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Primitive(PType::F64, Nullability::NonNullable),
        ],
    );

    // Create struct arrays for the FSL.
    let a_values = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let b_values = buffer![1.1f64, 2.2, 3.3, 4.4, 5.5, 6.6].into_array();

    let struct_array = StructArray::try_new(
        struct_fields.names().clone(),
        vec![a_values, b_values],
        fsl_len * fsl_size as usize,
        Validity::NonNullable,
    )
    .unwrap();

    // Create FSL of structs.
    let fsl = FixedSizeListArray::new(
        struct_array.into_array(),
        fsl_size,
        Validity::NonNullable,
        fsl_len,
    );

    assert_eq!(fsl.len(), fsl_len);
    assert_eq!(fsl.list_size(), fsl_size);

    // Check dtype.
    assert!(matches!(
        fsl.dtype(),
        DType::FixedSizeList(struct_dt, 3, Nullability::NonNullable)
            if matches!(struct_dt.as_ref(), DType::Struct(_, Nullability::NonNullable))
    ));
}

#[test]
fn test_fsl_of_nullable_struct() {
    let fsl_len = 3;
    let fsl_size = 2u32;

    // Create a nullable struct.
    let struct_fields = StructFields::new(
        FieldNames::from(["x", "y"].as_slice()),
        vec![
            DType::Primitive(PType::U32, Nullability::NonNullable),
            DType::Primitive(PType::U16, Nullability::NonNullable),
        ],
    );

    // Create struct arrays with some null structs.
    let x_values = buffer![10u32, 20, 30, 40, 50, 60].into_array();
    let y_values = buffer![1u16, 2, 3, 4, 5, 6].into_array();

    let struct_validity = Validity::from_iter([true, false, true, true, false, true]);
    let struct_array = StructArray::try_new(
        struct_fields.names().clone(),
        vec![x_values.into_array(), y_values.into_array()],
        fsl_len * fsl_size as usize,
        struct_validity,
    )
    .unwrap();

    // Create FSL of nullable structs.
    let fsl = FixedSizeListArray::new(
        struct_array.into_array(),
        fsl_size,
        Validity::NonNullable,
        fsl_len,
    );

    assert_eq!(fsl.len(), fsl_len);
    assert_eq!(fsl.list_size(), fsl_size);

    // The FSL itself is non-nullable, but contains nullable structs.
    assert!(matches!(
        fsl.dtype(),
        DType::FixedSizeList(struct_dt, 2, Nullability::NonNullable)
            if matches!(struct_dt.as_ref(), DType::Struct(_, Nullability::Nullable))
    ));
}

#[test]
fn test_fsl_with_empty_struct() {
    let fsl_len = 2;
    let fsl_size = 3u32;

    // Create an empty struct (no fields).
    let struct_fields = StructFields::empty();

    let struct_array = StructArray::try_new(
        struct_fields.names().clone(),
        vec![],
        fsl_len * fsl_size as usize,
        Validity::NonNullable,
    )
    .unwrap();

    // Create FSL of empty structs.
    let fsl = FixedSizeListArray::new(
        struct_array.into_array(),
        fsl_size,
        Validity::NonNullable,
        fsl_len,
    );

    assert_eq!(fsl.len(), fsl_len);
    assert_eq!(fsl.list_size(), fsl_size);
}

#[test]
fn test_struct_of_fsl() {
    // Create a struct containing FSL fields.
    let fsl1_elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl1 = FixedSizeListArray::new(fsl1_elements, 2, Validity::NonNullable, 3);

    let fsl2_elements = buffer![1.1f64, 2.2, 3.3, 4.4, 5.5, 6.6].into_array();
    let fsl2 = FixedSizeListArray::new(fsl2_elements, 2, Validity::NonNullable, 3);

    let struct_fields = StructFields::new(
        FieldNames::from(["int_lists", "float_lists"].as_slice()),
        vec![
            DType::FixedSizeList(
                Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                2,
                Nullability::NonNullable,
            ),
            DType::FixedSizeList(
                Arc::new(DType::Primitive(PType::F64, Nullability::NonNullable)),
                2,
                Nullability::NonNullable,
            ),
        ],
    );

    let struct_array = StructArray::try_new(
        struct_fields.names().clone(),
        vec![fsl1.into_array(), fsl2.into_array()],
        3,
        Validity::NonNullable,
    )
    .unwrap();

    assert_eq!(struct_array.len(), 3);

    // Check that the struct contains FSL fields.
    assert!(matches!(
        struct_array.dtype(),
        DType::Struct(st_dt, Nullability::NonNullable)
            if st_dt.field("int_lists").as_ref() == Some(&DType::FixedSizeList(
                Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                2,
                Nullability::NonNullable,
            ))
    ));
}
