// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;

use crate::IntoArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::validity::Validity;

////////////////////////////////////////////////////////////////////////////////////////////////////
// ListView of ListView with overlapping data
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_listview_of_listview_with_overlapping() {
    // Create elements that will be shared between inner lists.
    // Elements: [1, 2, 3, 4, 5, 6, 7, 8]
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array();

    // Create inner ListView where lists overlap:
    // Logical inner lists: [[1,2,3], [3,4,5], [2,3,4], [6,7,8], [1,2], [7,8]]
    // inner[0]: elements[0..3] = [1, 2, 3]
    // inner[1]: elements[2..5] = [3, 4, 5] (overlaps with inner[0])
    // inner[2]: elements[1..4] = [2, 3, 4] (overlaps with both)
    // inner[3]: elements[5..8] = [6, 7, 8]
    // inner[4]: elements[0..2] = [1, 2]
    // inner[5]: elements[6..8] = [7, 8] (overlaps with inner[3])
    let inner_offsets = buffer![0u32, 2, 1, 5, 0, 6].into_array();
    let inner_sizes = buffer![3u32, 3, 3, 3, 2, 2].into_array();

    let inner_listview =
        ListViewArray::new(elements, inner_offsets, inner_sizes, Validity::NonNullable);

    // Create outer ListView that groups the inner lists:
    // Logical outer lists: [inner[0..3], inner[3..6]]
    // outer[0]: contains inner[0..3] (3 inner lists)
    // outer[1]: contains inner[3..6] (3 inner lists)
    let outer_offsets = buffer![0u32, 3].into_array();
    let outer_sizes = buffer![3u32, 3].into_array();

    let outer_listview = unsafe {
        ListViewArray::new_unchecked(
            inner_listview.into_array(),
            outer_offsets,
            outer_sizes,
            Validity::NonNullable,
        )
        .with_zero_copy_to_list(true)
    };

    assert_eq!(outer_listview.len(), 2);

    // Verify the outer structure.
    let first_outer = outer_listview.list_elements_at(0).unwrap();
    let first_outer_lv = first_outer.as_::<ListView>();
    assert_eq!(first_outer_lv.len(), 3);

    // Verify overlapping data is preserved correctly.
    // inner[0] and inner[1] both contain element 3.
    let inner0 = first_outer_lv.list_elements_at(0).unwrap();
    let inner1 = first_outer_lv.list_elements_at(1).unwrap();

    // inner[0] should be [1, 2, 3].
    assert_eq!(
        inner0
            .scalar_at(0)
            .unwrap()
            .as_primitive()
            .as_::<i32>()
            .unwrap(),
        1
    );
    assert_eq!(
        inner0
            .scalar_at(2)
            .unwrap()
            .as_primitive()
            .as_::<i32>()
            .unwrap(),
        3
    );

    // inner[1] should be [3, 4, 5] - shares element 3 with inner[0].
    assert_eq!(
        inner1
            .scalar_at(0)
            .unwrap()
            .as_primitive()
            .as_::<i32>()
            .unwrap(),
        3
    );
    assert_eq!(
        inner1
            .scalar_at(1)
            .unwrap()
            .as_primitive()
            .as_::<i32>()
            .unwrap(),
        4
    );

    // Test slicing the outer ListView.
    let sliced = outer_listview.slice(1..2).unwrap();
    assert_eq!(sliced.len(), 1);
    let sliced_lv = sliced.as_::<ListView>();
    let inner_after_slice = sliced_lv.list_elements_at(0).unwrap();
    assert_eq!(inner_after_slice.len(), 3);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Deeply nested ListView with out-of-order offsets
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_deeply_nested_out_of_order() {
    // Create 3-level nested ListView with out-of-order offsets at each level.
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].into_array();

    // Level 1: ListView with out-of-order offsets.
    // 8 inner lists, accessed in scrambled order.
    // Logical lists: [[9,10], [1,2], [5,6], [13,14], [3,4], [11,12], [7,8], [15,16]]
    let l1_offsets = buffer![8u32, 0, 4, 12, 2, 10, 6, 14].into_array();
    let l1_sizes = buffer![2u32, 2, 2, 2, 2, 2, 2, 2].into_array();

    let level1 = ListViewArray::new(elements, l1_offsets, l1_sizes, Validity::NonNullable);

    // Level 2: Group level1 lists, also out-of-order.
    // 4 lists of 2 inner lists each.
    // Logical lists: [level1[6..8], level1[2..4], level1[0..2], level1[4..6]]
    let l2_offsets = buffer![6u32, 2, 0, 4].into_array();
    let l2_sizes = buffer![2u32, 2, 2, 2].into_array();

    let level2 = ListViewArray::new(
        level1.into_array(),
        l2_offsets,
        l2_sizes,
        Validity::NonNullable,
    );

    // Level 3: Top level groups.
    // 2 lists of 2 level2 lists each.
    // Logical lists: [level2[2..4], level2[0..2]]
    let l3_offsets = buffer![2u32, 0].into_array();
    let l3_sizes = buffer![2u32, 2].into_array();

    let level3 = ListViewArray::new(
        level2.into_array(),
        l3_offsets,
        l3_sizes,
        Validity::NonNullable,
    );

    assert_eq!(level3.len(), 2);

    // Navigate through the scrambled structure.
    let top0 = level3.list_elements_at(0).unwrap();
    let top0_lv = top0.as_::<ListView>();
    assert_eq!(top0_lv.len(), 2);

    // Due to out-of-order at level3, top0 actually contains level2[2] and level2[3].
    let mid0 = top0_lv.list_elements_at(0).unwrap();
    let mid0_lv = mid0.as_::<ListView>();
    assert_eq!(mid0_lv.len(), 2);

    // Verify data integrity through the scrambled offsets.
    // This should access the original elements correctly despite the scrambling.
    let inner = mid0_lv.list_elements_at(0).unwrap();
    assert_eq!(inner.len(), 2);

    // Test that operations work correctly with out-of-order offsets.
    let sliced = level3.slice(0..1).unwrap();
    assert_eq!(sliced.len(), 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Mixed offset and size types
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_mixed_offset_size_types() {
    // Test with u64 offsets and u8 sizes at outer level,
    // u32 offsets and u16 sizes at inner level.
    // This tests type conversion edge cases unique to ListView.

    let elements = PrimitiveArray::from_iter(0i32..256).into_array();

    // Inner ListView with u32 offsets and u16 sizes.
    // Logical lists: [[0..5], [10..18], [20..30], [30..37], [40..55], [50..70], [100..125],
    //                 [150..180], [200..250]]
    let inner_offsets = buffer![0u32, 10, 20, 30, 40, 50, 100, 150, 200].into_array();
    let inner_sizes = buffer![5u16, 8, 10, 7, 15, 20, 25, 30, 50].into_array();

    let inner_listview =
        ListViewArray::new(elements, inner_offsets, inner_sizes, Validity::NonNullable);

    // Outer ListView with u64 offsets and u8 sizes.
    // Using small sizes that fit in u8 to test the type difference.
    // Logical lists: [inner[0..3], inner[3..6], inner[6..9]]
    let outer_offsets = buffer![0u64, 3, 6].into_array();
    let outer_sizes = buffer![3u8, 3, 3].into_array();

    let outer_listview = ListViewArray::new(
        inner_listview.into_array(),
        outer_offsets,
        outer_sizes,
        Validity::NonNullable,
    );

    assert_eq!(outer_listview.len(), 3);

    // Verify that different integer types work correctly.
    let first_outer = outer_listview.list_elements_at(0).unwrap();
    assert_eq!(first_outer.len(), 3);

    // Test slicing with mixed types.
    let sliced = outer_listview.slice(1..3).unwrap();
    assert_eq!(sliced.len(), 2);
    let sliced_lv = sliced.as_::<ListView>();

    // Verify the sliced data maintains correct offsets despite type differences.
    let sliced_first = sliced_lv.list_elements_at(0).unwrap();
    assert_eq!(sliced_first.len(), 3);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Zero-sized and overlapping lists
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_listview_zero_and_overlapping() {
    // Mix of empty lists, overlapping lists, and normal lists.
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();

    // Create inner lists with various patterns:
    // Logical lists: [[], [1,2,3], [], [2,3,4], [5], [], [1,2,3,4,5], []]
    // inner[0]: empty list
    // inner[1]: [1, 2, 3] (normal)
    // inner[2]: empty list
    // inner[3]: [2, 3, 4] (overlaps with inner[1])
    // inner[4]: [5] (single element)
    // inner[5]: empty list
    // inner[6]: [1, 2, 3, 4, 5] (entire array)
    // inner[7]: empty list
    let inner_offsets = buffer![0u32, 0, 3, 1, 4, 0, 0, 2].into_array();
    let inner_sizes = buffer![0u32, 3, 0, 3, 1, 0, 5, 0].into_array();

    let inner_listview =
        ListViewArray::new(elements, inner_offsets, inner_sizes, Validity::NonNullable);

    // Create outer lists that group these:
    // Logical lists: [inner[0..3], inner[3..6], inner[6..8]]
    // outer[0]: [empty, [1,2,3], empty] - mix of empty and non-empty
    // outer[1]: [[2,3,4], [5], empty] - overlapping and empty
    // outer[2]: [[1,2,3,4,5], empty] - full and empty
    let outer_offsets = buffer![0u32, 3, 6].into_array();
    let outer_sizes = buffer![3u32, 3, 2].into_array();

    let outer_listview = ListViewArray::new(
        inner_listview.into_array(),
        outer_offsets,
        outer_sizes,
        Validity::NonNullable,
    );

    assert_eq!(outer_listview.len(), 3);

    // Test first outer list with mixed empty/non-empty.
    let first_outer = outer_listview.list_elements_at(0).unwrap();
    let first_outer_lv = first_outer.as_::<ListView>();

    let inner0 = first_outer_lv.list_elements_at(0).unwrap();
    assert_eq!(inner0.len(), 0); // Empty

    let inner1 = first_outer_lv.list_elements_at(1).unwrap();
    assert_eq!(inner1.len(), 3); // [1, 2, 3]
    assert_eq!(
        inner1
            .scalar_at(0)
            .unwrap()
            .as_primitive()
            .as_::<i32>()
            .unwrap(),
        1
    );

    let inner2 = first_outer_lv.list_elements_at(2).unwrap();
    assert_eq!(inner2.len(), 0); // Empty

    // Test second outer list with overlapping data.
    let second_outer = outer_listview.list_elements_at(1).unwrap();
    let second_outer_lv = second_outer.as_::<ListView>();

    let inner3 = second_outer_lv.list_elements_at(0).unwrap();
    assert_eq!(inner3.len(), 3); // [2, 3, 4]
    assert_eq!(
        inner3
            .scalar_at(0)
            .unwrap()
            .as_primitive()
            .as_::<i32>()
            .unwrap(),
        2
    );

    // Verify slicing works with empty lists.
    let sliced = outer_listview.slice(0..2).unwrap();
    assert_eq!(sliced.len(), 2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// ListView of Struct with nullable fields
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_listview_of_struct_with_nulls() {
    // Create structs with fields that could be null.
    let struct_fields = StructFields::new(
        FieldNames::from(["id", "value"].as_slice()),
        vec![
            DType::Primitive(PType::U32, Nullability::NonNullable),
            DType::Primitive(PType::F64, Nullability::Nullable),
        ],
    );

    // Create struct data with some null values.
    let id_values = buffer![1u32, 2, 3, 4, 5, 6].into_array();
    let value_values = PrimitiveArray::from_option_iter(vec![
        Some(1.1f64),
        None,
        Some(3.3),
        Some(4.4),
        None,
        Some(6.6),
    ])
    .into_array();

    // Some structs are null entirely.
    let struct_validity = Validity::from_iter([true, true, false, true, true, false]);

    let struct_array = StructArray::try_new(
        struct_fields.names().clone(),
        vec![id_values, value_values],
        6,
        struct_validity,
    )
    .unwrap();

    // Create ListView of structs with variable sizes and overlapping.
    // Logical lists: [structs[0..2], structs[1..4], structs[3..6]]
    // list[0]: structs[0..2] - one has null field
    // list[1]: structs[1..4] - one entire struct is null, overlaps with list[0]
    // list[2]: structs[3..6] - one entire struct is null
    let offsets = buffer![0u32, 1, 3].into_array();
    let sizes = buffer![2u32, 3, 3].into_array();

    let listview = ListViewArray::new(
        struct_array.into_array(),
        offsets,
        sizes,
        Validity::NonNullable,
    );

    assert_eq!(listview.len(), 3);

    // Verify first list.
    let list0 = listview.list_elements_at(0).unwrap();
    assert_eq!(list0.len(), 2);

    // Verify overlapping list with null struct.
    let list1 = listview.list_elements_at(1).unwrap();
    assert_eq!(list1.len(), 3);

    // The middle element (struct[2]) should be null.
    assert!(list1.scalar_at(1).unwrap().is_null());

    // Test slicing preserves null handling.
    let sliced = listview.slice(1..3).unwrap();
    assert_eq!(sliced.len(), 2);
}
