// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{IntegerPType, Nullability, match_each_integer_ptype};
use vortex_error::VortexExpect;

use crate::arrays::{ListArray, ListViewArray, ListViewShape};
use crate::builders::{ArrayBuilder, ListBuilder, PrimitiveBuilder};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, Canonical, IntoArray, ToCanonical};

/// Creates a [`ListViewArray`] from a [`ListArray`] by computing `sizes` from `offsets`.
pub fn list_view_from_list(list: ListArray) -> ListViewArray {
    // TODO(connor)[ListView]: Create a version of `Canonical::empty` for `ListView` once `ListView`
    // is canonicalized. It might also be worth specializing that for all canonical encodings.

    // If the list is empty, create an empty `ListViewArray` with the same offset `DType` as the
    // input.
    if list.is_empty() {
        let empty_offsets = Canonical::empty(list.offsets().dtype()).into_array();
        let empty_sizes = Canonical::empty(list.offsets().dtype()).into_array();
        let empty_validity = list.validity().clone();
        let shape = ListViewShape::as_zero_copy_to_list();

        // SAFETY: Everything is empty so all the variants are satisfied.
        return unsafe {
            ListViewArray::new_unchecked(
                list.elements().clone(),
                empty_offsets,
                empty_sizes,
                empty_validity,
                shape,
            )
        };
    }

    let len = list.len();

    // Get the `offsets` array directly from the `ListArray` (preserving its type).
    let list_offsets = list.offsets().clone();

    // We need to slice the `offsets` to remove the last element (`ListArray` has n+1 offsets).
    let adjusted_offsets = list_offsets.slice(0..len);

    // Create sizes array by computing differences between consecutive offsets.
    // Use the same dtype as the offsets array to ensure compatibility.
    let sizes = match_each_integer_ptype!(list_offsets.dtype().as_ptype(), |O| {
        build_sizes_from_offsets::<O>(&list)
    });

    // Since the data came from a valid `ListArray`, we know it is zero-copyable to a `ListArray`.
    let shape = ListViewShape::as_zero_copy_to_list();

    // SAFETY: Since everything came from an existing valid `ListArray`, and the `sizes` were
    // derived from valid and in-order `offsets`, we know these fields are valid.
    unsafe {
        ListViewArray::new_unchecked(
            list.elements().clone(),
            adjusted_offsets,
            sizes,
            list.validity().clone(),
            shape,
        )
    }
}

/// Builds a sizes array from a [`ListArray`] by computing differences between consecutive offsets.
fn build_sizes_from_offsets<O: IntegerPType>(list: &ListArray) -> ArrayRef {
    let len = list.len();
    let mut sizes_builder = PrimitiveBuilder::<O>::with_capacity(Nullability::NonNullable, len);

    // Create `UninitRange` for direct memory access.
    let mut sizes_range = sizes_builder.uninit_range(len);
    let offsets = list.offsets().to_primitive();
    let offsets_slice = offsets.as_slice::<O>();

    // Compute sizes as the difference between consecutive offsets.
    for i in 0..len {
        let size = offsets_slice[i + 1] - offsets_slice[i];
        sizes_range.set_value(i, size);
    }

    // SAFETY: We have initialized all values in the range.
    unsafe {
        sizes_range.finish();
    }

    sizes_builder.finish_into_primitive().into_array()
}

/// Creates a [`ListArray`] from a [`ListViewArray`].
///
/// If the [`ListViewShape::is_zero_copy_to_list`] is `true`, then this operation is fast (note that
/// it is not exactly zero-copy because we have to add a single offset at the end, but it is fast
/// enough).
///
/// Otherwise, this function fall back to the expensive path and will rebuild the `ListArray` from
/// scratch.
///
/// [`as_zero_copy_to_list()`]: ListViewShape::as_zero_copy_to_list
pub fn list_from_list_view(list_view: ListViewArray) -> ListArray {
    if list_view.shape().is_zero_copy_to_list() {
        let list_offsets = match_each_integer_ptype!(list_view.offsets().dtype().as_ptype(), |O| {
            // SAFETY: We checked that the shape of the array is correct.
            unsafe { build_list_offsets_from_list_view::<O>(&list_view) }
        });

        // SAFETY: Because the shape of the `ListViewArray` is zero-copyable to a `ListArray`, we
        // can simply reuse all of the data (besides the offsets).
        // See the documentation of `ListViewShape` for more information.
        return unsafe {
            ListArray::new_unchecked(
                list_view.elements().clone(),
                list_offsets,
                list_view.validity().clone(),
            )
        };
    }

    let elements_dtype = list_view
        .dtype()
        .as_list_element_opt()
        .vortex_expect("`DType` of `ListView` was somehow not a `List`");
    let nullability = list_view.dtype().nullability();
    let len = list_view.len();

    match_each_integer_ptype!(list_view.offsets().dtype().as_ptype(), |O| {
        let mut builder = ListBuilder::<O>::with_capacity(elements_dtype.clone(), nullability, len);

        for i in 0..len {
            builder
                .append_scalar(&list_view.scalar_at(i))
                .vortex_expect(
                    "The `ListView` scalars are `ListScalar`, which the `ListBuilder` must accept",
                )
        }

        builder.finish_into_list()
    })
}

/// Builds a [`ListArray`] offsets array from a [`ListViewArray`] by constructing n+1 offsets.
/// The last offset is computed as last_offset + last_size.
///
/// # Safety
///
/// The [`ListViewArray`] must have a shape that allows (near) zero-copying to [`ListArray`].
unsafe fn build_list_offsets_from_list_view<O: IntegerPType>(
    list_view: &ListViewArray,
) -> ArrayRef {
    let len = list_view.len();
    let mut offsets_builder =
        PrimitiveBuilder::<O>::with_capacity(Nullability::NonNullable, len + 1);

    // Create uninit range for direct memory access.
    let mut offsets_range = offsets_builder.uninit_range(len + 1);

    let offsets = list_view.offsets().to_primitive();
    let offsets_slice = offsets.as_slice::<O>();

    // Copy the existing n offsets.
    offsets_range.copy_from_slice(0, offsets_slice);

    // Append the final offset (last offset + last size).
    let final_offset = if len != 0 {
        let last_offset = offsets_slice[len - 1];

        let last_size = list_view.size_at(len - 1);
        let last_size =
            O::from_usize(last_size).vortex_expect("size somehow did not fit into offsets");

        last_offset + last_size
    } else {
        O::zero()
    };

    offsets_range.set_value(len, final_offset);

    // SAFETY: We have initialized all values in the range.
    unsafe {
        offsets_range.finish();
    }

    offsets_builder.finish_into_primitive().into_array()
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::super::tests::common::{
        create_basic_listview, create_empty_lists_listview, create_nullable_listview,
        create_overlapping_listview,
    };
    use crate::arrays::{
        BoolArray, ListArray, PrimitiveArray, list_from_list_view, list_view_from_list,
    };
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;
    use crate::{IntoArray, assert_arrays_eq};

    #[test]
    fn test_list_to_listview_basic() {
        // Create a basic ListArray: [[0,1,2], [3,4], [5,6], [7,8,9]].
        let elements = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
        let offsets = buffer![0u32, 3, 5, 7, 10].into_array();
        let list_array =
            ListArray::try_new(elements.clone(), offsets.clone(), Validity::NonNullable).unwrap();

        let list_view = list_view_from_list(list_array.clone());

        // Verify structure.
        assert_eq!(list_view.len(), 4);
        assert_arrays_eq!(elements, list_view.elements().clone());

        // Verify offsets (should be same but without last element).
        let expected_offsets = buffer![0u32, 3, 5, 7].into_array();
        assert_arrays_eq!(expected_offsets, list_view.offsets().clone());

        // Verify sizes.
        let expected_sizes = buffer![3u32, 2, 2, 3].into_array();
        assert_arrays_eq!(expected_sizes, list_view.sizes().clone());

        // Verify shape is zero-copyable.
        assert!(list_view.shape().is_zero_copy_to_list());

        // Verify data integrity.
        assert_arrays_eq!(list_array, list_view);
    }

    #[test]
    fn test_listview_to_list_zero_copy() {
        let list_view = create_basic_listview();
        assert!(list_view.shape().is_zero_copy_to_list());

        let list_array = list_from_list_view(list_view.clone());

        // Should have same elements.
        assert_arrays_eq!(list_view.elements().clone(), list_array.elements().clone());

        // ListArray offsets should have n+1 elements for n lists (add the final offset).
        // Check that the first n offsets match.
        let list_array_offsets_without_last = list_array.offsets().slice(0..list_view.len());
        assert_arrays_eq!(list_view.offsets().clone(), list_array_offsets_without_last);

        // Verify data integrity.
        assert_arrays_eq!(list_view, list_array);
    }

    #[test]
    fn test_empty_array_conversions() {
        // Empty ListArray to ListViewArray.
        let empty_elements = PrimitiveArray::from_iter::<[i32; 0]>([]).into_array();
        let empty_offsets = buffer![0u32].into_array();
        let empty_list =
            ListArray::try_new(empty_elements.clone(), empty_offsets, Validity::NonNullable)
                .unwrap();

        // This conversion will create an empty ListViewArray.
        // Note: list_view_from_list handles the empty case specially.
        let empty_list_view = list_view_from_list(empty_list.clone());
        assert_eq!(empty_list_view.len(), 0);
        assert!(empty_list_view.shape().is_zero_copy_to_list());

        // Convert back.
        let converted_back = list_from_list_view(empty_list_view);
        assert_eq!(converted_back.len(), 0);
        // For empty arrays, we can't use assert_arrays_eq directly since the offsets might differ.
        // Just check that it's empty.
        assert_eq!(empty_list.len(), converted_back.len());
    }

    #[test]
    fn test_nullable_conversions() {
        // Create nullable ListArray: [[10,20], null, [50]].
        let elements = buffer![10i32, 20, 30, 40, 50].into_array();
        let offsets = buffer![0u32, 2, 4, 5].into_array();
        let validity = Validity::Array(BoolArray::from_iter(vec![true, false, true]).into_array());
        let nullable_list =
            ListArray::try_new(elements.clone(), offsets.clone(), validity.clone()).unwrap();

        let nullable_list_view = list_view_from_list(nullable_list.clone());

        // Verify validity is preserved.
        assert_eq!(nullable_list_view.validity(), &validity);
        assert_eq!(nullable_list_view.len(), 3);

        // Round-trip conversion.
        let converted_back = list_from_list_view(nullable_list_view);
        assert_arrays_eq!(nullable_list, converted_back);
    }

    #[test]
    fn test_non_zero_copy_listview_to_list() {
        // Create ListViewArray with overlapping lists (not zero-copyable).
        let list_view = create_overlapping_listview();
        assert!(!list_view.shape().is_zero_copy_to_list());

        let list_array = list_from_list_view(list_view.clone());

        // The data should still be correct even though it required a rebuild.
        assert_arrays_eq!(list_view, list_array.clone());

        // The resulting ListArray should have monotonic offsets.
        for i in 0..list_array.len() {
            let start = list_array.offset_at(i);
            let end = list_array.offset_at(i + 1);
            assert!(end >= start, "Offsets should be monotonic after conversion");
        }
    }

    #[test]
    fn test_empty_sublists() {
        let empty_lists_view = create_empty_lists_listview();

        // Convert to ListArray.
        let list_array = list_from_list_view(empty_lists_view.clone());
        assert_eq!(list_array.len(), 4);

        // All sublists should be empty.
        for i in 0..list_array.len() {
            assert_eq!(list_array.list_elements_at(i).len(), 0);
        }

        // Round-trip.
        let converted_back = list_view_from_list(list_array);
        assert_arrays_eq!(empty_lists_view, converted_back);
    }

    #[test]
    fn test_different_offset_types() {
        // Test with i32 offsets.
        let elements = buffer![1i32, 2, 3, 4, 5].into_array();
        let i32_offsets = buffer![0i32, 2, 5].into_array();
        let list_i32 =
            ListArray::try_new(elements.clone(), i32_offsets.clone(), Validity::NonNullable)
                .unwrap();

        let list_view_i32 = list_view_from_list(list_i32.clone());
        assert_eq!(list_view_i32.offsets().dtype(), i32_offsets.dtype());
        assert_eq!(list_view_i32.sizes().dtype(), i32_offsets.dtype());

        // Test with i64 offsets.
        let i64_offsets = buffer![0i64, 2, 5].into_array();
        let list_i64 =
            ListArray::try_new(elements.clone(), i64_offsets.clone(), Validity::NonNullable)
                .unwrap();

        let list_view_i64 = list_view_from_list(list_i64.clone());
        assert_eq!(list_view_i64.offsets().dtype(), i64_offsets.dtype());
        assert_eq!(list_view_i64.sizes().dtype(), i64_offsets.dtype());

        // Verify data integrity.
        assert_arrays_eq!(list_i32, list_view_i32);
        assert_arrays_eq!(list_i64, list_view_i64);
    }

    #[test]
    fn test_round_trip_conversions() {
        // Test 1: Basic round-trip.
        let original = create_basic_listview();
        let to_list = list_from_list_view(original.clone());
        let back_to_view = list_view_from_list(to_list);
        assert_arrays_eq!(original, back_to_view);

        // Test 2: Nullable round-trip.
        let nullable = create_nullable_listview();
        let nullable_to_list = list_from_list_view(nullable.clone());
        let nullable_back = list_view_from_list(nullable_to_list);
        assert_arrays_eq!(nullable, nullable_back);

        // Test 3: Non-zero-copyable round-trip.
        // Note: After conversion to ListArray and back, the shape will be zero-copyable.
        let overlapping = create_overlapping_listview();
        assert!(!overlapping.shape().is_zero_copy_to_list());

        let overlapping_to_list = list_from_list_view(overlapping.clone());
        let overlapping_back = list_view_from_list(overlapping_to_list);
        assert!(overlapping_back.shape().is_zero_copy_to_list()); // Now it's zero-copyable!
        assert_arrays_eq!(overlapping, overlapping_back);
    }

    #[test]
    fn test_single_element_lists() {
        // Create lists with single elements: [[100], [200], [300]].
        let elements = buffer![100i32, 200, 300].into_array();
        let offsets = buffer![0u32, 1, 2, 3].into_array();
        let single_elem_list =
            ListArray::try_new(elements.clone(), offsets, Validity::NonNullable).unwrap();

        let list_view = list_view_from_list(single_elem_list.clone());
        assert_eq!(list_view.len(), 3);

        // Verify sizes are all 1.
        let expected_sizes = buffer![1u32, 1, 1].into_array();
        assert_arrays_eq!(expected_sizes, list_view.sizes().clone());

        // Round-trip.
        let converted_back = list_from_list_view(list_view.clone());
        assert_arrays_eq!(single_elem_list, converted_back);
    }

    #[test]
    fn test_mixed_empty_and_non_empty_lists() {
        // Create: [[1,2], [], [3], [], [4,5,6]].
        let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let offsets = buffer![0u32, 2, 2, 3, 3, 6].into_array();
        let mixed_list =
            ListArray::try_new(elements.clone(), offsets.clone(), Validity::NonNullable).unwrap();

        let list_view = list_view_from_list(mixed_list.clone());
        assert_eq!(list_view.len(), 5);

        // Verify sizes.
        let expected_sizes = buffer![2u32, 0, 1, 0, 3].into_array();
        assert_arrays_eq!(expected_sizes, list_view.sizes().clone());

        // Round-trip.
        let converted_back = list_from_list_view(list_view.clone());
        assert_arrays_eq!(mixed_list, converted_back);
    }
}
