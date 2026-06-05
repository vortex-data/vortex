// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::list::ListArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewRebuildMode;
use crate::arrays::struct_::StructArrayExt;
use crate::builders::PrimitiveBuilder;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::match_each_integer_ptype;

/// Creates a `ListViewArray` from a `ListArray` by computing `sizes` from `offsets`.
///
/// The output `ListViewArray` will be zero-copyable back to a `ListArray`, and additionally it
/// will not have any leading or trailing garbage data.
pub fn list_view_from_list(list: ListArray, ctx: &mut ExecutionCtx) -> VortexResult<ListViewArray> {
    // If the list is empty, create an empty `ListViewArray` with the same offset `DType` as the
    // input.
    if list.is_empty() {
        return Ok(Canonical::empty(list.dtype()).into_listview());
    }

    // We reset the offsets here because mostly for convenience, and also because callers of this
    // function might not expect the output `ListViewArray` to have a bunch of leading and trailing
    // garbage data when they turn it back into a `ListArray`.
    let list = list.reset_offsets(false).vortex_expect("This can't fail");

    let list_offsets = list.offsets().clone();

    // Create `sizes` array by computing differences between consecutive offsets.
    // We use the same `DType` for the sizes as the `offsets` array to ensure compatibility.
    let sizes = match_each_integer_ptype!(list_offsets.dtype().as_ptype(), |O| {
        build_sizes_from_offsets::<O>(&list, ctx)?
    });

    // We need to slice the `offsets` to remove the last element (`ListArray` has `n + 1` offsets).
    debug_assert_eq!(list_offsets.len(), list.len() + 1);
    let adjusted_offsets = list_offsets.slice(0..list.len())?;

    // SAFETY: Since everything came from an existing valid `ListArray`, and the `sizes` were
    // derived from valid and in-order `offsets`, we know these fields are valid.
    // We also just came directly from a `ListArray`, so we know this is zero-copyable.
    Ok(unsafe {
        ListViewArray::new_unchecked(
            list.elements().clone(),
            adjusted_offsets,
            sizes,
            list.validity()?,
        )
        .with_zero_copy_to_list(true)
    })
}

/// Builds a sizes array from a `ListArray` by computing differences between consecutive offsets.
fn build_sizes_from_offsets<O: IntegerPType>(
    list: &ListArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = list.len();
    let mut sizes_builder = PrimitiveBuilder::<O>::with_capacity(Nullability::NonNullable, len);

    // Create `UninitRange` for direct memory access.
    let mut sizes_range = sizes_builder.uninit_range(len);

    let offsets = list.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let offsets_slice = offsets.as_slice::<O>();
    debug_assert_eq!(len + 1, offsets_slice.len());
    debug_assert!(offsets_slice.is_sorted());

    // Compute sizes as the difference between consecutive offsets.
    for i in 0..len {
        let size = offsets_slice[i + 1] - offsets_slice[i];
        sizes_range.set_value(i, size);
    }

    // SAFETY: We have initialized all values in the range.
    unsafe {
        sizes_range.finish();
    }

    Ok(sizes_builder.finish_into_primitive().into_array())
}

// TODO(connor)[ListView]: Note that it is not exactly zero-copy because we have to add a single
// offset at the end, but it is fast enough.
/// Creates a `ListArray` from a `ListViewArray`. The resulting `ListArray` will not have any
/// leading or trailing garbage data.
///
/// If `ListViewArray::is_zero_copy_to_list` is `true`, then this operation is fast
///
/// Otherwise, this function fall back to the (very) expensive path and will rebuild the
/// `ListArray` from scratch.
pub fn list_from_list_view(list_view: ListViewArray) -> VortexResult<ListArray> {
    // Rebuild as zero-copyable to list array and also trim all leading and trailing elements.
    let zctl_array = list_view.rebuild(ListViewRebuildMode::MakeExact)?;
    debug_assert!(zctl_array.is_zero_copy_to_list());

    let list_offsets = match_each_integer_ptype!(zctl_array.offsets().dtype().as_ptype(), |O| {
        // SAFETY: We just made the array zero-copyable to `ListArray`, so the safety contract is
        // upheld.
        unsafe { build_list_offsets_from_list_view::<O>(&zctl_array) }
    });

    // SAFETY: Because the shape of the `ListViewArray` is zero-copyable to a `ListArray`, we
    // can simply reuse all of the data (besides the offsets). We also trim all of the elements to
    // make it easier for the caller to use the `ListArray`.
    Ok(unsafe {
        ListArray::new_unchecked(
            zctl_array.elements().clone(),
            list_offsets,
            zctl_array.validity()?,
        )
    })
}

// TODO(connor)[ListView]: We can optimize this by always keeping extra memory in `ListViewArray`
// offsets for an `n+1`th offset.
/// Builds a `ListArray` offsets array from a `ListViewArray` by constructing `n+1` offsets.
/// The last offset is computed as `last_offset + last_size`.
///
/// # Safety
///
/// The `ListViewArray` must have offsets that are sorted, and every size must be equal to the gap
/// between `offset[i]` and `offset[i + 1]`.
unsafe fn build_list_offsets_from_list_view<O: IntegerPType>(
    list_view: &ListViewArray,
) -> ArrayRef {
    let len = list_view.len();
    let mut offsets_builder =
        PrimitiveBuilder::<O>::with_capacity(Nullability::NonNullable, len + 1);

    // Create uninit range for direct memory access.
    let mut offsets_range = offsets_builder.uninit_range(len + 1);

    #[expect(deprecated)]
    let offsets = list_view.offsets().to_primitive();
    let offsets_slice = offsets.as_slice::<O>();
    debug_assert!(offsets_slice.is_sorted());

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

/// Recursively converts all `ListViewArray`s to `ListArray`s in a nested array structure.
///
/// The conversion happens bottom-up, processing children before parents.
pub fn recursive_list_from_list_view(array: ArrayRef) -> VortexResult<ArrayRef> {
    if !array.dtype().is_nested() {
        return Ok(array);
    }

    #[expect(deprecated)]
    let canonical = array.to_canonical()?;

    Ok(match canonical {
        Canonical::List(listview) => {
            let converted_elements = recursive_list_from_list_view(listview.elements().clone())?;
            debug_assert_eq!(converted_elements.len(), listview.elements().len());

            // Avoid cloning if elements didn't change.
            let listview_with_converted_elements =
                if !ArrayRef::ptr_eq(&converted_elements, listview.elements()) {
                    // SAFETY: We are effectively just replacing the child elements array, which
                    // must have the same length, so all invariants are maintained.
                    unsafe {
                        ListViewArray::new_unchecked(
                            converted_elements,
                            listview.offsets().clone(),
                            listview.sizes().clone(),
                            listview.validity()?,
                        )
                        .with_zero_copy_to_list(listview.is_zero_copy_to_list())
                    }
                } else {
                    listview
                };

            // Make the conversion to `ListArray`.
            let list_array = list_from_list_view(listview_with_converted_elements)?;
            list_array.into_array()
        }
        Canonical::FixedSizeList(fixed_size_list) => {
            let converted_elements =
                recursive_list_from_list_view(fixed_size_list.elements().clone())?;

            // Avoid cloning if elements didn't change.
            if !ArrayRef::ptr_eq(&converted_elements, fixed_size_list.elements()) {
                FixedSizeListArray::try_new(
                    converted_elements,
                    fixed_size_list.list_size(),
                    fixed_size_list.validity()?,
                    fixed_size_list.len(),
                )
                .vortex_expect(
                    "FixedSizeListArray reconstruction should not fail with valid components",
                )
                .into_array()
            } else {
                fixed_size_list.into_array()
            }
        }
        Canonical::Struct(struct_array) => {
            let fields = struct_array.unmasked_fields();
            let mut converted_fields = Vec::with_capacity(fields.len());
            let mut any_changed = false;

            for field in fields.iter() {
                let converted_field = recursive_list_from_list_view(field.clone())?;
                // Avoid cloning if elements didn't change.
                any_changed |= !ArrayRef::ptr_eq(&converted_field, field);
                converted_fields.push(converted_field);
            }

            if any_changed {
                StructArray::try_new(
                    struct_array.names().clone(),
                    converted_fields,
                    struct_array.len(),
                    struct_array.validity()?,
                )
                .vortex_expect("StructArray reconstruction should not fail with valid components")
                .into_array()
            } else {
                struct_array.into_array()
            }
        }
        Canonical::Extension(ext_array) => {
            let converted_storage =
                recursive_list_from_list_view(ext_array.storage_array().clone())?;

            // Avoid cloning if elements didn't change.
            if !ArrayRef::ptr_eq(&converted_storage, ext_array.storage_array()) {
                ExtensionArray::new(ext_array.ext_dtype().clone(), converted_storage).into_array()
            } else {
                ext_array.into_array()
            }
        }
        _ => unreachable!(),
    })
}

#[cfg(test)]
mod tests {

    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use super::super::tests::common::create_basic_listview;
    use super::super::tests::common::create_empty_lists_listview;
    use super::super::tests::common::create_nullable_listview;
    use super::super::tests::common::create_overlapping_listview;
    use super::recursive_list_from_list_view;
    use crate::ArrayEq;
    use crate::ArrayRef;
    use crate::EqMode;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::list::ListArrayExt;
    use crate::arrays::listview::ListViewArrayExt;
    use crate::arrays::listview::list_from_list_view;
    use crate::arrays::listview::list_view_from_list;
    use crate::assert_arrays_eq;
    use crate::dtype::FieldNames;
    use crate::validity::Validity;

    #[test]
    fn test_list_to_listview_basic() -> VortexResult<()> {
        // Create a basic ListArray: [[0,1,2], [3,4], [5,6], [7,8,9]].
        let elements = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
        let offsets = buffer![0u32, 3, 5, 7, 10].into_array();
        let list_array = ListArray::try_new(elements.clone(), offsets, Validity::NonNullable)?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let list_view = list_view_from_list(list_array.clone(), &mut ctx)?;

        // Verify structure.
        assert_eq!(list_view.len(), 4);
        assert_arrays_eq!(elements, list_view.elements().clone());

        // Verify offsets (should be same but without last element).
        let expected_offsets = buffer![0u32, 3, 5, 7].into_array();
        assert_arrays_eq!(expected_offsets, list_view.offsets().clone());

        // Verify sizes.
        let expected_sizes = buffer![3u32, 2, 2, 3].into_array();
        assert_arrays_eq!(expected_sizes, list_view.sizes().clone());

        // Verify data integrity.
        assert_arrays_eq!(list_array, list_view);
        Ok(())
    }

    #[test]
    fn test_listview_to_list_zero_copy() -> VortexResult<()> {
        let list_view = create_basic_listview();
        let list_array = list_from_list_view(list_view.clone())?;

        // Should have same elements.
        assert_arrays_eq!(list_view.elements().clone(), list_array.elements().clone());

        // ListArray offsets should have n+1 elements for n lists (add the final offset).
        // Check that the first n offsets match.
        let list_array_offsets_without_last = list_array.offsets().slice(0..list_view.len())?;
        assert_arrays_eq!(list_view.offsets().clone(), list_array_offsets_without_last);

        // Verify data integrity.
        assert_arrays_eq!(list_view, list_array);
        Ok(())
    }

    #[test]
    fn test_empty_array_conversions() -> VortexResult<()> {
        // Empty ListArray to ListViewArray.
        let empty_elements = PrimitiveArray::from_iter::<[i32; 0]>([]).into_array();
        let empty_offsets = buffer![0u32].into_array();
        let empty_list = ListArray::try_new(empty_elements, empty_offsets, Validity::NonNullable)?;

        // This conversion will create an empty ListViewArray.
        // Note: list_view_from_list handles the empty case specially.
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let empty_list_view = list_view_from_list(empty_list.clone(), &mut ctx)?;
        assert_eq!(empty_list_view.len(), 0);

        // Convert back.
        let converted_back = list_from_list_view(empty_list_view)?;
        assert_eq!(converted_back.len(), 0);
        // For empty arrays, we can't use assert_arrays_eq directly since the offsets might differ.
        // Just check that it's empty.
        assert_eq!(empty_list.len(), converted_back.len());
        Ok(())
    }

    #[test]
    fn test_nullable_conversions() -> VortexResult<()> {
        // Create nullable ListArray: [[10,20], null, [50]].
        let elements = buffer![10i32, 20, 30, 40, 50].into_array();
        let offsets = buffer![0u32, 2, 4, 5].into_array();
        let validity = Validity::Array(BoolArray::from_iter(vec![true, false, true]).into_array());
        let nullable_list = ListArray::try_new(elements, offsets, validity.clone())?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let nullable_list_view = list_view_from_list(nullable_list.clone(), &mut ctx)?;

        // Verify validity is preserved.
        assert!(
            nullable_list_view
                .validity()
                .vortex_expect("listview validity should be derivable")
                .array_eq(&validity, EqMode::Ptr)
        );
        assert_eq!(nullable_list_view.len(), 3);

        // Round-trip conversion.
        let converted_back = list_from_list_view(nullable_list_view)?;
        assert_arrays_eq!(nullable_list, converted_back);
        Ok(())
    }

    #[test]
    fn test_non_zero_copy_listview_to_list() -> VortexResult<()> {
        // Create ListViewArray with overlapping lists (not zero-copyable).
        let list_view = create_overlapping_listview();
        let list_array = list_from_list_view(list_view.clone())?;

        // The resulting ListArray should have monotonic offsets.
        for i in 0..list_array.len() {
            let start = list_array.offset_at(i)?;
            let end = list_array.offset_at(i + 1)?;
            assert!(end >= start, "Offsets should be monotonic after conversion");
        }

        // The data should still be correct even though it required a rebuild.
        assert_arrays_eq!(list_view, list_array);
        Ok(())
    }

    #[test]
    fn test_empty_sublists() -> VortexResult<()> {
        let empty_lists_view = create_empty_lists_listview();

        // Convert to ListArray.
        let list_array = list_from_list_view(empty_lists_view.clone())?;
        assert_eq!(list_array.len(), 4);

        // All sublists should be empty.
        for i in 0..list_array.len() {
            assert_eq!(list_array.list_elements_at(i)?.len(), 0);
        }

        // Round-trip.
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let converted_back = list_view_from_list(list_array, &mut ctx)?;
        assert_arrays_eq!(empty_lists_view, converted_back);
        Ok(())
    }

    #[test]
    fn test_different_offset_types() -> VortexResult<()> {
        // Test with i32 offsets.
        let elements = buffer![1i32, 2, 3, 4, 5].into_array();
        let i32_offsets = buffer![0i32, 2, 5].into_array();
        let list_i32 =
            ListArray::try_new(elements.clone(), i32_offsets.clone(), Validity::NonNullable)?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let list_view_i32 = list_view_from_list(list_i32.clone(), &mut ctx)?;
        assert_eq!(list_view_i32.offsets().dtype(), i32_offsets.dtype());
        assert_eq!(list_view_i32.sizes().dtype(), i32_offsets.dtype());

        // Test with i64 offsets.
        let i64_offsets = buffer![0i64, 2, 5].into_array();
        let list_i64 = ListArray::try_new(elements, i64_offsets.clone(), Validity::NonNullable)?;

        let list_view_i64 = list_view_from_list(list_i64.clone(), &mut ctx)?;
        assert_eq!(list_view_i64.offsets().dtype(), i64_offsets.dtype());
        assert_eq!(list_view_i64.sizes().dtype(), i64_offsets.dtype());

        // Verify data integrity.
        assert_arrays_eq!(list_i32, list_view_i32);
        assert_arrays_eq!(list_i64, list_view_i64);
        Ok(())
    }

    #[test]
    fn test_round_trip_conversions() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        // Test 1: Basic round-trip.
        let original = create_basic_listview();
        let to_list = list_from_list_view(original.clone())?;
        let back_to_view = list_view_from_list(to_list, &mut ctx)?;
        assert_arrays_eq!(original, back_to_view);

        // Test 2: Nullable round-trip.
        let nullable = create_nullable_listview();
        let nullable_to_list = list_from_list_view(nullable.clone())?;
        let nullable_back = list_view_from_list(nullable_to_list, &mut ctx)?;
        assert_arrays_eq!(nullable, nullable_back);

        // Test 3: Non-zero-copyable round-trip.
        let overlapping = create_overlapping_listview();

        let overlapping_to_list = list_from_list_view(overlapping.clone())?;
        let overlapping_back = list_view_from_list(overlapping_to_list, &mut ctx)?;
        assert_arrays_eq!(overlapping, overlapping_back);
        Ok(())
    }

    #[test]
    fn test_single_element_lists() -> VortexResult<()> {
        // Create lists with single elements: [[100], [200], [300]].
        let elements = buffer![100i32, 200, 300].into_array();
        let offsets = buffer![0u32, 1, 2, 3].into_array();
        let single_elem_list = ListArray::try_new(elements, offsets, Validity::NonNullable)?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let list_view = list_view_from_list(single_elem_list.clone(), &mut ctx)?;
        assert_eq!(list_view.len(), 3);

        // Verify sizes are all 1.
        let expected_sizes = buffer![1u32, 1, 1].into_array();
        assert_arrays_eq!(expected_sizes, list_view.sizes().clone());

        // Round-trip.
        let converted_back = list_from_list_view(list_view)?;
        assert_arrays_eq!(single_elem_list, converted_back);
        Ok(())
    }

    #[test]
    fn test_mixed_empty_and_non_empty_lists() -> VortexResult<()> {
        // Create: [[1,2], [], [3], [], [4,5,6]].
        let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let offsets = buffer![0u32, 2, 2, 3, 3, 6].into_array();
        let mixed_list = ListArray::try_new(elements, offsets, Validity::NonNullable)?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let list_view = list_view_from_list(mixed_list.clone(), &mut ctx)?;
        assert_eq!(list_view.len(), 5);

        // Verify sizes.
        let expected_sizes = buffer![2u32, 0, 1, 0, 3].into_array();
        assert_arrays_eq!(expected_sizes, list_view.sizes().clone());

        // Round-trip.
        let converted_back = list_from_list_view(list_view)?;
        assert_arrays_eq!(mixed_list, converted_back);
        Ok(())
    }

    #[test]
    fn test_recursive_simple_listview() -> VortexResult<()> {
        let list_view = create_basic_listview();
        let result = recursive_list_from_list_view(list_view.clone().into_array())?;

        assert_eq!(result.len(), list_view.len());
        assert_arrays_eq!(list_view.into_array(), result);
        Ok(())
    }

    #[test]
    fn test_recursive_nested_listview() -> VortexResult<()> {
        let inner_elements = buffer![1i32, 2, 3].into_array();
        let inner_offsets = buffer![0u32, 2].into_array();
        let inner_sizes = buffer![2u32, 1].into_array();
        let inner_listview = unsafe {
            ListViewArray::new_unchecked(
                inner_elements,
                inner_offsets,
                inner_sizes,
                Validity::NonNullable,
            )
            .with_zero_copy_to_list(true)
        };

        let outer_offsets = buffer![0u32, 1].into_array();
        let outer_sizes = buffer![1u32, 1].into_array();
        let outer_listview = unsafe {
            ListViewArray::new_unchecked(
                inner_listview.into_array(),
                outer_offsets,
                outer_sizes,
                Validity::NonNullable,
            )
            .with_zero_copy_to_list(true)
        };

        let result = recursive_list_from_list_view(outer_listview.clone().into_array())?;

        assert_eq!(result.len(), 2);
        assert_arrays_eq!(outer_listview.into_array(), result);
        Ok(())
    }

    #[test]
    fn test_recursive_struct_with_listview_fields() -> VortexResult<()> {
        let listview_field = create_basic_listview().into_array();
        let primitive_field = buffer![10i32, 20, 30, 40].into_array();

        let struct_array = StructArray::try_new(
            FieldNames::from(["lists", "values"]),
            vec![listview_field, primitive_field],
            4,
            Validity::NonNullable,
        )?;

        let result = recursive_list_from_list_view(struct_array.clone().into_array())?;

        assert_eq!(result.len(), 4);
        assert_arrays_eq!(struct_array.into_array(), result);
        Ok(())
    }

    #[test]
    fn test_recursive_fixed_size_list_with_listview_elements() -> VortexResult<()> {
        let lv1_elements = buffer![1i32, 2].into_array();
        let lv1_offsets = buffer![0u32].into_array();
        let lv1_sizes = buffer![2u32].into_array();
        let lv1 = unsafe {
            ListViewArray::new_unchecked(
                lv1_elements,
                lv1_offsets,
                lv1_sizes,
                Validity::NonNullable,
            )
            .with_zero_copy_to_list(true)
        };

        let lv2_elements = buffer![3i32, 4].into_array();
        let lv2_offsets = buffer![0u32].into_array();
        let lv2_sizes = buffer![2u32].into_array();
        let lv2 = unsafe {
            ListViewArray::new_unchecked(
                lv2_elements,
                lv2_offsets,
                lv2_sizes,
                Validity::NonNullable,
            )
            .with_zero_copy_to_list(true)
        };

        let dtype = lv1.dtype().clone();
        let chunked_listviews =
            crate::arrays::ChunkedArray::try_new(vec![lv1.into_array(), lv2.into_array()], dtype)?;

        let fixed_list =
            FixedSizeListArray::new(chunked_listviews.into_array(), 1, Validity::NonNullable, 2);

        let result = recursive_list_from_list_view(fixed_list.clone().into_array())?;

        assert_eq!(result.len(), 2);
        assert_arrays_eq!(fixed_list.into_array(), result);
        Ok(())
    }

    #[test]
    fn test_recursive_deep_nesting() -> VortexResult<()> {
        let innermost_elements = buffer![1i32, 2, 3].into_array();
        let innermost_offsets = buffer![0u32, 2].into_array();
        let innermost_sizes = buffer![2u32, 1].into_array();
        let innermost_listview = unsafe {
            ListViewArray::new_unchecked(
                innermost_elements,
                innermost_offsets,
                innermost_sizes,
                Validity::NonNullable,
            )
            .with_zero_copy_to_list(true)
        };

        let struct_array = StructArray::try_new(
            FieldNames::from(["inner_lists"]),
            vec![innermost_listview.into_array()],
            2,
            Validity::NonNullable,
        )?;

        let outer_offsets = buffer![0u32, 1].into_array();
        let outer_sizes = buffer![1u32, 1].into_array();
        let outer_listview = unsafe {
            ListViewArray::new_unchecked(
                struct_array.into_array(),
                outer_offsets,
                outer_sizes,
                Validity::NonNullable,
            )
            .with_zero_copy_to_list(true)
        };

        let result = recursive_list_from_list_view(outer_listview.clone().into_array())?;

        assert_eq!(result.len(), 2);
        assert_arrays_eq!(outer_listview.into_array(), result);
        Ok(())
    }

    #[test]
    fn test_recursive_primitive_unchanged() -> VortexResult<()> {
        let prim = buffer![1i32, 2, 3].into_array();
        let prim_clone = prim.clone();
        let result = recursive_list_from_list_view(prim)?;

        assert!(ArrayRef::ptr_eq(&result, &prim_clone));
        Ok(())
    }

    #[test]
    fn test_recursive_mixed_listview_and_list() -> VortexResult<()> {
        let listview = create_basic_listview();
        let list = list_from_list_view(listview.clone())?;

        let struct_array = StructArray::try_new(
            FieldNames::from(["listview_field", "list_field"]),
            vec![listview.into_array(), list.into_array()],
            4,
            Validity::NonNullable,
        )?;

        let result = recursive_list_from_list_view(struct_array.clone().into_array())?;

        assert_eq!(result.len(), 4);
        assert_arrays_eq!(struct_array.into_array(), result);
        Ok(())
    }

    /// Regression test for <https://github.com/vortex-data/vortex/issues/6882>.
    ///
    /// An empty `ListViewArray` constructed via `try_new` has `is_zero_copy_to_list: false`.
    /// `list_from_list_view` should still succeed because empty arrays are trivially
    /// zero-copyable.
    #[test]
    fn test_empty_listview_to_list_without_zctl_flag() -> VortexResult<()> {
        let elements = VarBinViewArray::from_iter_str(Vec::<&str>::new()).into_array();
        let offsets = PrimitiveArray::from_iter(Vec::<i16>::new()).into_array();
        let sizes = PrimitiveArray::from_iter(Vec::<i16>::new()).into_array();
        let list_view = ListViewArray::try_new(elements, offsets, sizes, Validity::AllValid)?;

        // `try_new` sets `is_zero_copy_to_list: false`.
        assert!(!list_view.is_zero_copy_to_list());

        let list_array = list_from_list_view(list_view)?;
        assert_eq!(list_array.len(), 0);
        Ok(())
    }
}
