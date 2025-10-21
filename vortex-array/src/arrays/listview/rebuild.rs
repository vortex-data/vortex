// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{IntegerPType, Nullability, match_each_integer_ptype};
use vortex_error::VortexExpect;

use crate::arrays::ListViewArray;
use crate::builders::{ArrayBuilder, ListViewBuilder, PrimitiveBuilder, builder_with_capacity};
use crate::vtable::ValidityHelper;
use crate::{Array, IntoArray, ToCanonical};

/// Modes for rebuilding a [`ListViewArray`].
pub enum ListViewRebuildMode {
    /// Removes all unused data and flattens out all list data, such that the array is zero-copyable
    /// to a [`ListArray`].
    ///
    /// This mode will deduplicate all overlapping list views, such that the [`ListViewArray`] looks
    /// like a [`ListArray`] but with an additional `sizes` array.
    ///
    /// [`ListArray`]: crate::arrays::ListArray
    MakeZeroCopyToList,

    /// Removes any unused data from the underlying `elements` array.
    ///
    /// This mode will rebuild the `elements` array in the process, but it will keep any overlapping
    /// lists if they exist.
    RemoveGaps,

    // TODO(connor)[ListView]: Implement some version of this.
    /// Finds the shortest packing / overlapping of list elements.
    ///
    /// This problem is known to be NP-hard, so maybe when someone proves that P=NP we can implement
    /// this algorithm (but in all seriousness there are many approximate algorithms that could
    /// work well here).
    OverlapCompression,
}

impl ListViewArray {
    /// Rebuilds the [`ListViewArray`] according to the specified mode.
    pub fn rebuild(&self, mode: ListViewRebuildMode) -> ListViewArray {
        match mode {
            ListViewRebuildMode::RemoveGaps => self.rebuild_remove_gaps(),
            ListViewRebuildMode::MakeZeroCopyToList => self.rebuild_zero_copy_to_list(),
            ListViewRebuildMode::OverlapCompression => unimplemented!("Does P=NP?"),
        }
    }

    /// Rebuilds a [`ListViewArray`], removing all data overlaps and creating a flattened layout.
    ///
    /// This is useful when the `elements` child array of the [`ListViewArray`] might have
    /// overlapping, duplicate, and garbage data, and we want to have fully sequential data like
    /// a [`ListArray`].
    ///
    /// [`ListArray`]: crate::arrays::ListArray
    fn rebuild_zero_copy_to_list(&self) -> ListViewArray {
        let element_dtype = self
            .dtype()
            .as_list_element_opt()
            .vortex_expect("somehow had a canonical list that was not a list");

        let offsets_ptype = self.offsets().dtype().as_ptype();
        let sizes_ptype = self.sizes().dtype().as_ptype();

        match_each_integer_ptype!(offsets_ptype, |O| {
            match_each_integer_ptype!(sizes_ptype, |S| {
                let mut builder = ListViewBuilder::<O, S>::with_capacity(
                    element_dtype.clone(),
                    self.dtype().nullability(),
                    self.elements().len(),
                    self.len(),
                );

                builder.extend_from_array(self.as_ref());
                builder.finish_into_listview()
            })
        })
    }

    /// Rebuilds a [`ListViewArray`] by removing unreferenced elements while preserving overlaps.
    ///
    /// This removes "garbage" elements that are not referenced by any list. Unlike
    /// [`rebuild_flatten()`], this preserves overlap structure where multiple lists may reference
    /// the same elements.
    ///
    /// [`rebuild_flatten()`]: Self::rebuild_flatten
    fn rebuild_remove_gaps(&self) -> ListViewArray {
        if self.is_empty() {
            return self.clone();
        }

        let offset_ptype = self.offsets().dtype().as_ptype();
        let size_ptype = self.sizes().dtype().as_ptype();

        match_each_integer_ptype!(offset_ptype, |O| {
            match_each_integer_ptype!(size_ptype, |S| { self.rebuild_remove_gaps_inner::<O, S>() })
        })
    }

    fn rebuild_remove_gaps_inner<O, S>(&self) -> ListViewArray
    where
        O: IntegerPType,
        S: IntegerPType,
    {
        let offsets_primitive = self.offsets().to_primitive();
        let sizes_primitive = self.sizes().to_primitive();
        let offsets_slice = offsets_primitive.as_slice::<O>();
        let sizes_slice = sizes_primitive.as_slice::<S>();

        // Mark which elements in the elements array are referenced by at least one list.
        let mut referenced_elements = vec![false; self.elements().len()];
        for i in 0..self.len() {
            let offset: usize = offsets_slice[i].as_();
            let size: usize = sizes_slice[i].as_();
            for j in offset..offset + size {
                referenced_elements[j] = true;
            }
        }

        // Fast path: if all elements are referenced, no garbage collection needed.
        if referenced_elements
            .iter()
            .all(|&is_referenced| is_referenced)
        {
            return self.clone();
        }

        // Build a prefix sum that maps each old element index to its new compacted index.
        // For example, if elements [0, 2, 3] are used, the prefix sum maps: 0->0, 1->1, 2->1, 3->2.
        let mut prefix_sum = vec![0; self.elements().len()];
        let mut cumulative_sum = 0;
        for i in 0..referenced_elements.len() {
            prefix_sum[i] = cumulative_sum;
            if referenced_elements[i] {
                cumulative_sum += 1;
            }
        }

        // Copy only the referenced elements into a new compacted elements array.
        let mut elements_builder = builder_with_capacity(self.elements().dtype(), cumulative_sum);
        for (i, &is_referenced) in referenced_elements.iter().enumerate() {
            if is_referenced {
                elements_builder
                    .append_scalar(&self.elements().scalar_at(i))
                    .vortex_expect("append scalar");
            }
        }

        // Remap each old offset to its corresponding new offset in the compacted array.
        let mut new_offsets_builder =
            PrimitiveBuilder::<O>::with_capacity(Nullability::NonNullable, self.len());
        for &old_offset in offsets_slice {
            let new_offset = prefix_sum[old_offset.as_()];
            let offset_value =
                O::from_usize(new_offset).vortex_expect("offset must fit in offset type");
            new_offsets_builder.append_value(offset_value);
        }

        // SAFETY: The new offsets array is guaranteed to be valid because:
        // 1. Each new offset is derived from prefix_sum[old_offset], which maps to a valid index
        //    in the compacted elements array (guaranteed by the prefix sum construction).
        // 2. The sizes array is unchanged, so each (new_offset, size) pair still describes a
        //    valid range within the compacted elements array.
        // 3. The validity is unchanged and still matches the array length.
        unsafe {
            ListViewArray::new_unchecked(
                elements_builder.finish(),
                new_offsets_builder.finish().into_array(),
                self.sizes().clone(),
                self.validity().clone(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_dtype::Nullability;

    use crate::arrays::{ListViewArray, PrimitiveArray};
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;
    use crate::{IntoArray, ToCanonical};

    #[test]
    fn test_rebuild_remove_gaps_with_leading_garbage() {
        // Create a list view with garbage at the beginning: [_, _, A, B, C]
        // List 0: offset=2, size=2 -> [A, B]
        // List 1: offset=3, size=2 -> [B, C]
        let elements = PrimitiveArray::from_iter(vec![99i32, 98, 1, 2, 3]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![2u32, 3]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 2]).into_array();

        let listview =
            ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable).unwrap();

        let compacted = listview.rebuild_remove_gaps();

        // After GC: elements should be [A, B, C] = [1, 2, 3]
        assert_eq!(compacted.elements().len(), 3);

        // Offsets should be remapped: old offset 2 -> new offset 0, old offset 3 -> new offset 1
        assert_eq!(compacted.offset_at(0), 0);
        assert_eq!(compacted.size_at(0), 2);
        assert_eq!(compacted.offset_at(1), 1);
        assert_eq!(compacted.size_at(1), 2);

        // Verify the data is correct by reading the lists
        let list0 = compacted.list_elements_at(0).to_primitive();
        assert_eq!(list0.as_slice::<i32>(), &[1, 2]);

        let list1 = compacted.list_elements_at(1).to_primitive();
        assert_eq!(list1.as_slice::<i32>(), &[2, 3]);
    }

    #[test]
    fn test_rebuild_remove_gaps_with_trailing_garbage() {
        // Create a list view with garbage at the end: [A, B, C, _, _]
        // List 0: offset=0, size=2 -> [A, B]
        // List 1: offset=1, size=2 -> [B, C]
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3, 99, 98]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 1]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 2]).into_array();

        let listview =
            ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable).unwrap();

        let compacted = listview.rebuild_remove_gaps();

        // After GC: elements should be [A, B, C] = [1, 2, 3]
        assert_eq!(compacted.elements().len(), 3);

        // Offsets should remain the same since no leading garbage
        assert_eq!(compacted.offset_at(0), 0);
        assert_eq!(compacted.offset_at(1), 1);
    }

    #[test]
    fn test_rebuild_remove_gaps_no_garbage() {
        // Create a list view with no garbage: [A, B, C]
        // List 0: offset=0, size=2 -> [A, B]
        // List 1: offset=2, size=1 -> [C]
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 2]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 1]).into_array();

        let listview =
            ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable).unwrap();

        let compacted = listview.rebuild_remove_gaps();

        // Should be unchanged (fast path)
        assert_eq!(compacted.elements().len(), 3);
        assert_eq!(compacted.offset_at(0), 0);
        assert_eq!(compacted.offset_at(1), 2);
    }

    #[test]
    fn test_rebuild_remove_gaps_preserves_overlaps() {
        // Create a list view with overlapping lists and garbage: [_, A, B, C, _]
        // List 0: offset=1, size=3 -> [A, B, C]
        // List 1: offset=2, size=2 -> [B, C] (overlaps with List 0)
        let elements = PrimitiveArray::from_iter(vec![99i32, 1, 2, 3, 98]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![1u32, 2]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![3u32, 2]).into_array();

        let listview =
            ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable).unwrap();

        let compacted = listview.rebuild_remove_gaps();

        // After GC: elements should be [A, B, C] = [1, 2, 3]
        assert_eq!(compacted.elements().len(), 3);

        // Offsets should be remapped but still overlapping:
        // old offset 1 -> new offset 0, old offset 2 -> new offset 1
        assert_eq!(compacted.offset_at(0), 0);
        assert_eq!(compacted.size_at(0), 3);
        assert_eq!(compacted.offset_at(1), 1);
        assert_eq!(compacted.size_at(1), 2);

        // Verify both lists still overlap and contain correct data
        let list0 = compacted.list_elements_at(0).to_primitive();
        assert_eq!(list0.as_slice::<i32>(), &[1, 2, 3]);

        let list1 = compacted.list_elements_at(1).to_primitive();
        assert_eq!(list1.as_slice::<i32>(), &[2, 3]);
    }

    #[test]
    fn test_rebuild_remove_gaps_multiple_gaps() {
        // Create a list view with multiple gaps:
        // [_, A, B, _, C, D, _, E, F, _]
        //  0  1  2  3  4  5  6  7  8  9
        // List 0: offset=1, size=2 -> [A, B]
        // List 1: offset=4, size=2 -> [C, D]
        // List 2: offset=7, size=2 -> [E, F]
        // Gaps at: 0, 3, 6, 9
        let elements =
            PrimitiveArray::from_iter(vec![99i32, 1, 2, 98, 3, 4, 97, 5, 6, 96]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![1u32, 4, 7]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 2, 2]).into_array();

        let listview =
            ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable).unwrap();

        let compacted = listview.rebuild_remove_gaps();

        // After GC: elements should be [A, B, C, D, E, F] = [1, 2, 3, 4, 5, 6]
        assert_eq!(compacted.elements().len(), 6);

        // Verify offset remapping:
        // old offset 1 -> new offset 0 (after removing leading gap)
        // old offset 4 -> new offset 2 (after removing gaps at 0, 3)
        // old offset 7 -> new offset 4 (after removing gaps at 0, 3, 6)
        assert_eq!(compacted.offset_at(0), 0);
        assert_eq!(compacted.size_at(0), 2);
        assert_eq!(compacted.offset_at(1), 2);
        assert_eq!(compacted.size_at(1), 2);
        assert_eq!(compacted.offset_at(2), 4);
        assert_eq!(compacted.size_at(2), 2);

        // Verify the data is correct
        let list0 = compacted.list_elements_at(0).to_primitive();
        assert_eq!(list0.as_slice::<i32>(), &[1, 2]);

        let list1 = compacted.list_elements_at(1).to_primitive();
        assert_eq!(list1.as_slice::<i32>(), &[3, 4]);

        let list2 = compacted.list_elements_at(2).to_primitive();
        assert_eq!(list2.as_slice::<i32>(), &[5, 6]);
    }

    #[test]
    fn test_rebuild_flatten_removes_overlaps() {
        // Create a list view with overlapping lists: [A, B, C]
        // List 0: offset=0, size=3 -> [A, B, C]
        // List 1: offset=1, size=2 -> [B, C] (overlaps with List 0)
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 1]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![3u32, 2]).into_array();

        let listview =
            ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable).unwrap();

        let flattened = listview.rebuild_zero_copy_to_list();

        // After flatten: elements should be [A, B, C, B, C] = [1, 2, 3, 2, 3]
        // Lists should be sequential with no overlaps
        assert_eq!(flattened.elements().len(), 5);

        // Offsets should be sequential
        assert_eq!(flattened.offset_at(0), 0);
        assert_eq!(flattened.size_at(0), 3);
        assert_eq!(flattened.offset_at(1), 3);
        assert_eq!(flattened.size_at(1), 2);

        // Verify the data is correct
        let list0 = flattened.list_elements_at(0).to_primitive();
        assert_eq!(list0.as_slice::<i32>(), &[1, 2, 3]);

        let list1 = flattened.list_elements_at(1).to_primitive();
        assert_eq!(list1.as_slice::<i32>(), &[2, 3]);
    }

    #[test]
    fn test_rebuild_flatten_with_nullable() {
        use crate::arrays::BoolArray;

        // Create a nullable list view with a null list
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 1, 2]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 1, 1]).into_array();
        let validity = Validity::Array(
            BoolArray::from_bit_buffer(
                BitBuffer::from(vec![true, false, true]),
                Validity::NonNullable,
            )
            .into_array(),
        );

        let listview = ListViewArray::try_new(elements, offsets, sizes, validity).unwrap();

        let flattened = listview.rebuild_zero_copy_to_list();

        // Verify nullability is preserved
        assert_eq!(flattened.dtype().nullability(), Nullability::Nullable);
        assert!(flattened.validity().is_valid(0));
        assert!(!flattened.validity().is_valid(1));
        assert!(flattened.validity().is_valid(2));

        // Verify valid lists contain correct data
        let list0 = flattened.list_elements_at(0).to_primitive();
        assert_eq!(list0.as_slice::<i32>(), &[1, 2]);

        let list2 = flattened.list_elements_at(2).to_primitive();
        assert_eq!(list2.as_slice::<i32>(), &[3]);
    }
}
