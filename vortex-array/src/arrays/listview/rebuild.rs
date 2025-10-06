// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::{FromPrimitive, Zero};
use vortex_dtype::{IntegerPType, Nullability, match_each_integer_ptype};
use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use crate::arrays::{ListViewArray, PrimitiveArray};
use crate::builders::{ArrayBuilder, ListViewBuilder, PrimitiveBuilder, builder_with_capacity};
use crate::vtable::ValidityHelper;
use crate::{Array, IntoArray, ToCanonical, compute};

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

    /// Removes any leading or trailing elements that are unused / not referenced by any views in
    /// the [`ListViewArray`].
    TrimElements,

    /// Removes any unused data from the underlying `elements` array.
    ///
    /// This mode will rebuild the `elements` array in the process, but it will keep any overlapping
    /// lists if they exist.
    ///
    /// Note that this is a more heavyweight version of `TrimElements`, as it will trim any leading
    /// or trailing elements as well as remove inner gaps.
    ///
    /// Also note that we do not remove gaps caused by null views. Use the `RemoveNulls` variant of
    /// `ListViewRebuildMode` if you need that functionality.
    RemoveGaps,

    /// Rebuilds in a similar manner to `RemoveGaps` but also removes any unused data caused by null
    /// views.
    ///
    /// If the [`ListViewArray`] is non-nullable this has the same effect as `RemoveGaps`.
    RemoveNulls,

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
        if self.is_empty() {
            return self.clone();
        }

        match mode {
            ListViewRebuildMode::MakeZeroCopyToList => self.rebuild_zero_copy_to_list(),
            ListViewRebuildMode::TrimElements => self.rebuild_trim_elements(),
            ListViewRebuildMode::RemoveGaps => self.rebuild_remove_gaps::<false>(),
            ListViewRebuildMode::RemoveNulls => self.rebuild_remove_gaps::<true>(),
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
        if self.shape().is_zero_copy_to_list() {
            // Note that since everything in `ListViewArray` is `Arc`ed, this is quite cheap.
            return self.clone();
        }

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

    /// Rebuilds a [`ListViewArray`] by trimming any unused / unreferenced leading and trailing
    /// elements, which is defined as a contiguous run of values in the `elements` array that are
    /// not referecened by any views in the corresponding [`ListViewArray`].
    fn rebuild_trim_elements(&self) -> ListViewArray {
        let start = if self.shape().has_sorted_offsets() {
            // If offsets are sorted, then the minimum offset is the first offset.
            self.offset_at(0)
        } else {
            self.offsets().statistics().compute_min().vortex_expect(
                "[ListViewArray::rebuild]: `offsets` must report min statistic that is a `usize`",
            )
        };

        let end = if self.shape().has_sorted_offsets() && self.shape().has_no_overlaps() {
            // If offsets are sorted and there are no overlaps (views are always "increasing"), we
            // can just grab the last offset and last size.
            let last_offset = self.offset_at(self.len() - 1);
            let last_size = self.size_at(self.len() - 1);
            last_offset + last_size
        } else {
            let min_max = compute::min_max(
                &compute::add(self.offsets(), self.sizes())
                    .vortex_expect("`offsets + sizes` somehow overflowed"),
            )
            .vortex_expect("Something went wrong while computing min and max")
            .vortex_expect("We checked that the array was not empty in the top-level `rebuild`");

            min_max
                .max
                .as_primitive()
                .as_::<usize>()
                .vortex_expect("unable to interpret the max `offset + size` as a `usize`")
        };

        let adjusted_offsets = match_each_integer_ptype!(self.offsets().dtype().as_ptype(), |O| {
            let offset = <O as FromPrimitive>::from_usize(start)
                .vortex_expect("unable to convert the min offset `start` into a `usize`");
            let scalar = Scalar::primitive(offset, Nullability::NonNullable);

            compute::sub_scalar(self.offsets(), scalar)
                .vortex_expect("was somehow unable to adjust offsets down by their minimum")
        });

        let sliced_elements = self.elements().slice(start..end);

        // SAFETY: The only thing we changed was the elements (which we verify through mins and
        // maxes that all adjusted offsets + sizes are within the correct bounds), so the parameters
        // are valid.
        unsafe {
            ListViewArray::new_unchecked(
                sliced_elements,
                adjusted_offsets,
                self.sizes().clone(),
                self.validity().clone(),
                self.shape(),
            )
        }
    }

    /// Rebuilds a [`ListViewArray`] by removing unreferenced elements while preserving overlaps.
    ///
    /// This removes "garbage" elements that are not referenced by any list. Unlike
    /// [`rebuild_flatten()`], this preserves overlap structure where multiple lists may reference
    /// the same elements.
    ///
    /// [`rebuild_flatten()`]: Self::rebuild_flatten
    #[allow(clippy::cognitive_complexity)]
    fn rebuild_remove_gaps<const REMOVE_NULLS: bool>(&self) -> ListViewArray {
        // If there are already no gaps, then we can just trim the `elements` array.
        // However, if we are removing nulls and it is possible we might have null viws, we must
        // fall back to the inner function.
        if self.shape().has_no_gaps() && (!REMOVE_NULLS || !self.dtype().is_nullable()) {
            return self.rebuild_trim_elements();
        }

        let offset_ptype = self.offsets().dtype().as_ptype();
        let size_ptype = self.sizes().dtype().as_ptype();

        match_each_integer_ptype!(offset_ptype, |O| {
            match_each_integer_ptype!(size_ptype, |S| {
                if REMOVE_NULLS && self.dtype().is_nullable() {
                    self.rebuild_remove_gaps_inner::<O, S, true>()
                } else {
                    self.rebuild_remove_gaps_inner::<O, S, false>()
                }
            })
        })
    }

    /// The inner function for [`rebuild_remove_gaps()`](Self::rebuild_remove_gaps).
    fn rebuild_remove_gaps_inner<O, S, const REMOVE_NULLS: bool>(&self) -> ListViewArray
    where
        O: IntegerPType,
        S: IntegerPType,
    {
        let offsets_primitive = self.offsets().to_primitive();
        let sizes_primitive = self.sizes().to_primitive();
        let offsets_slice = offsets_primitive.as_slice::<O>();
        let sizes_slice = sizes_primitive.as_slice::<S>();

        // Some `dyn` magic to generically iterate over the indices we want. We need to store
        // `validity` outside of the `if` statement otherwise it does not live long enough.
        let validity = REMOVE_NULLS.then(|| self.validity.to_mask(self.len()).to_boolean_buffer());
        let indices: &mut dyn Iterator<Item = usize> = if let Some(ref validity) = validity {
            &mut (0..self.len())
                .zip(validity.iter())
                .filter_map(|(i, is_valid)| is_valid.then_some(i))
        } else {
            &mut (0..self.len())
        };

        // Mark which elements in the elements array are referenced by at least one list.
        let mut referenced_elements = vec![false; self.elements().len()];
        for i in indices {
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

        // We set the size of any view that is denoted as null to be 0 (to ensure all views are
        // still valid).
        let new_sizes = if REMOVE_NULLS && self.dtype().is_nullable() {
            let primitive_sizes = self.sizes().to_primitive();
            let view_validity = self.validity.clone();

            match_each_integer_ptype!(primitive_sizes.dtype().as_ptype(), |S| {
                // Create a nullable version of `sizes` and then fill with nulls.
                let nullable_sizes =
                    PrimitiveArray::try_new(primitive_sizes.into_buffer::<S>(), view_validity)
                        .vortex_expect("validity length somehow not matching `sizes` length");

                // Filling with a non-nullable scalar will result in a non-nullable array.
                let zero_scalar = Scalar::primitive(<S as Zero>::zero(), Nullability::NonNullable);

                compute::fill_null(nullable_sizes.as_ref(), &zero_scalar)
                    .vortex_expect("fill_null should not fail filling with zeros")
            })
        } else {
            self.sizes().clone()
        };

        // Create a new `ListViewShape` but set `has_no_gaps` to `true`.
        let new_shape = self.shape().with_no_gaps(true);

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
                new_sizes,
                self.validity().clone(),
                new_shape,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use vortex_dtype::Nullability;

    use super::ListViewRebuildMode;
    use crate::arrays::{BoolArray, ListViewArray, ListViewShape, PrimitiveArray};
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;
    use crate::{IntoArray, ToCanonical};

    #[test]
    fn test_rebuild_remove_gaps_leading_and_trailing() {
        // Combine testing of both leading and trailing garbage removal.
        // Create a list view with garbage at both ends: [_, _, A, B, C, _, _]
        // List 0: offset=2, size=2 -> [A, B]
        // List 1: offset=3, size=2 -> [B, C]
        let elements = PrimitiveArray::from_iter(vec![99i32, 98, 1, 2, 3, 97, 96]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![2u32, 3]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 2]).into_array();

        let listview = ListViewArray::try_new(
            elements,
            offsets,
            sizes,
            Validity::NonNullable,
            ListViewShape::as_zero_copy_to_list().with_no_overlaps(false),
        )
        .unwrap();

        let compacted = listview.rebuild(ListViewRebuildMode::RemoveGaps);

        // After GC: elements should be [A, B, C] = [1, 2, 3].
        assert_eq!(compacted.elements().len(), 3);

        // Offsets should be remapped: old offset 2 -> new offset 0, old offset 3 -> new offset 1.
        assert_eq!(compacted.offset_at(0), 0);
        assert_eq!(compacted.size_at(0), 2);
        assert_eq!(compacted.offset_at(1), 1);
        assert_eq!(compacted.size_at(1), 2);

        // Verify the data is correct by reading the lists.
        let list0 = compacted.list_elements_at(0).to_primitive();
        assert_eq!(list0.as_slice::<i32>(), &[1, 2]);

        let list1 = compacted.list_elements_at(1).to_primitive();
        assert_eq!(list1.as_slice::<i32>(), &[2, 3]);
    }

    #[test]
    fn test_rebuild_remove_gaps_no_garbage() {
        // Create a list view with no garbage: [A, B, C]
        // List 0: offset=0, size=2 -> [A, B]
        // List 1: offset=2, size=1 -> [C]
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 2]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 1]).into_array();

        let listview = ListViewArray::try_new(
            elements,
            offsets,
            sizes,
            Validity::NonNullable,
            ListViewShape::as_zero_copy_to_list(),
        )
        .unwrap();

        let compacted = listview.rebuild(ListViewRebuildMode::RemoveGaps);

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

        let listview = ListViewArray::try_new(
            elements,
            offsets,
            sizes,
            Validity::NonNullable,
            ListViewShape::as_zero_copy_to_list().with_no_overlaps(false),
        )
        .unwrap();

        let compacted = listview.rebuild(ListViewRebuildMode::RemoveGaps);

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

        let listview = ListViewArray::try_new(
            elements,
            offsets,
            sizes,
            Validity::NonNullable,
            ListViewShape::as_zero_copy_to_list().with_no_gaps(false),
        )
        .unwrap();

        let compacted = listview.rebuild(ListViewRebuildMode::RemoveGaps);

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

    #[ignore = "TODO(connor)[ListView]: Reenable when `ListView` becomes canonical"]
    #[test]
    fn test_rebuild_flatten_removes_overlaps() {
        // Create a list view with overlapping lists: [A, B, C]
        // List 0: offset=0, size=3 -> [A, B, C]
        // List 1: offset=1, size=2 -> [B, C] (overlaps with List 0)
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 1]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![3u32, 2]).into_array();

        let listview = ListViewArray::try_new(
            elements,
            offsets,
            sizes,
            Validity::NonNullable,
            ListViewShape::as_zero_copy_to_list().with_no_overlaps(false),
        )
        .unwrap();

        let flattened = listview.rebuild(ListViewRebuildMode::MakeZeroCopyToList);

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

    #[ignore = "TODO(connor)[ListView]: Reenable when `ListView` becomes canonical"]
    #[test]
    fn test_rebuild_flatten_with_nullable() {
        // Create a nullable list view: [A, B, C]
        // List 0: offset=0, size=2 -> [A, B] (valid)
        // List 1: offset=1, size=1 -> null
        // List 2: offset=2, size=1 -> [C] (valid)
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 1, 2]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 1, 1]).into_array();
        let validity = Validity::Array(
            BoolArray::from(BooleanBuffer::from(vec![true, false, true])).into_array(),
        );

        let listview = ListViewArray::try_new(
            elements,
            offsets,
            sizes,
            validity,
            ListViewShape::as_zero_copy_to_list().with_no_overlaps(false),
        )
        .unwrap();

        let flattened = listview.rebuild(ListViewRebuildMode::MakeZeroCopyToList);

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

    #[test]
    fn test_rebuild_remove_gaps_complex() {
        // Test with unsorted offsets, gaps, and overlaps combined.
        // Elements: [_, A, B, C, _, D, E, _]
        //            0  1  2  3  4  5  6  7
        // List 0: offset=5, size=2 -> [D, E]
        // List 1: offset=1, size=3 -> [A, B, C]
        // List 2: offset=2, size=2 -> [B, C] (overlaps with List 1)
        let elements = PrimitiveArray::from_iter(vec![99i32, 1, 2, 3, 98, 4, 5, 97]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![5u32, 1, 2]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 3, 2]).into_array();

        let listview = ListViewArray::try_new(
            elements,
            offsets,
            sizes,
            Validity::NonNullable,
            ListViewShape::as_zero_copy_to_list()
                .with_sorted_offsets(false)
                .with_no_overlaps(false)
                .with_no_gaps(false),
        )
        .unwrap();

        let compacted = listview.rebuild(ListViewRebuildMode::RemoveGaps);

        // After GC: elements should be [A, B, C, D, E] = [1, 2, 3, 4, 5].
        assert_eq!(compacted.elements().len(), 5);

        // Verify offset remapping:
        // old offset 5 -> new offset 3 (D is at index 3 in compacted)
        // old offset 1 -> new offset 0 (A is at index 0 in compacted)
        // old offset 2 -> new offset 1 (B is at index 1 in compacted)
        assert_eq!(compacted.offset_at(0), 3);
        assert_eq!(compacted.size_at(0), 2);
        assert_eq!(compacted.offset_at(1), 0);
        assert_eq!(compacted.size_at(1), 3);
        assert_eq!(compacted.offset_at(2), 1);
        assert_eq!(compacted.size_at(2), 2);

        // Verify the data is correct.
        let list0 = compacted.list_elements_at(0).to_primitive();
        assert_eq!(list0.as_slice::<i32>(), &[4, 5]);

        let list1 = compacted.list_elements_at(1).to_primitive();
        assert_eq!(list1.as_slice::<i32>(), &[1, 2, 3]);

        let list2 = compacted.list_elements_at(2).to_primitive();
        assert_eq!(list2.as_slice::<i32>(), &[2, 3]);
    }

    #[test]
    fn test_rebuild_trim_elements_basic() {
        // Test trimming both leading and trailing unused elements while preserving gaps in the
        // middle.
        // Elements: [_, _, A, B, _, C, D, _, _]
        //            0  1  2  3  4  5  6  7  8
        // List 0: offset=2, size=2 -> [A, B]
        // List 1: offset=5, size=2 -> [C, D]
        // Should trim to: [A, B, _, C, D] with adjusted offsets.
        let elements =
            PrimitiveArray::from_iter(vec![99i32, 98, 1, 2, 97, 3, 4, 96, 95]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![2u32, 5]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 2]).into_array();

        let listview = ListViewArray::try_new(
            elements,
            offsets,
            sizes,
            Validity::NonNullable,
            ListViewShape::as_zero_copy_to_list().with_no_gaps(false),
        )
        .unwrap();

        let trimmed = listview.rebuild(ListViewRebuildMode::TrimElements);

        // After trimming: elements should be [A, B, _, C, D] = [1, 2, 97, 3, 4].
        assert_eq!(trimmed.elements().len(), 5);

        // Offsets should be adjusted: old offset 2 -> new offset 0, old offset 5 -> new offset 3.
        assert_eq!(trimmed.offset_at(0), 0);
        assert_eq!(trimmed.size_at(0), 2);
        assert_eq!(trimmed.offset_at(1), 3);
        assert_eq!(trimmed.size_at(1), 2);

        // Verify the data is correct.
        let list0 = trimmed.list_elements_at(0).to_primitive();
        assert_eq!(list0.as_slice::<i32>(), &[1, 2]);

        let list1 = trimmed.list_elements_at(1).to_primitive();
        assert_eq!(list1.as_slice::<i32>(), &[3, 4]);

        // Note that element at index 2 (97) is preserved as a gap.
        let all_elements = trimmed.elements().to_primitive();
        assert_eq!(all_elements.scalar_at(2), 97i32.into());
    }

    #[test]
    fn test_rebuild_remove_nulls_basic() {
        // Test removing elements that are only referenced by null views.
        // Elements: [A, B, C, D, E]
        // List 0: offset=0, size=2 -> [A, B] (valid)
        // List 1: offset=2, size=2 -> [C, D] (null)
        // List 2: offset=4, size=1 -> [E] (valid)
        // After rebuild: should remove [C, D] since only referenced by null view.
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3, 4, 5]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 2, 4]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 2, 1]).into_array();
        let validity = Validity::Array(
            BoolArray::from(BooleanBuffer::from(vec![true, false, true])).into_array(),
        );

        let listview = ListViewArray::try_new(
            elements,
            offsets,
            sizes,
            validity,
            ListViewShape::as_zero_copy_to_list(),
        )
        .unwrap();

        let rebuilt = listview.rebuild(ListViewRebuildMode::RemoveNulls);

        // After removing nulls: elements should be [A, B, E] = [1, 2, 5].
        assert_eq!(rebuilt.elements().len(), 3);

        // Offsets should be adjusted: offset 0 stays 0, offset 2 would be mapped but unused,
        // offset 4 -> 2.
        assert_eq!(rebuilt.offset_at(0), 0);
        assert_eq!(rebuilt.size_at(0), 2);

        // The null view's size should be 0 (filled by fill_null).
        assert_eq!(rebuilt.size_at(1), 0);

        assert_eq!(rebuilt.offset_at(2), 2);
        assert_eq!(rebuilt.size_at(2), 1);

        // Verify validity is preserved.
        assert!(rebuilt.validity().is_valid(0));
        assert!(!rebuilt.validity().is_valid(1));
        assert!(rebuilt.validity().is_valid(2));

        // Verify the valid lists contain correct data.
        let list0 = rebuilt.list_elements_at(0).to_primitive();
        assert_eq!(list0.as_slice::<i32>(), &[1, 2]);

        let list2 = rebuilt.list_elements_at(2).to_primitive();
        assert_eq!(list2.as_slice::<i32>(), &[5]);
    }

    #[test]
    fn test_rebuild_remove_nulls_non_nullable() {
        // Test that non-nullable arrays behave identically to RemoveGaps mode.
        // Elements: [_, A, B, _, C, D, _]
        //            0  1  2  3  4  5  6
        // List 0: offset=1, size=2 -> [A, B]
        // List 1: offset=4, size=2 -> [C, D]
        let elements = PrimitiveArray::from_iter(vec![99i32, 1, 2, 98, 3, 4, 97]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![1u32, 4]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 2]).into_array();

        let listview = ListViewArray::try_new(
            elements,
            offsets,
            sizes,
            Validity::NonNullable,
            ListViewShape::as_zero_copy_to_list().with_no_gaps(false),
        )
        .unwrap();

        // Apply both RemoveNulls and RemoveGaps to verify they produce identical results.
        let remove_nulls_result = listview.rebuild(ListViewRebuildMode::RemoveNulls);
        let remove_gaps_result = listview.rebuild(ListViewRebuildMode::RemoveGaps);

        // Both should produce the same compacted result.
        assert_eq!(
            remove_nulls_result.elements().len(),
            remove_gaps_result.elements().len()
        );
        assert_eq!(
            remove_nulls_result.offset_at(0),
            remove_gaps_result.offset_at(0)
        );
        assert_eq!(
            remove_nulls_result.offset_at(1),
            remove_gaps_result.offset_at(1)
        );

        // Verify the result is correct: [A, B, C, D] = [1, 2, 3, 4].
        assert_eq!(remove_nulls_result.elements().len(), 4);
        assert_eq!(remove_nulls_result.offset_at(0), 0);
        assert_eq!(remove_nulls_result.offset_at(1), 2);
    }
}
