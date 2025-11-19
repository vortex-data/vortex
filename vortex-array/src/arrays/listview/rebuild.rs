// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::FromPrimitive;
use vortex_buffer::BufferMut;
use vortex_dtype::{IntegerPType, Nullability, match_each_integer_ptype};
use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use crate::arrays::{ChunkedArray, ListViewArray};
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

    /// Equivalent to `MakeZeroCopyToList` plus `TrimElements`.
    ///
    /// This is useful when concatenating multiple [`ListViewArray`]s together to create a new
    /// [`ListViewArray`] that is also zero-copy to a [`ListArray`].
    ///
    /// [`ListArray`]: crate::arrays::ListArray
    MakeExact,

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
            ListViewRebuildMode::MakeExact => self.rebuild_make_exact(),
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
        if self.is_zero_copy_to_list() {
            // Note that since everything in `ListViewArray` is `Arc`ed, this is quite cheap.
            return self.clone();
        }

        let offsets_ptype = self.offsets().dtype().as_ptype();
        let sizes_ptype = self.sizes().dtype().as_ptype();

        match_each_integer_ptype!(sizes_ptype, |S| {
            match offsets_ptype {
                PType::U8 => self.naive_rebuild::<u8, u32, S>(),
                PType::U16 => self.naive_rebuild::<u16, u32, S>(),
                PType::U32 => self.naive_rebuild::<u32, u32, S>(),
                PType::U64 => self.naive_rebuild::<u64, u64, S>(),
                PType::I8 => self.naive_rebuild::<i8, i32, S>(),
                PType::I16 => self.naive_rebuild::<i16, i32, S>(),
                PType::I32 => self.naive_rebuild::<i32, i32, S>(),
                PType::I64 => self.naive_rebuild::<i64, i64, S>(),
                _ => unreachable!("invalid offsets PType"),
            }
        })
    }

    /// The inner function for `rebuild_zero_copy_to_list`, which naively rebuilds a `ListViewArray`
    /// via `append_scalar`.
    fn naive_rebuild<O: IntegerPType, NewOffset: IntegerPType, S: IntegerPType>(
        &self,
    ) -> ListViewArray {
        let element_dtype = self
            .dtype()
            .as_list_element_opt()
            .vortex_expect("somehow had a canonical list that was not a list");

        // Upfront canonicalize the list elements, we're going to be doing a lot of
        // slicing with them.
        let elements_canonical = self.elements().to_canonical().into_array();
        let offsets_canonical = self.offsets().to_primitive();
        let sizes_canonical = self.sizes().to_primitive();

        let offsets_canonical = offsets_canonical.as_slice::<O>();
        let sizes_canonical = sizes_canonical.as_slice::<S>();

        let mut offsets = BufferMut::<NewOffset>::with_capacity(self.len());
        let mut sizes = BufferMut::<S>::with_capacity(self.len());

        let mut chunks = Vec::with_capacity(self.len());

        let mut n_elements = NewOffset::zero();

        for index in 0..self.len() {
            if !self.is_valid(index) {
                offsets.push(offsets.last().copied().unwrap_or_default());
                sizes.push(S::zero());
                continue;
            }

            let offset = offsets_canonical[index];
            let size = sizes_canonical[index];

            let start = offset.as_();
            let stop = start + size.as_();

            chunks.push(elements_canonical.slice(start..stop));
            offsets.push(n_elements);
            sizes.push(size);

            n_elements += num_traits::cast(size).vortex_expect("cast");
        }

        let offsets = offsets.into_array();
        let sizes = sizes.into_array();

        // SAFETY: all chunks were sliced from the same array so have same DType.
        let elements =
            unsafe { ChunkedArray::new_unchecked(chunks, element_dtype.as_ref().clone()) };

        // SAFETY: elements are contiguous, offsets and sizes hand-built to be zero copy
        //  to list.
        unsafe {
            ListViewArray::new_unchecked(
                elements.to_canonical().into_array(),
                offsets,
                sizes,
                self.validity.clone(),
            )
            .with_zero_copy_to_list(true)
        }
    }

    /// Rebuilds a [`ListViewArray`] by trimming any unused / unreferenced leading and trailing
    /// elements, which is defined as a contiguous run of values in the `elements` array that are
    /// not referecened by any views in the corresponding [`ListViewArray`].
    fn rebuild_trim_elements(&self) -> ListViewArray {
        let start = if self.is_zero_copy_to_list() {
            // If offsets are sorted, then the minimum offset is the first offset.
            // Note that even if the first view is null, offsets must always be valid, so it is
            // completely fine for us to use this as a lower-bounded start of the `elements`.
            self.offset_at(0)
        } else {
            self.offsets().statistics().compute_min().vortex_expect(
                "[ListViewArray::rebuild]: `offsets` must report min statistic that is a `usize`",
            )
        };

        let end = if self.is_zero_copy_to_list() {
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
        // are valid. And if the original array was zero-copyable to list, trimming elements doesn't
        // change that property.
        unsafe {
            ListViewArray::new_unchecked(
                sliced_elements,
                adjusted_offsets,
                self.sizes().clone(),
                self.validity().clone(),
            )
            .with_zero_copy_to_list(self.is_zero_copy_to_list())
        }
    }

    fn rebuild_make_exact(&self) -> ListViewArray {
        if self.is_zero_copy_to_list() {
            self.rebuild_trim_elements()
        } else {
            // When we completely rebuild the `ListViewArray`, we get the benefit that we also trim
            // any leading and trailing garbage data.
            self.rebuild_zero_copy_to_list()
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_dtype::Nullability;

    use super::ListViewRebuildMode;
    use crate::arrays::{ListViewArray, PrimitiveArray};
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;
    use crate::{IntoArray, ToCanonical};

    #[test]
    fn test_rebuild_flatten_removes_overlaps() {
        // Create a list view with overlapping lists: [A, B, C]
        // List 0: offset=0, size=3 -> [A, B, C]
        // List 1: offset=1, size=2 -> [B, C] (overlaps with List 0)
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 1]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![3u32, 2]).into_array();

        let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);

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

        let listview = ListViewArray::new(elements, offsets, sizes, validity);

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

        let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);

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
}
