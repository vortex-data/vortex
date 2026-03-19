// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::FromPrimitive;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DynArray;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::ToCanonical;
use crate::VortexSessionExecute;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::arrays::ConstantArray;
use crate::arrays::ListViewArray;
use crate::builders::builder_with_capacity;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;
use crate::vtable::ValidityHelper;

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
    pub fn rebuild(&self, mode: ListViewRebuildMode) -> VortexResult<ListViewArray> {
        if self.is_empty() {
            // SAFETY: An empty array is trivially zero-copyable to a `ListArray`.
            return Ok(unsafe { self.clone().with_zero_copy_to_list(true) });
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
    fn rebuild_zero_copy_to_list(&self) -> VortexResult<ListViewArray> {
        if self.is_zero_copy_to_list() {
            // Note that since everything in `ListViewArray` is `Arc`ed, this is quite cheap.
            return Ok(self.clone());
        }

        let offsets_ptype = self.offsets().dtype().as_ptype();
        let sizes_ptype = self.sizes().dtype().as_ptype();

        // One of the main purposes behind adding this "zero-copyable to `ListArray`" optimization
        // is that we want to pass data to systems that expect Arrow data.
        // The arrow specification only allows for `i32` and `i64` offset and sizes types, so in
        // order to also make `ListView` zero-copyable to **Arrow**'s `ListArray` (not just Vortex's
        // `ListArray`), we rebuild the offsets as 32-bit or 64-bit integer types.
        // TODO(connor)[ListView]: This is true for `sizes` as well, we should do this conversion
        // for sizes as well.
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

    /// Picks between [`rebuild_with_take`](Self::rebuild_with_take) and
    /// [`rebuild_list_by_list`](Self::rebuild_list_by_list) based on element dtype and average
    /// list size.
    fn naive_rebuild<O: IntegerPType, NewOffset: IntegerPType, S: IntegerPType>(
        &self,
    ) -> VortexResult<ListViewArray> {
        let sizes_canonical = self.sizes().to_primitive();
        let total: u64 = sizes_canonical
            .as_slice::<S>()
            .iter()
            .map(|s| (*s).as_() as u64)
            .sum();
        if Self::should_use_take(total, self.len()) {
            self.rebuild_with_take::<O, NewOffset, S>()
        } else {
            self.rebuild_list_by_list::<O, NewOffset, S>()
        }
    }

    /// Returns `true` when we are confident that `rebuild_with_take` will
    /// outperform `rebuild_list_by_list`.
    ///
    /// Take is dramatically faster for small lists (often 10-100×) because it
    /// avoids per-list builder overhead. LBL is the safer default for larger
    /// lists since its sequential memcpy scales well. We only choose take when
    /// the average list size is small enough that take clearly dominates.
    fn should_use_take(total_output_elements: u64, num_lists: usize) -> bool {
        if num_lists == 0 {
            return true;
        }
        let avg = total_output_elements / num_lists as u64;
        avg < 128
    }

    /// Rebuilds elements using a single bulk `take`: collect all element indices into a flat
    /// `BufferMut<u64>`, perform a single `take`.
    fn rebuild_with_take<O: IntegerPType, NewOffset: IntegerPType, S: IntegerPType>(
        &self,
    ) -> VortexResult<ListViewArray> {
        let offsets_canonical = self.offsets().to_primitive();
        let offsets_slice = offsets_canonical.as_slice::<O>();
        let sizes_canonical = self.sizes().to_primitive();
        let sizes_slice = sizes_canonical.as_slice::<S>();

        let len = offsets_slice.len();

        let mut new_offsets = BufferMut::<NewOffset>::with_capacity(len);
        let mut new_sizes = BufferMut::<S>::with_capacity(len);
        let mut take_indices = BufferMut::<u64>::with_capacity(self.elements().len());

        let mut n_elements = NewOffset::zero();
        for index in 0..len {
            if !self.is_valid(index)? {
                new_offsets.push(n_elements);
                new_sizes.push(S::zero());
                continue;
            }

            let offset = offsets_slice[index];
            let size = sizes_slice[index];
            let start = offset.as_();
            let stop = start + size.as_();

            new_offsets.push(n_elements);
            new_sizes.push(size);
            take_indices.extend(start as u64..stop as u64);
            n_elements += num_traits::cast(size).vortex_expect("Cast failed");
        }

        let elements = self.elements().take(take_indices.into_array())?;
        let offsets = new_offsets.into_array();
        let sizes = new_sizes.into_array();

        // SAFETY: same invariants as `rebuild_list_by_list` — offsets are sequential and
        // non-overlapping, all (offset, size) pairs reference valid elements, and the validity
        // array is preserved from the original.
        Ok(unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, self.validity.clone())
                .with_zero_copy_to_list(true)
        })
    }

    /// Rebuilds elements list-by-list: canonicalize elements upfront, then for each list `slice`
    /// the relevant range and `extend_from_array` into a typed builder.
    fn rebuild_list_by_list<O: IntegerPType, NewOffset: IntegerPType, S: IntegerPType>(
        &self,
    ) -> VortexResult<ListViewArray> {
        let element_dtype = self
            .dtype()
            .as_list_element_opt()
            .vortex_expect("somehow had a canonical list that was not a list");

        let offsets_canonical = self.offsets().to_primitive();
        let offsets_slice = offsets_canonical.as_slice::<O>();
        let sizes_canonical = self.sizes().to_primitive();
        let sizes_slice = sizes_canonical.as_slice::<S>();

        let len = offsets_slice.len();

        let mut new_offsets = BufferMut::<NewOffset>::with_capacity(len);
        // TODO(connor)[ListView]: Do we really need to do this?
        // The only reason we need to rebuild the sizes here is that the validity may indicate that
        // a list is null even though it has a non-zero size. This rebuild will set the size of all
        // null lists to 0.
        let mut new_sizes = BufferMut::<S>::with_capacity(len);

        // Canonicalize the elements up front as we will be slicing the elements quite a lot.
        let elements_canonical = self
            .elements()
            .to_canonical()
            .vortex_expect("canonicalize elements for rebuild")
            .into_array();

        // Note that we do not know what the exact capacity should be of the new elements since
        // there could be overlaps in the existing `ListViewArray`.
        let mut new_elements_builder =
            builder_with_capacity(element_dtype.as_ref(), self.elements().len());

        let mut n_elements = NewOffset::zero();
        for index in 0..len {
            if !self.is_valid(index)? {
                // For NULL lists, place them after the previous item's data to maintain the
                // no-overlap invariant for zero-copy to `ListArray` arrays.
                new_offsets.push(n_elements);
                new_sizes.push(S::zero());
                continue;
            }

            let offset = offsets_slice[index];
            let size = sizes_slice[index];

            let start = offset.as_();
            let stop = start + size.as_();

            new_offsets.push(n_elements);
            new_sizes.push(size);
            new_elements_builder.extend_from_array(&elements_canonical.slice(start..stop)?);

            n_elements += num_traits::cast(size).vortex_expect("Cast failed");
        }

        let offsets = new_offsets.into_array();
        let sizes = new_sizes.into_array();
        let elements = new_elements_builder.finish();

        debug_assert_eq!(
            n_elements.as_(),
            elements.len(),
            "The accumulated elements somehow had the wrong length"
        );

        // SAFETY:
        // - All offsets are sequential and non-overlapping (`n_elements` tracks running total).
        // - Each `offset[i] + size[i]` equals `offset[i+1]` for all valid indices (including null
        //   lists).
        // - All elements referenced by (offset, size) pairs exist within the new `elements` array.
        // - The validity array is preserved from the original array unchanged
        // - The array satisfies the zero-copy-to-list property by having sorted offsets, no gaps,
        //   and no overlaps.
        Ok(unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, self.validity.clone())
                .with_zero_copy_to_list(true)
        })
    }

    /// Rebuilds a [`ListViewArray`] by trimming any unused / unreferenced leading and trailing
    /// elements, which is defined as a contiguous run of values in the `elements` array that are
    /// not referecened by any views in the corresponding [`ListViewArray`].
    fn rebuild_trim_elements(&self) -> VortexResult<ListViewArray> {
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
            // Cast offsets and sizes to the widest integer type to prevent
            // overflow when computing offsets + sizes. The end offset may not
            // fit in the integer width otherwise.
            let wide_dtype = DType::from(if self.offsets().dtype().as_ptype().is_unsigned_int() {
                PType::U64
            } else {
                PType::I64
            });
            let offsets = self.offsets().cast(wide_dtype.clone())?;
            let sizes = self.sizes().cast(wide_dtype)?;

            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let min_max = min_max(
                &offsets
                    .binary(sizes, Operator::Add)
                    .vortex_expect("`offsets + sizes` somehow overflowed"),
                &mut ctx,
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

            self.offsets()
                .to_array()
                .binary(
                    ConstantArray::new(scalar, self.offsets().len()).into_array(),
                    Operator::Sub,
                )
                .vortex_expect("was somehow unable to adjust offsets down by their minimum")
        });

        let sliced_elements = self.elements().slice(start..end)?;

        // SAFETY: The only thing we changed was the elements (which we verify through mins and
        // maxes that all adjusted offsets + sizes are within the correct bounds), so the parameters
        // are valid. And if the original array was zero-copyable to list, trimming elements doesn't
        // change that property.
        Ok(unsafe {
            ListViewArray::new_unchecked(
                sliced_elements,
                adjusted_offsets,
                self.sizes().clone(),
                self.validity().clone(),
            )
            .with_zero_copy_to_list(self.is_zero_copy_to_list())
        })
    }

    fn rebuild_make_exact(&self) -> VortexResult<ListViewArray> {
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
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;

    use super::ListViewRebuildMode;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::Nullability;
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;

    #[test]
    fn test_rebuild_flatten_removes_overlaps() -> VortexResult<()> {
        // Create a list view with overlapping lists: [A, B, C]
        // List 0: offset=0, size=3 -> [A, B, C]
        // List 1: offset=1, size=2 -> [B, C] (overlaps with List 0)
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 1]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![3u32, 2]).into_array();

        let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);

        let flattened = listview.rebuild(ListViewRebuildMode::MakeZeroCopyToList)?;

        // After flatten: elements should be [A, B, C, B, C] = [1, 2, 3, 2, 3]
        // Lists should be sequential with no overlaps
        assert_eq!(flattened.elements().len(), 5);

        // Offsets should be sequential
        assert_eq!(flattened.offset_at(0), 0);
        assert_eq!(flattened.size_at(0), 3);
        assert_eq!(flattened.offset_at(1), 3);
        assert_eq!(flattened.size_at(1), 2);

        // Verify the data is correct
        assert_arrays_eq!(
            flattened.list_elements_at(0).unwrap(),
            PrimitiveArray::from_iter([1i32, 2, 3])
        );

        assert_arrays_eq!(
            flattened.list_elements_at(1).unwrap(),
            PrimitiveArray::from_iter([2i32, 3])
        );
        Ok(())
    }

    #[test]
    fn test_rebuild_flatten_with_nullable() -> VortexResult<()> {
        use crate::arrays::BoolArray;

        // Create a nullable list view with a null list
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 1, 2]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 1, 1]).into_array();
        let validity = Validity::Array(
            BoolArray::new(
                BitBuffer::from(vec![true, false, true]),
                Validity::NonNullable,
            )
            .into_array(),
        );

        let listview = ListViewArray::new(elements, offsets, sizes, validity);

        let flattened = listview.rebuild(ListViewRebuildMode::MakeZeroCopyToList)?;

        // Verify nullability is preserved
        assert_eq!(flattened.dtype().nullability(), Nullability::Nullable);
        assert!(flattened.validity().is_valid(0).unwrap());
        assert!(!flattened.validity().is_valid(1).unwrap());
        assert!(flattened.validity().is_valid(2).unwrap());

        // Verify valid lists contain correct data
        assert_arrays_eq!(
            flattened.list_elements_at(0).unwrap(),
            PrimitiveArray::from_iter([1i32, 2])
        );

        assert_arrays_eq!(
            flattened.list_elements_at(2).unwrap(),
            PrimitiveArray::from_iter([3i32])
        );
        Ok(())
    }

    #[test]
    fn test_rebuild_trim_elements_basic() -> VortexResult<()> {
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

        let trimmed = listview.rebuild(ListViewRebuildMode::TrimElements)?;

        // After trimming: elements should be [A, B, _, C, D] = [1, 2, 97, 3, 4].
        assert_eq!(trimmed.elements().len(), 5);

        // Offsets should be adjusted: old offset 2 -> new offset 0, old offset 5 -> new offset 3.
        assert_eq!(trimmed.offset_at(0), 0);
        assert_eq!(trimmed.size_at(0), 2);
        assert_eq!(trimmed.offset_at(1), 3);
        assert_eq!(trimmed.size_at(1), 2);

        // Verify the data is correct.
        assert_arrays_eq!(
            trimmed.list_elements_at(0).unwrap(),
            PrimitiveArray::from_iter([1i32, 2])
        );

        assert_arrays_eq!(
            trimmed.list_elements_at(1).unwrap(),
            PrimitiveArray::from_iter([3i32, 4])
        );

        // Note that element at index 2 (97) is preserved as a gap.
        let all_elements = trimmed.elements().to_primitive();
        assert_eq!(all_elements.scalar_at(2).unwrap(), 97i32.into());
        Ok(())
    }

    #[test]
    fn test_rebuild_with_trailing_nulls_regression() -> VortexResult<()> {
        // Regression test for issue #5412
        // Tests that zero-copy-to-list arrays with trailing NULLs correctly calculate
        // offsets for NULL items to maintain no-overlap invariant

        // Create a ListViewArray with trailing NULLs
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3, 4]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 2, 0, 0]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 2, 0, 0]).into_array();
        let validity = Validity::from_iter(vec![true, true, false, false]);

        let listview = ListViewArray::new(elements, offsets, sizes, validity);

        // First rebuild to make it zero-copy-to-list
        let rebuilt = listview.rebuild(ListViewRebuildMode::MakeZeroCopyToList)?;
        assert!(rebuilt.is_zero_copy_to_list());

        // Verify NULL items have correct offsets (should not reuse previous offsets)
        // After rebuild: offsets should be [0, 2, 4, 4] for zero-copy-to-list
        assert_eq!(rebuilt.offset_at(0), 0);
        assert_eq!(rebuilt.offset_at(1), 2);
        assert_eq!(rebuilt.offset_at(2), 4); // NULL should be at position 4
        assert_eq!(rebuilt.offset_at(3), 4); // Second NULL also at position 4

        // All sizes should be correct
        assert_eq!(rebuilt.size_at(0), 2);
        assert_eq!(rebuilt.size_at(1), 2);
        assert_eq!(rebuilt.size_at(2), 0); // NULL has size 0
        assert_eq!(rebuilt.size_at(3), 0); // NULL has size 0

        // Now rebuild with MakeExact (which calls naive_rebuild then trim_elements)
        // This should not panic (issue #5412)
        let exact = rebuilt.rebuild(ListViewRebuildMode::MakeExact)?;

        // Verify the result is still valid
        assert!(exact.is_valid(0).unwrap());
        assert!(exact.is_valid(1).unwrap());
        assert!(!exact.is_valid(2).unwrap());
        assert!(!exact.is_valid(3).unwrap());

        // Verify data is preserved
        assert_arrays_eq!(
            exact.list_elements_at(0).unwrap(),
            PrimitiveArray::from_iter([1i32, 2])
        );

        assert_arrays_eq!(
            exact.list_elements_at(1).unwrap(),
            PrimitiveArray::from_iter([3i32, 4])
        );
        Ok(())
    }

    /// Regression test for <https://github.com/vortex-data/vortex/issues/6773>.
    /// u32 offsets exceed u16::MAX, so u16 sizes are widened to u32 for the add.
    #[test]
    fn test_rebuild_trim_elements_offsets_wider_than_sizes() -> VortexResult<()> {
        let mut elems = vec![0i32; 70_005];
        elems[70_000] = 10;
        elems[70_001] = 20;
        elems[70_002] = 30;
        elems[70_003] = 40;
        let elements = PrimitiveArray::from_iter(elems).into_array();
        let offsets = PrimitiveArray::from_iter(vec![70_000u32, 70_002]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u16, 2]).into_array();

        let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);
        let trimmed = listview.rebuild(ListViewRebuildMode::TrimElements)?;
        assert_arrays_eq!(
            trimmed.list_elements_at(1).unwrap(),
            PrimitiveArray::from_iter([30i32, 40])
        );
        Ok(())
    }

    /// Regression test for <https://github.com/vortex-data/vortex/issues/6773>.
    /// u32 sizes exceed u16::MAX, so u16 offsets are widened to u32 for the add.
    #[test]
    fn test_rebuild_trim_elements_sizes_wider_than_offsets() -> VortexResult<()> {
        let mut elems = vec![0i32; 70_001];
        elems[3] = 30;
        elems[4] = 40;
        let elements = PrimitiveArray::from_iter(elems).into_array();
        let offsets = PrimitiveArray::from_iter(vec![1u16, 3]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![70_000u32, 2]).into_array();

        let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);
        let trimmed = listview.rebuild(ListViewRebuildMode::TrimElements)?;
        assert_arrays_eq!(
            trimmed.list_elements_at(1).unwrap(),
            PrimitiveArray::from_iter([30i32, 40])
        );
        Ok(())
    }

    // ── should_use_take heuristic tests ────────────────────────────────────

    #[test]
    fn heuristic_zero_lists_uses_take() {
        assert!(ListViewArray::should_use_take(0, 0));
    }

    #[test]
    fn heuristic_small_lists_use_take() {
        // avg = 127 → take
        assert!(ListViewArray::should_use_take(127_000, 1_000));
        // avg = 128 → LBL
        assert!(!ListViewArray::should_use_take(128_000, 1_000));
    }

    /// Regression test for <https://github.com/vortex-data/vortex/issues/6973>.
    /// Both offsets and sizes are u8, and offset + size exceeds u8::MAX.
    #[test]
    fn test_rebuild_trim_elements_sum_overflows_type() -> VortexResult<()> {
        let elements = PrimitiveArray::from_iter(vec![0i32; 261]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![215u8, 0]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![46u8, 10]).into_array();

        let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);
        let trimmed = listview.rebuild(ListViewRebuildMode::TrimElements)?;

        // min(offsets) = 0, so nothing to trim; output should equal input.
        assert_arrays_eq!(trimmed, listview);
        Ok(())
    }
}
