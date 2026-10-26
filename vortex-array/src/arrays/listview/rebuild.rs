// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::FromPrimitive;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ListViewArray;
use crate::arrays::PiecewiseSequenceArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::list::drop_unreferenced_elements;
use crate::arrays::list::null_runs_holding_elements;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_integer_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

/// The widened offset type used for a rebuilt, zero-copy-to-list array.
///
/// Offsets are widened to at least 32 bits (Arrow only permits 32/64-bit list offsets) while
/// preserving signedness, since a signed result keeps the array zero-copyable to Arrow's
/// `ListArray`.
fn rebuilt_offset_ptype(offsets_ptype: PType) -> PType {
    match offsets_ptype {
        PType::U8 | PType::U16 | PType::U32 => PType::U32,
        PType::U64 => PType::U64,
        PType::I8 | PType::I16 | PType::I32 => PType::I32,
        PType::I64 => PType::I64,
        _ => unreachable!("invalid offsets PType"),
    }
}

/// Density threshold to decide whether to rebuild a sparse `ListViewArray`.
///
/// A `ListViewArray` can accumulate unreferenced bytes in its `elements` buffer after
/// metadata-only operations like `take` and `filter`. When density (referenced fraction of `elements`)
/// falls below this threshold, the benefits of a rebuild may outweigh its cost.
///
/// This is a somewhat arbitrary rule-of-thumb and may be suboptimal depending on different use cases and
/// list element dtypes.
pub const DEFAULT_REBUILD_DENSITY_THRESHOLD: f32 = 0.1;

/// Waste threshold to decide whether to trim a zero-copy-to-list `ListViewArray`.
///
/// A zero-copy-to-list array has no overlaps and no interior gaps, so its only unreferenced bytes
/// are leading and trailing elements. Trimming those is much cheaper than a full rebuild (a lazy
/// `elements` slice plus an `O(num_lists)` offset adjustment, with no element data copy), so we use
/// a more aggressive threshold than [`DEFAULT_REBUILD_DENSITY_THRESHOLD`].
///
/// When the unreferenced (leading + trailing) fraction of `elements` exceeds this threshold, we trim.
pub const DEFAULT_TRIM_ELEMENTS_THRESHOLD: f32 = 0.05;

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
    ///
    /// If the referenced `[start, end)` bounds are already known, prefer calling
    /// [`trim_elements`](ListViewArray::trim_elements) directly to avoid recomputing them.
    TrimElements,

    /// Equivalent to `MakeZeroCopyToList` plus `TrimElements`.
    ///
    /// An array that is not yet zero-copy to a [`ListArray`] is copied with only its valid lists'
    /// elements, and its null lists come out empty. One that already is zero-copy only loses the
    /// elements before its first list and after its last, so its null lists keep theirs; use
    /// [`MakeExactWithEmptyNulls`](Self::MakeExactWithEmptyNulls) to drop those too.
    ///
    /// This is useful when concatenating multiple [`ListViewArray`]s together to create a new
    /// [`ListViewArray`] that is also zero-copy to a [`ListArray`].
    ///
    /// [`ListArray`]: crate::arrays::ListArray
    MakeExact,

    /// Equivalent to `MakeExact`, but the null lists of an array that is already zero-copy to a
    /// [`ListArray`] come out empty as well, so only the valid lists' elements are left.
    ///
    /// This is useful for a function that must not see elements no valid list shows, such as one
    /// that can fail on a value. An array that is already zero-copy to a [`ListArray`] keeps its
    /// offsets' and sizes' types, and is returned as it is when there is nothing to drop.
    ///
    /// [`ListArray`]: crate::arrays::ListArray
    MakeExactWithEmptyNulls,

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
    pub fn rebuild(
        &self,
        mode: ListViewRebuildMode,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ListViewArray> {
        if self.is_empty() {
            // An empty array uses none of its elements, so every mode drops them.
            return Ok(Canonical::empty(self.dtype()).into_listview());
        }

        match mode {
            ListViewRebuildMode::MakeZeroCopyToList => self.rebuild_zero_copy_to_list(ctx),
            ListViewRebuildMode::TrimElements => self.rebuild_trim_elements(ctx),
            ListViewRebuildMode::MakeExact => self.rebuild_make_exact(ctx),
            ListViewRebuildMode::MakeExactWithEmptyNulls => {
                self.rebuild_make_exact_with_empty_nulls(ctx)
            }
            ListViewRebuildMode::OverlapCompression => {
                vortex_bail!("ListViewRebuildMode::OverlapCompression is not implemented")
            }
        }
    }

    /// Rebuilds a non-empty [`ListViewArray`] over only the elements that valid lists reference:
    /// nothing before the first list, after the last, between lists or under a null list.
    fn rebuild_make_exact_with_empty_nulls(
        &self,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ListViewArray> {
        if !self.is_zero_copy_to_list() {
            return self.rebuild_make_exact(ctx);
        }
        let validity = self.validity()?;
        // The lists are back to back, so the last one ends where the referenced elements do.
        let last = self.len() - 1;
        let end = self.offset_at(last, ctx) + self.size_at(last, ctx);
        if validity.definitely_no_nulls()
            && self.offset_at(0, ctx) == 0
            && end == self.elements().len()
        {
            return Ok(self.clone());
        }

        let offsets = self.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let offsets_ptype = offsets.ptype();
        // Offsets are never negative, so they are rebuilt unsigned and reinterpreted back.
        let offsets = offsets.reinterpret_cast(offsets_ptype.to_unsigned());
        let valid = validity.execute_mask(self.len(), ctx)?;
        match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
            self.rebuild_back_to_back(offsets.as_slice::<O>(), offsets_ptype, end, &valid, ctx)
        })
    }

    /// [`MakeExactWithEmptyNulls`](ListViewRebuildMode::MakeExactWithEmptyNulls) for an array that
    /// is zero-copy to a list, whose lists start at `starts` and end at `end`.
    fn rebuild_back_to_back<O: IntegerPType>(
        &self,
        starts: &[O],
        offsets_ptype: PType,
        end: usize,
        valid: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ListViewArray> {
        // A list holds elements if and only if the next one starts later.
        let null_runs = null_runs_holding_elements(valid, |index| {
            starts[index].as_() != starts.get(index + 1).map_or(end, |next| next.as_())
        });
        if null_runs.is_empty() && starts[0].as_() == 0 && end == self.elements().len() {
            return Ok(self.clone());
        }

        let (elements, offsets) =
            drop_unreferenced_elements(self.elements(), starts, end, &null_runs, ctx.allocator())?;
        let offsets = PrimitiveArray::new(offsets.freeze(), Validity::NonNullable)
            .reinterpret_cast(offsets_ptype)
            .into_array();
        let sizes = if null_runs.is_empty() {
            self.sizes().clone()
        } else {
            let sizes = self.sizes().clone().execute::<PrimitiveArray>(ctx)?;
            let sizes_ptype = sizes.ptype();
            let sizes = sizes.reinterpret_cast(sizes_ptype.to_unsigned());
            let sizes = match_each_unsigned_integer_ptype!(sizes.ptype(), |S| {
                empty_runs(sizes.as_slice::<S>(), &null_runs, ctx.allocator())
            });
            sizes.reinterpret_cast(sizes_ptype).into_array()
        };

        // SAFETY: the lists stay back to back and in order over the elements kept: each valid list
        // keeps its elements, and each null list that held elements is now empty.
        Ok(unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, self.validity()?)
                .with_zero_copy_to_list(true)
        })
    }

    /// Rebuilds a [`ListViewArray`], removing all data overlaps and creating a flattened layout.
    ///
    /// This is useful when the `elements` child array of the [`ListViewArray`] might have
    /// overlapping, duplicate, and garbage data, and we want to have fully sequential data like
    /// a [`ListArray`].
    ///
    /// [`ListArray`]: crate::arrays::ListArray
    fn rebuild_zero_copy_to_list(&self, ctx: &mut ExecutionCtx) -> VortexResult<ListViewArray> {
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
        match_each_unsigned_integer_ptype!(sizes_ptype.to_unsigned(), |S| {
            match offsets_ptype.to_unsigned() {
                PType::U8 => self.naive_rebuild::<u8, u32, S>(ctx),
                PType::U16 => self.naive_rebuild::<u16, u32, S>(ctx),
                PType::U32 => self.naive_rebuild::<u32, u32, S>(ctx),
                PType::U64 => self.naive_rebuild::<u64, u64, S>(ctx),
                _ => unreachable!("invalid offsets PType"),
            }
        })
    }

    /// Picks between [`rebuild_with_take`](Self::rebuild_with_take) and
    /// [`rebuild_with_piecewise`](Self::rebuild_with_piecewise) based on average list size.
    fn naive_rebuild<O: IntegerPType, NewOffset: IntegerPType, S: IntegerPType>(
        &self,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ListViewArray> {
        let sizes_canonical = self.sizes().clone().execute::<PrimitiveArray>(ctx)?;
        let sizes_canonical =
            sizes_canonical.reinterpret_cast(sizes_canonical.ptype().to_unsigned());
        let total: u64 = sizes_canonical
            .as_slice::<S>()
            .iter()
            .map(|s| (*s).as_() as u64)
            .sum();
        if Self::should_use_take(total, self.len()) {
            self.rebuild_with_take::<O, NewOffset, S>(ctx)
        } else {
            self.rebuild_with_piecewise::<NewOffset, S>(ctx)
        }
    }

    /// Returns `true` when we are confident that `rebuild_with_take` will
    /// outperform `rebuild_with_piecewise`.
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
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ListViewArray> {
        let new_offset_ptype = rebuilt_offset_ptype(self.offsets().dtype().as_ptype());
        let size_ptype = self.sizes().dtype().as_ptype();

        let offsets_canonical = self.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let offsets_canonical =
            offsets_canonical.reinterpret_cast(offsets_canonical.ptype().to_unsigned());
        let offsets_slice = offsets_canonical.as_slice::<O>();
        let sizes_canonical = self.sizes().clone().execute::<PrimitiveArray>(ctx)?;
        let sizes_canonical =
            sizes_canonical.reinterpret_cast(sizes_canonical.ptype().to_unsigned());
        let sizes_slice = sizes_canonical.as_slice::<S>();

        let len = offsets_slice.len();

        let mut new_offsets = BufferMut::<NewOffset>::with_capacity(len);
        let mut new_sizes = BufferMut::<S>::with_capacity(len);
        let mut take_indices = BufferMut::<u64>::with_capacity(self.elements().len());

        // Resolve validity to a mask once instead of probing it per row: `execute_is_valid`
        // executes a scalar on every call for array-backed validity, which is O(len) work repeated
        // `len` times.
        let validity = self.validity()?.execute_mask(len, ctx)?;

        let mut n_elements = NewOffset::zero();
        for index in 0..len {
            if !validity.value(index) {
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
        // Built unsigned; reinterpret back to the signed-preserving result types.
        let offsets = PrimitiveArray::new(new_offsets.freeze(), Validity::NonNullable)
            .reinterpret_cast(new_offset_ptype)
            .into_array();
        let sizes = PrimitiveArray::new(new_sizes.freeze(), Validity::NonNullable)
            .reinterpret_cast(size_ptype)
            .into_array();

        // SAFETY: offsets are sequential and non-overlapping, all (offset, size) pairs reference
        // valid elements, and the validity array is preserved from the original.
        Ok(unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, self.validity()?)
                .with_zero_copy_to_list(true)
        })
    }

    /// Rebuilds elements using one contiguous-run take over the element child.
    fn rebuild_with_piecewise<NewOffset: IntegerPType, S: IntegerPType>(
        &self,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ListViewArray> {
        let new_offset_ptype = rebuilt_offset_ptype(self.offsets().dtype().as_ptype());
        let size_ptype = self.sizes().dtype().as_ptype();

        let offsets_canonical = self.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let offsets_canonical =
            offsets_canonical.reinterpret_cast(offsets_canonical.ptype().to_unsigned());
        let sizes_canonical = self.sizes().clone().execute::<PrimitiveArray>(ctx)?;
        let sizes_canonical =
            sizes_canonical.reinterpret_cast(sizes_canonical.ptype().to_unsigned());

        let len = offsets_canonical.len();
        let validity = self.validity()?;
        let validity_mask = validity.execute_mask(len, ctx)?;

        let ranges = match_each_unsigned_integer_ptype!(offsets_canonical.ptype(), |O| {
            rebuild_ranges::<NewOffset, O, S>(
                offsets_canonical.as_slice::<O>(),
                sizes_canonical.as_slice::<S>(),
                &validity_mask,
            )
        })?;

        let RebuildRanges {
            new_offsets,
            new_sizes,
            starts,
            lengths,
            elements_len,
        } = ranges;
        let constant_length = lengths
            .first()
            .copied()
            .filter(|first| lengths.iter().all(|length| *length == *first));
        let lengths = match constant_length {
            Some(length) => ConstantArray::new(length, starts.len()).into_array(),
            None => lengths.into_array(),
        };

        // SAFETY: range starts and lengths are derived from valid ListView metadata; elements_len
        // is the sum of all generated range lengths. Multiplier 1 preserves contiguous ranges.
        let multipliers = ConstantArray::new(1u64, starts.len()).into_array();
        let element_indices = unsafe {
            PiecewiseSequenceArray::new_unchecked(
                starts.into_array(),
                lengths,
                multipliers,
                elements_len,
            )
        };
        let elements = self.elements().take(element_indices.into_array())?;

        // Built unsigned; reinterpret back to the signed-preserving result types.
        let offsets = PrimitiveArray::new(new_offsets.freeze(), Validity::NonNullable)
            .reinterpret_cast(new_offset_ptype)
            .into_array();
        let sizes = PrimitiveArray::new(new_sizes.freeze(), Validity::NonNullable)
            .reinterpret_cast(size_ptype)
            .into_array();

        // SAFETY: offsets are sequential and non-overlapping, all (offset, size) pairs reference
        // valid elements, and validity is preserved from the original array.
        Ok(unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, validity)
                .with_zero_copy_to_list(true)
        })
    }

    /// Rebuilds a [`ListViewArray`] by trimming any unused / unreferenced leading and trailing
    /// elements, which is defined as a contiguous run of values in the `elements` array that are
    /// not referenced by any views in the corresponding [`ListViewArray`].
    fn rebuild_trim_elements(&self, ctx: &mut ExecutionCtx) -> VortexResult<ListViewArray> {
        let (start, end) = self.referenced_element_bounds(ctx)?;

        // SAFETY: we calculated valid start and end bounds
        unsafe { self.trim_elements(start, end) }
    }

    /// Unsafely trims `elements` to the referenced half-open range `[start, end)`, adjusting every offset
    /// down by `start`. The result preserves the original `is_zero_copy_to_list` flag.
    ///
    /// # SAFETY
    ///
    /// `start` must be the minimum value of `offsets`, and end should be the maximum value of `offsets[i] + size[i]`
    /// over all indices `i` of offsets. Otherwise, `offsets` and `sizes` may hold references to elements that no longer exist
    /// and the array will be corrupted.
    pub unsafe fn trim_elements(&self, start: usize, end: usize) -> VortexResult<ListViewArray> {
        let adjusted_offsets = match_each_integer_ptype!(self.offsets().dtype().as_ptype(), |O| {
            let offset = <O as FromPrimitive>::from_usize(start)
                .vortex_expect("unable to convert the min offset `start` into a `usize`");
            let scalar = Scalar::primitive(offset, Nullability::NonNullable);

            self.offsets()
                .clone()
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
                self.validity()?,
            )
            .with_zero_copy_to_list(self.is_zero_copy_to_list())
        })
    }

    fn rebuild_make_exact(&self, ctx: &mut ExecutionCtx) -> VortexResult<ListViewArray> {
        if self.is_zero_copy_to_list() {
            self.rebuild_trim_elements(ctx)
        } else {
            // When we completely rebuild the `ListViewArray`, we get the benefit that we also trim
            // any leading and trailing garbage data.
            self.rebuild_zero_copy_to_list(ctx)
        }
    }
}

/// The sizes with the lists in `runs` emptied.
fn empty_runs<S: IntegerPType>(
    sizes: &[S],
    runs: &[(usize, usize)],
    allocator: &BufferAllocatorRef,
) -> PrimitiveArray {
    let mut emptied = BufferMut::with_capacity_in(sizes.len(), allocator.clone());
    let mut from = 0;
    for &(run_start, run_end) in runs {
        emptied.extend_from_slice(&sizes[from..run_start]);
        emptied.push_n(S::zero(), run_end - run_start);
        from = run_end;
    }
    emptied.extend_from_slice(&sizes[from..]);
    PrimitiveArray::new(emptied.freeze(), Validity::NonNullable)
}

struct RebuildRanges<NewOffset, S> {
    new_offsets: BufferMut<NewOffset>,
    new_sizes: BufferMut<S>,
    starts: BufferMut<u64>,
    lengths: BufferMut<u64>,
    elements_len: usize,
}

fn rebuild_ranges<NewOffset, O, S>(
    offsets: &[O],
    sizes: &[S],
    validity_mask: &Mask,
) -> VortexResult<RebuildRanges<NewOffset, S>>
where
    NewOffset: IntegerPType,
    O: IntegerPType,
    S: IntegerPType,
{
    let len = offsets.len();
    let mut new_offsets = BufferMut::<NewOffset>::with_capacity(len);
    let mut new_sizes = BufferMut::<S>::with_capacity(len);
    let mut starts = BufferMut::<u64>::with_capacity(len);
    let mut lengths = BufferMut::<u64>::with_capacity(len);
    let mut n_elements = NewOffset::zero();
    let mut elements_len = 0usize;

    for (index, is_valid) in validity_mask.iter().enumerate() {
        if !is_valid {
            new_offsets.push(n_elements);
            new_sizes.push(S::zero());
            starts.push(0);
            lengths.push(0);
            continue;
        }

        let size = sizes[index];
        let start: usize = offsets[index].as_();
        let length: usize = size.as_();
        start.checked_add(length).ok_or_else(|| {
            vortex_err!(
                "ListView rebuild element range overflow for start {start} and length {length}"
            )
        })?;
        elements_len = elements_len
            .checked_add(length)
            .ok_or_else(|| vortex_err!("ListView rebuild elements length overflow"))?;

        new_offsets.push(n_elements);
        new_sizes.push(size);
        starts.push(start as u64);
        lengths.push(length as u64);
        n_elements += num_traits::cast(size).vortex_expect("Cast failed");
    }

    Ok(RebuildRanges {
        new_offsets,
        new_sizes,
        starts,
        lengths,
        elements_len,
    })
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;

    use super::super::tests::common::SESSION;
    use super::ListViewRebuildMode;
    use crate::ArrayRef;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::list::ListArraySlotsExt;
    use crate::arrays::listview::ListViewArrayExt;
    use crate::arrays::listview::ListViewArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    #[rstest]
    #[case::nothing_to_drop(&[1, 2], &[(0, 1), (1, 1)], &[true, true], true, &[1, 2], &[(0, 1), (1, 1)])]
    #[case::leftovers_at_the_ends(
        &[9, 1, 2, 9],
        &[(1, 1), (2, 1)],
        &[true, true],
        true,
        &[1, 2],
        &[(0, 1), (1, 1)]
    )]
    #[case::null_list_holding_elements(
        &[9, 1, 9, 2],
        &[(1, 1), (2, 1), (3, 1)],
        &[true, false, true],
        true,
        &[1, 2],
        &[(0, 1), (1, 0), (1, 1)]
    )]
    #[case::gap_between_lists(&[1, 9, 2], &[(0, 1), (2, 1)], &[true, true], false, &[1, 2], &[(0, 1), (1, 1)])]
    #[case::lists_out_of_order(
        &[1, 2, 3, 9],
        &[(2, 1), (0, 2)],
        &[true, true],
        false,
        &[3, 1, 2],
        &[(0, 1), (1, 2)]
    )]
    #[case::null_list_holding_elements_not_zero_copy(
        &[9, 1, 9, 2],
        &[(1, 1), (2, 1), (3, 1)],
        &[true, false, true],
        false,
        &[1, 2],
        &[(0, 1), (1, 0), (1, 1)]
    )]
    #[case::empty_null_list_with_leftovers_at_the_ends(
        &[9, 1, 2, 9],
        &[(1, 1), (2, 0), (2, 1)],
        &[true, false, true],
        true,
        &[1, 2],
        &[(0, 1), (1, 0), (1, 1)]
    )]
    #[case::every_list_null(
        &[9, 9],
        &[(0, 1), (1, 1)],
        &[false, false],
        true,
        &[],
        &[(0, 0), (0, 0)]
    )]
    #[case::empty_lists_with_leftovers(
        &[9, 9],
        &[(1, 0), (1, 0)],
        &[true, true],
        true,
        &[],
        &[(0, 0), (0, 0)]
    )]
    #[case::overlapping_lists(
        &[1, 2, 3, 9],
        &[(0, 3), (1, 2)],
        &[true, true],
        false,
        &[1, 2, 3, 2, 3],
        &[(0, 3), (3, 2)]
    )]
    fn test_rebuild_make_exact_with_empty_nulls(
        #[case] elements: &[i32],
        #[case] views: &[(u32, u32)],
        #[case] validity: &[bool],
        #[case] zero_copy_to_list: bool,
        #[case] expected_elements: &[i32],
        #[case] expected_views: &[(u32, u32)],
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let view = ListViewArray::try_new(
            PrimitiveArray::from_iter(elements.iter().copied()).into_array(),
            PrimitiveArray::from_iter(views.iter().map(|(offset, _)| *offset)).into_array(),
            PrimitiveArray::from_iter(views.iter().map(|(_, size)| *size)).into_array(),
            Validity::from_iter(validity.iter().copied()),
            &mut ctx,
        )?;
        // SAFETY: the cases marked zero-copy to a list have sorted lists with nothing between them.
        let view = unsafe { view.with_zero_copy_to_list(zero_copy_to_list) };
        let rebuilt = view.rebuild(ListViewRebuildMode::MakeExactWithEmptyNulls, &mut ctx)?;
        assert!(rebuilt.is_zero_copy_to_list());
        assert_arrays_eq!(
            rebuilt.elements().clone(),
            PrimitiveArray::from_iter(expected_elements.iter().copied()),
            &mut ctx
        );
        assert_arrays_eq!(
            rebuilt.offsets().clone(),
            PrimitiveArray::from_iter(expected_views.iter().map(|(offset, _)| *offset)),
            &mut ctx
        );
        assert_arrays_eq!(
            rebuilt.sizes().clone(),
            PrimitiveArray::from_iter(expected_views.iter().map(|(_, size)| *size)),
            &mut ctx
        );
        assert_arrays_eq!(rebuilt, view, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::non_nullable(Validity::NonNullable)]
    #[case::empty_null_list(Validity::from_iter([true, false, true]))]
    fn test_rebuild_make_exact_with_empty_nulls_returns_an_exact_list_view_as_is(
        #[case] validity: Validity,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let view = ListViewArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            PrimitiveArray::from_iter([0u32, 1, 1]).into_array(),
            PrimitiveArray::from_iter([1u32, 0, 2]).into_array(),
            validity,
            &mut ctx,
        )?;
        // SAFETY: the lists are sorted with nothing between them.
        let view = unsafe { view.with_zero_copy_to_list(true) };
        let rebuilt = view.rebuild(ListViewRebuildMode::MakeExactWithEmptyNulls, &mut ctx)?;
        assert!(ArrayRef::ptr_eq(rebuilt.elements(), view.elements()));
        assert!(ArrayRef::ptr_eq(rebuilt.offsets(), view.offsets()));
        assert!(ArrayRef::ptr_eq(rebuilt.sizes(), view.sizes()));
        Ok(())
    }

    /// `values` as a primitive array of `ptype`.
    fn primitive_of(
        values: impl IntoIterator<Item = u32>,
        ptype: PType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::from_iter(values)
            .into_array()
            .cast(DType::Primitive(ptype, Nullability::NonNullable))?
            .execute::<PrimitiveArray>(ctx)?
            .into_array())
    }

    /// Random lists with elements before, after and under null lists, rebuilt as a `ListArray`
    /// and as list views with and without the zero-copy flag, against a reference that gathers
    /// each valid list's elements back to back.
    #[test]
    fn rebuild_with_referenced_valid_elements_matches_a_reference() -> VortexResult<()> {
        const PTYPES: [PType; 5] = [PType::U8, PType::U16, PType::U64, PType::I32, PType::I64];
        let u32_dtype = DType::Primitive(PType::U32, Nullability::NonNullable);
        let mut ctx = SESSION.create_execution_ctx();
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..500 {
            let len = rng.random_range(0..10usize);
            let mut offset = rng.random_range(0..3u32);
            let mut offsets = vec![offset];
            for _ in 0..len {
                offset += rng.random_range(0..3u32);
                offsets.push(offset);
            }
            let elements_len = offset + rng.random_range(0..3u32);
            let valid: Vec<bool> = (0..len).map(|_| rng.random_bool(0.6)).collect();
            let validity = match rng.random_range(0..3u8) {
                0 if valid.iter().all(|valid| *valid) => Validity::NonNullable,
                // Validity that starts mid-byte.
                1 => Validity::Array(
                    BoolArray::from_iter([false].into_iter().chain(valid.iter().copied()))
                        .into_array()
                        .slice(1..len + 1)?,
                ),
                _ => Validity::from_iter(valid.iter().copied()),
            };
            let ptype = PTYPES[rng.random_range(0..PTYPES.len())];

            let mut expected_elements = Vec::new();
            let mut expected_offsets = vec![0u32];
            let mut kept = 0u32;
            for (bounds, is_valid) in offsets.windows(2).zip(&valid) {
                if *is_valid {
                    expected_elements.extend(bounds[0]..bounds[1]);
                    kept += bounds[1] - bounds[0];
                }
                expected_offsets.push(kept);
            }
            let expected_elements = PrimitiveArray::from_iter(expected_elements);
            let expected_sizes =
                PrimitiveArray::from_iter(expected_offsets.windows(2).map(|w| w[1] - w[0]));

            let elements = PrimitiveArray::from_iter(0..elements_len).into_array();
            let list = ListArray::try_new(
                elements.clone(),
                primitive_of(offsets.iter().copied(), ptype, &mut ctx)?,
                validity.clone(),
                &mut ctx,
            )?;
            let rebuilt = list.rebuild_with_referenced_valid_elements(&mut ctx)?;
            assert_arrays_eq!(
                rebuilt.elements().clone(),
                expected_elements.clone(),
                &mut ctx
            );
            assert_eq!(rebuilt.offsets().dtype(), list.offsets().dtype());
            assert_arrays_eq!(
                rebuilt.offsets().cast(u32_dtype.clone())?,
                PrimitiveArray::from_iter(expected_offsets.iter().copied()),
                &mut ctx
            );
            assert_arrays_eq!(rebuilt, list, &mut ctx);

            for zero_copy_to_list in [true, false] {
                let view = ListViewArray::try_new(
                    elements.clone(),
                    primitive_of(offsets[..len].iter().copied(), ptype, &mut ctx)?,
                    primitive_of(offsets.windows(2).map(|w| w[1] - w[0]), ptype, &mut ctx)?,
                    validity.clone(),
                    &mut ctx,
                )?;
                // SAFETY: the views are a `ListArray`'s lists, sorted with nothing between them.
                let view = unsafe { view.with_zero_copy_to_list(zero_copy_to_list) };
                let rebuilt =
                    view.rebuild(ListViewRebuildMode::MakeExactWithEmptyNulls, &mut ctx)?;
                assert!(rebuilt.is_zero_copy_to_list());
                assert_arrays_eq!(
                    rebuilt.elements().clone(),
                    expected_elements.clone(),
                    &mut ctx
                );
                // The copy widens the offsets, so compare their values.
                assert_arrays_eq!(
                    rebuilt.offsets().cast(u32_dtype.clone())?,
                    PrimitiveArray::from_iter(expected_offsets[..len].iter().copied()),
                    &mut ctx
                );
                assert_arrays_eq!(
                    rebuilt.sizes().cast(u32_dtype.clone())?,
                    expected_sizes.clone(),
                    &mut ctx
                );
                assert_arrays_eq!(rebuilt, view, &mut ctx);
            }
        }
        Ok(())
    }

    #[test]
    fn test_rebuild_flatten_removes_overlaps() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a list view with overlapping lists: [A, B, C]
        // List 0: offset=0, size=3 -> [A, B, C]
        // List 1: offset=1, size=2 -> [B, C] (overlaps with List 0)
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 1]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![3u32, 2]).into_array();

        let listview =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable, &mut ctx);

        let mut ctx = SESSION.create_execution_ctx();
        let flattened = listview.rebuild(ListViewRebuildMode::MakeZeroCopyToList, &mut ctx)?;

        // After flatten: elements should be [A, B, C, B, C] = [1, 2, 3, 2, 3]
        // Lists should be sequential with no overlaps
        assert_eq!(flattened.elements().len(), 5);

        // Offsets should be sequential
        assert_eq!(flattened.offset_at(0, &mut ctx), 0);
        assert_eq!(flattened.size_at(0, &mut ctx), 3);
        assert_eq!(flattened.offset_at(1, &mut ctx), 3);
        assert_eq!(flattened.size_at(1, &mut ctx), 2);

        // Verify the data is correct
        assert_arrays_eq!(
            flattened.list_elements_at(0, &mut ctx)?,
            PrimitiveArray::from_iter([1i32, 2, 3]),
            &mut ctx
        );

        assert_arrays_eq!(
            flattened.list_elements_at(1, &mut ctx)?,
            PrimitiveArray::from_iter([2i32, 3]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_rebuild_flatten_with_nullable() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
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

        let listview = ListViewArray::new(elements, offsets, sizes, validity, &mut ctx);

        let mut ctx = SESSION.create_execution_ctx();
        let flattened = listview.rebuild(ListViewRebuildMode::MakeZeroCopyToList, &mut ctx)?;

        // Verify nullability is preserved
        assert_eq!(flattened.dtype().nullability(), Nullability::Nullable);
        assert!(flattened.validity()?.execute_is_valid(0, &mut ctx)?);
        assert!(!flattened.validity()?.execute_is_valid(1, &mut ctx)?);
        assert!(flattened.validity()?.execute_is_valid(2, &mut ctx)?);

        // Verify valid lists contain correct data
        assert_arrays_eq!(
            flattened.list_elements_at(0, &mut ctx)?,
            PrimitiveArray::from_iter([1i32, 2]),
            &mut ctx
        );

        assert_arrays_eq!(
            flattened.list_elements_at(2, &mut ctx)?,
            PrimitiveArray::from_iter([3i32]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_rebuild_flatten_null_row_uses_valid_empty_range() -> VortexResult<()> {
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3, 4]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 1, 2]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![1u32, 2, 2]).into_array();
        let validity = Validity::from_iter([true, false, true]);

        // SAFETY: all source ranges are valid, including the null row's non-empty range.
        let listview = unsafe { ListViewArray::new_unchecked(elements, offsets, sizes, validity) };

        let mut ctx = SESSION.create_execution_ctx();
        let flattened = listview.rebuild(ListViewRebuildMode::MakeZeroCopyToList, &mut ctx)?;

        assert_eq!(flattened.offset_at(0, &mut ctx), 0);
        assert_eq!(flattened.size_at(0, &mut ctx), 1);
        assert_eq!(flattened.offset_at(1, &mut ctx), 1);
        assert_eq!(flattened.size_at(1, &mut ctx), 0);
        assert_eq!(flattened.offset_at(2, &mut ctx), 1);
        assert_eq!(flattened.size_at(2, &mut ctx), 2);
        assert!(flattened.validity()?.execute_is_valid(0, &mut ctx)?);
        assert!(!flattened.validity()?.execute_is_valid(1, &mut ctx)?);
        assert!(flattened.validity()?.execute_is_valid(2, &mut ctx)?);

        assert_arrays_eq!(
            flattened.list_elements_at(0, &mut ctx)?,
            PrimitiveArray::from_iter([1i32]),
            &mut ctx
        );
        assert_arrays_eq!(
            flattened.list_elements_at(2, &mut ctx)?,
            PrimitiveArray::from_iter([3i32, 4]),
            &mut ctx
        );
        Ok(())
    }

    #[rstest]
    #[case::zero_copy_to_list(ListViewRebuildMode::MakeZeroCopyToList)]
    #[case::trim_elements(ListViewRebuildMode::TrimElements)]
    #[case::make_exact(ListViewRebuildMode::MakeExact)]
    #[case::make_exact_with_empty_nulls(ListViewRebuildMode::MakeExactWithEmptyNulls)]
    fn test_rebuild_empty_drops_elements(#[case] mode: ListViewRebuildMode) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // No lists, but three elements left in the buffer.
        let listview = ListViewArray::new(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            PrimitiveArray::from_iter::<[u32; 0]>([]).into_array(),
            PrimitiveArray::from_iter::<[u32; 0]>([]).into_array(),
            Validity::NonNullable,
            &mut ctx,
        );

        let rebuilt = listview.rebuild(mode, &mut ctx)?;
        assert!(rebuilt.is_empty());
        assert_eq!(rebuilt.elements().len(), 0);
        assert!(rebuilt.is_zero_copy_to_list());
        Ok(())
    }

    #[test]
    fn test_rebuild_trim_elements_basic() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
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

        let listview =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable, &mut ctx);

        let mut ctx = SESSION.create_execution_ctx();
        let trimmed = listview.rebuild(ListViewRebuildMode::TrimElements, &mut ctx)?;

        // After trimming: elements should be [A, B, _, C, D] = [1, 2, 97, 3, 4].
        assert_eq!(trimmed.elements().len(), 5);

        // Offsets should be adjusted: old offset 2 -> new offset 0, old offset 5 -> new offset 3.
        assert_eq!(trimmed.offset_at(0, &mut ctx), 0);
        assert_eq!(trimmed.size_at(0, &mut ctx), 2);
        assert_eq!(trimmed.offset_at(1, &mut ctx), 3);
        assert_eq!(trimmed.size_at(1, &mut ctx), 2);

        // Verify the data is correct.
        assert_arrays_eq!(
            trimmed.list_elements_at(0, &mut ctx)?,
            PrimitiveArray::from_iter([1i32, 2]),
            &mut ctx
        );

        assert_arrays_eq!(
            trimmed.list_elements_at(1, &mut ctx)?,
            PrimitiveArray::from_iter([3i32, 4]),
            &mut ctx
        );

        // Note that element at index 2 (97) is preserved as a gap.
        let all_elements = trimmed
            .elements()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(all_elements.execute_scalar(2, &mut ctx)?, 97i32.into());
        Ok(())
    }

    #[test]
    fn test_rebuild_flatten_large_lists_with_piecewise_indices() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = PrimitiveArray::from_iter(0i32..300).into_array();
        let offsets = PrimitiveArray::from_iter(vec![10u32, 0]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![150u32, 130]).into_array();
        let listview =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable, &mut ctx);

        let flattened = listview.rebuild(ListViewRebuildMode::MakeZeroCopyToList, &mut ctx)?;

        assert_eq!(flattened.elements().len(), 280);
        assert_eq!(flattened.offset_at(0, &mut ctx), 0);
        assert_eq!(flattened.size_at(0, &mut ctx), 150);
        assert_eq!(flattened.offset_at(1, &mut ctx), 150);
        assert_eq!(flattened.size_at(1, &mut ctx), 130);

        assert_arrays_eq!(
            flattened.list_elements_at(0, &mut ctx)?,
            PrimitiveArray::from_iter(10i32..160),
            &mut ctx
        );
        assert_arrays_eq!(
            flattened.list_elements_at(1, &mut ctx)?,
            PrimitiveArray::from_iter(0i32..130),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_rebuild_with_trailing_nulls_regression() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Regression test for issue #5412
        // Tests that zero-copy-to-list arrays with trailing NULLs correctly calculate
        // offsets for NULL items to maintain no-overlap invariant

        // Create a ListViewArray with trailing NULLs
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3, 4]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0u32, 2, 0, 0]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u32, 2, 0, 0]).into_array();
        let validity = Validity::from_iter(vec![true, true, false, false]);

        let listview = ListViewArray::new(elements, offsets, sizes, validity, &mut ctx);

        // First rebuild to make it zero-copy-to-list
        let mut ctx = SESSION.create_execution_ctx();
        let rebuilt = listview.rebuild(ListViewRebuildMode::MakeZeroCopyToList, &mut ctx)?;
        assert!(rebuilt.is_zero_copy_to_list());

        // Verify NULL items have correct offsets (should not reuse previous offsets)
        // After rebuild: offsets should be [0, 2, 4, 4] for zero-copy-to-list
        assert_eq!(rebuilt.offset_at(0, &mut ctx), 0);
        assert_eq!(rebuilt.offset_at(1, &mut ctx), 2);
        assert_eq!(rebuilt.offset_at(2, &mut ctx), 4); // NULL should be at position 4
        assert_eq!(rebuilt.offset_at(3, &mut ctx), 4); // Second NULL also at position 4

        // All sizes should be correct
        assert_eq!(rebuilt.size_at(0, &mut ctx), 2);
        assert_eq!(rebuilt.size_at(1, &mut ctx), 2);
        assert_eq!(rebuilt.size_at(2, &mut ctx), 0); // NULL has size 0
        assert_eq!(rebuilt.size_at(3, &mut ctx), 0); // NULL has size 0

        // Now rebuild with MakeExact (which calls naive_rebuild then trim_elements)
        // This should not panic (issue #5412)
        let exact = rebuilt.rebuild(ListViewRebuildMode::MakeExact, &mut ctx)?;

        // Verify the result is still valid
        assert!(exact.is_valid(0, &mut ctx)?);
        assert!(exact.is_valid(1, &mut ctx)?);
        assert!(!exact.is_valid(2, &mut ctx)?);
        assert!(!exact.is_valid(3, &mut ctx)?);

        // Verify data is preserved
        assert_arrays_eq!(
            exact.list_elements_at(0, &mut ctx)?,
            PrimitiveArray::from_iter([1i32, 2]),
            &mut ctx
        );

        assert_arrays_eq!(
            exact.list_elements_at(1, &mut ctx)?,
            PrimitiveArray::from_iter([3i32, 4]),
            &mut ctx
        );
        Ok(())
    }

    /// Regression test for <https://github.com/vortex-data/vortex/issues/6773>.
    /// u32 offsets exceed u16::MAX, so u16 sizes are widened to u32 for the add.
    #[test]
    fn test_rebuild_trim_elements_offsets_wider_than_sizes() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut elems = vec![0i32; 70_005];
        elems[70_000] = 10;
        elems[70_001] = 20;
        elems[70_002] = 30;
        elems[70_003] = 40;
        let elements = PrimitiveArray::from_iter(elems).into_array();
        let offsets = PrimitiveArray::from_iter(vec![70_000u32, 70_002]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![2u16, 2]).into_array();

        let listview =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable, &mut ctx);
        let mut ctx = SESSION.create_execution_ctx();
        let trimmed = listview.rebuild(ListViewRebuildMode::TrimElements, &mut ctx)?;
        assert_arrays_eq!(
            trimmed.list_elements_at(1, &mut ctx)?,
            PrimitiveArray::from_iter([30i32, 40]),
            &mut ctx
        );
        Ok(())
    }

    /// Regression test for <https://github.com/vortex-data/vortex/issues/6773>.
    /// u32 sizes exceed u16::MAX, so u16 offsets are widened to u32 for the add.
    #[test]
    fn test_rebuild_trim_elements_sizes_wider_than_offsets() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut elems = vec![0i32; 70_001];
        elems[3] = 30;
        elems[4] = 40;
        let elements = PrimitiveArray::from_iter(elems).into_array();
        let offsets = PrimitiveArray::from_iter(vec![1u16, 3]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![70_000u32, 2]).into_array();

        let listview =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable, &mut ctx);
        let mut ctx = SESSION.create_execution_ctx();
        let trimmed = listview.rebuild(ListViewRebuildMode::TrimElements, &mut ctx)?;
        assert_arrays_eq!(
            trimmed.list_elements_at(1, &mut ctx)?,
            PrimitiveArray::from_iter([30i32, 40]),
            &mut ctx
        );
        Ok(())
    }

    /// Rebuild with signed offsets/sizes: exercises the unsigned-reinterpret read path and asserts
    /// the result offset/size dtypes preserve signedness (widened to >=32-bit for offsets).
    #[test]
    fn test_rebuild_preserves_signed_offset_and_size_types() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        use crate::arrays::listview::ListViewArraySlotsExt;
        use crate::dtype::PType;

        // Overlapping lists force an actual rebuild rather than the zero-copy fast path.
        let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![0i32, 1]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![3i16, 2]).into_array();

        let listview =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable, &mut ctx);
        let mut ctx = SESSION.create_execution_ctx();
        let rebuilt = listview.rebuild(ListViewRebuildMode::MakeZeroCopyToList, &mut ctx)?;

        // Values: [1,2,3] and [2,3].
        assert_arrays_eq!(
            rebuilt.list_elements_at(0, &mut ctx)?,
            PrimitiveArray::from_iter([1i32, 2, 3]),
            &mut ctx
        );
        assert_arrays_eq!(
            rebuilt.list_elements_at(1, &mut ctx)?,
            PrimitiveArray::from_iter([2i32, 3]),
            &mut ctx
        );

        // Signed input -> signed result (offsets widened to i32, sizes kept i16).
        assert_eq!(rebuilt.offsets().dtype().as_ptype(), PType::I32);
        assert_eq!(rebuilt.sizes().dtype().as_ptype(), PType::I16);
        Ok(())
    }

    // ── should_use_take heuristic tests ────────────────────────────────────

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
        let mut ctx = SESSION.create_execution_ctx();
        let elements = PrimitiveArray::from_iter(vec![0i32; 261]).into_array();
        let offsets = PrimitiveArray::from_iter(vec![215u8, 0]).into_array();
        let sizes = PrimitiveArray::from_iter(vec![46u8, 10]).into_array();

        let listview =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable, &mut ctx);
        let mut ctx = SESSION.create_execution_ctx();
        let trimmed = listview.rebuild(ListViewRebuildMode::TrimElements, &mut ctx)?;

        // min(offsets) = 0, so nothing to trim; output should equal input.
        assert_arrays_eq!(trimmed, listview, &mut ctx);
        Ok(())
    }
}
