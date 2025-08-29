// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Range;

use arrow_buffer::BooleanBuffer;
use itertools::Itertools as _;
use num_traits::{NumCast, ToPrimitive};
use serde::{Deserialize, Serialize};
use vortex_buffer::BufferMut;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{
    DType, NativePType, PType, match_each_integer_ptype, match_each_unsigned_integer_ptype,
};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_mask::{AllOr, Mask};
use vortex_scalar::{PValue, Scalar};
use vortex_utils::aliases::hash_map::HashMap;

use crate::arrays::PrimitiveArray;
use crate::compute::{cast, filter, take};
use crate::search_sorted::{SearchResult, SearchSorted, SearchSortedSide};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

#[derive(Copy, Clone, Serialize, Deserialize, prost::Message)]
pub struct PatchesMetadata {
    #[prost(uint64, tag = "1")]
    len: u64,
    #[prost(uint64, tag = "2")]
    offset: u64,
    #[prost(enumeration = "PType", tag = "3")]
    indices_ptype: i32,
}

impl PatchesMetadata {
    pub fn new(len: usize, offset: usize, indices_ptype: PType) -> Self {
        Self {
            len: len as u64,
            offset: offset as u64,
            indices_ptype: indices_ptype as i32,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        usize::try_from(self.len).vortex_expect("len is a valid usize")
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn offset(&self) -> usize {
        usize::try_from(self.offset).vortex_expect("offset is a valid usize")
    }

    #[inline]
    pub fn indices_dtype(&self) -> DType {
        assert!(
            self.indices_ptype().is_unsigned_int(),
            "Patch indices must be unsigned integers"
        );
        DType::Primitive(self.indices_ptype(), NonNullable)
    }
}

/// A helper for working with patched arrays.
#[derive(Debug, Clone)]
pub struct Patches {
    array_len: usize,
    offset: usize,
    indices: ArrayRef,
    values: ArrayRef,
}

impl Patches {
    pub fn new(array_len: usize, offset: usize, indices: ArrayRef, values: ArrayRef) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "Patch indices and values must have the same length"
        );
        assert!(
            indices.dtype().is_unsigned_int() && !indices.dtype().is_nullable(),
            "Patch indices must be non-nullable unsigned integers, got {:?}",
            indices.dtype()
        );
        assert!(
            indices.len() <= array_len,
            "Patch indices must be shorter than the array length"
        );
        assert!(!indices.is_empty(), "Patch indices must not be empty");
        let max = usize::try_from(&indices.scalar_at(indices.len() - 1))
            .vortex_expect("indices must be a number");
        assert!(
            max - offset < array_len,
            "Patch indices {max:?}, offset {offset} are longer than the array length {array_len}"
        );

        Self {
            array_len,
            offset,
            indices,
            values,
        }
    }

    /// Construct new patches without validating any of the arguments
    ///
    /// # Safety
    ///
    /// Users have to assert that
    /// * Indices and values have the same length
    /// * Indices is an unsigned integer type
    /// * Indices must be sorted
    /// * Last value in indices is smaller than array_len
    pub unsafe fn new_unchecked(
        array_len: usize,
        offset: usize,
        indices: ArrayRef,
        values: ArrayRef,
    ) -> Self {
        Self {
            array_len,
            offset,
            indices,
            values,
        }
    }

    pub fn array_len(&self) -> usize {
        self.array_len
    }

    pub fn num_patches(&self) -> usize {
        self.indices.len()
    }

    pub fn dtype(&self) -> &DType {
        self.values.dtype()
    }

    pub fn indices(&self) -> &ArrayRef {
        &self.indices
    }

    pub fn into_indices(self) -> ArrayRef {
        self.indices
    }

    pub fn indices_mut(&mut self) -> &mut ArrayRef {
        &mut self.indices
    }

    pub fn values(&self) -> &ArrayRef {
        &self.values
    }

    pub fn into_values(self) -> ArrayRef {
        self.values
    }

    pub fn values_mut(&mut self) -> &mut ArrayRef {
        &mut self.values
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn indices_ptype(&self) -> PType {
        PType::try_from(self.indices.dtype()).vortex_expect("primitive indices")
    }

    pub fn to_metadata(&self, len: usize, dtype: &DType) -> VortexResult<PatchesMetadata> {
        if self.indices.len() > len {
            vortex_bail!(
                "Patch indices {} are longer than the array length {}",
                self.indices.len(),
                len
            );
        }
        if self.values.dtype() != dtype {
            vortex_bail!(
                "Patch values dtype {} does not match array dtype {}",
                self.values.dtype(),
                dtype
            );
        }
        Ok(PatchesMetadata {
            len: self.indices.len() as u64,
            offset: self.offset as u64,
            indices_ptype: PType::try_from(self.indices.dtype()).vortex_expect("primitive indices")
                as i32,
        })
    }

    pub fn cast_values(self, values_dtype: &DType) -> VortexResult<Self> {
        // SAFETY: casting does not affect the relationship between the indices and values
        unsafe {
            Ok(Self::new_unchecked(
                self.array_len,
                self.offset,
                self.indices,
                cast(&self.values, values_dtype)?,
            ))
        }
    }

    /// Get the patched value at a given index if it exists.
    pub fn get_patched(&self, index: usize) -> Option<Scalar> {
        self.search_index(index)
            .to_found()
            .map(|patch_idx| self.values().scalar_at(patch_idx))
    }

    /// Return the insertion point of `index` in the [Self::indices].
    pub fn search_index(&self, index: usize) -> SearchResult {
        if self.indices.is_canonical() {
            let primitive = self.indices.to_primitive();
            match_each_integer_ptype!(primitive.ptype(), |T| {
                let Ok(needle) = T::try_from(index + self.offset) else {
                    // If the needle is not of type T, then it cannot possibly be in this array.
                    //
                    // The needle is a non-negative integer (a usize); therefore, it must be larger
                    // than all values in this array.
                    return SearchResult::NotFound(primitive.len());
                };
                return primitive
                    .as_slice::<T>()
                    .search_sorted(&needle, SearchSortedSide::Left);
            });
        }
        self.indices.as_primitive_typed().search_sorted(
            &PValue::U64((index + self.offset) as u64),
            SearchSortedSide::Left,
        )
    }

    /// Return the search_sorted result for the given target re-mapped into the original indices.
    pub fn search_sorted<T: Into<Scalar>>(
        &self,
        target: T,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        let target = target.into();

        let sr = if self.values().dtype().is_primitive() {
            self.values()
                .as_primitive_typed()
                .search_sorted(&target.as_primitive().pvalue(), side)
        } else {
            self.values().search_sorted(&target, side)
        };

        let index_idx = sr.to_offsets_index(self.indices().len(), side);
        let index = usize::try_from(&self.indices().scalar_at(index_idx))? - self.offset;
        Ok(match sr {
            // If we reached the end of patched values when searching then the result is one after the last patch index
            SearchResult::Found(i) => SearchResult::Found(
                if i == self.indices().len() || side == SearchSortedSide::Right {
                    index + 1
                } else {
                    index
                },
            ),
            // If the result is NotFound we should return index that's one after the nearest not found index for the corresponding value
            SearchResult::NotFound(i) => {
                SearchResult::NotFound(if i == 0 { index } else { index + 1 })
            }
        })
    }

    /// Returns the minimum patch index
    pub fn min_index(&self) -> usize {
        let first = self
            .indices
            .scalar_at(0)
            .as_primitive()
            .as_::<usize>()
            .vortex_expect("non-null");
        first - self.offset
    }

    /// Returns the maximum patch index
    pub fn max_index(&self) -> usize {
        let last = self
            .indices
            .scalar_at(self.indices.len() - 1)
            .as_primitive()
            .as_::<usize>()
            .vortex_expect("non-null");
        last - self.offset
    }

    /// Filter the patches by a mask, resulting in new patches for the filtered array.
    pub fn filter(&self, mask: &Mask) -> VortexResult<Option<Self>> {
        if mask.len() != self.array_len {
            vortex_bail!(
                "Filter mask length {} does not match array length {}",
                mask.len(),
                self.array_len
            );
        }

        match mask.indices() {
            AllOr::All => Ok(Some(self.clone())),
            AllOr::None => Ok(None),
            AllOr::Some(mask_indices) => {
                let flat_indices = self.indices().to_primitive();
                match_each_unsigned_integer_ptype!(flat_indices.ptype(), |I| {
                    filter_patches_with_mask(
                        flat_indices.as_slice::<I>(),
                        self.offset(),
                        self.values(),
                        mask_indices,
                    )
                })
            }
        }
    }

    /// Mask the patches, REMOVING the patches where the mask is true.
    /// Unlike filter, this preserves the patch indices.
    /// Unlike mask on a single array, this does not set masked values to null.
    pub fn mask(&self, mask: &Mask) -> VortexResult<Option<Self>> {
        if mask.len() != self.array_len {
            vortex_bail!(
                "Filter mask length {} does not match array length {}",
                mask.len(),
                self.array_len
            );
        }

        let filter_mask = match mask.boolean_buffer() {
            AllOr::All => return Ok(None),
            AllOr::None => return Ok(Some(self.clone())),
            AllOr::Some(masked) => {
                let patch_indices = self.indices().to_primitive();
                match_each_unsigned_integer_ptype!(patch_indices.ptype(), |P| {
                    let patch_indices = patch_indices.as_slice::<P>();
                    Mask::from_buffer(BooleanBuffer::collect_bool(patch_indices.len(), |i| {
                        #[allow(clippy::cast_possible_truncation)]
                        let idx = (patch_indices[i] as usize) - self.offset;
                        !masked.value(idx)
                    }))
                })
            }
        };

        if filter_mask.all_false() {
            return Ok(None);
        }

        // SAFETY: filtering indices/values with same mask maintains their 1:1 relationship
        unsafe {
            Ok(Some(Self::new_unchecked(
                self.array_len,
                self.offset,
                filter(&self.indices, &filter_mask)?,
                filter(&self.values, &filter_mask)?,
            )))
        }
    }

    /// Slice the patches by a range of the patched array.
    pub fn slice(&self, range: Range<usize>) -> Option<Self> {
        if range.len() == 1 {
            let patch_index = self.search_index(range.start).to_found()?;
            let values = self.values.slice(patch_index..patch_index + 1);
            let indices = self.indices.slice(patch_index..patch_index + 1);
            return Some(Self::new(1, range.start + self.offset(), indices, values));
        }

        let patch_start = self.search_index(range.start).to_index();
        let patch_stop = self.search_index(range.end).to_index();

        if patch_start == patch_stop {
            return None;
        }

        // Slice out the values and indices
        let values = self.values().slice(patch_start..patch_stop);
        let indices = self.indices().slice(patch_start..patch_stop);

        Some(Self::new(
            range.len(),
            range.start + self.offset(),
            indices,
            values,
        ))
    }

    // https://docs.google.com/spreadsheets/d/1D9vBZ1QJ6mwcIvV5wIL0hjGgVchcEnAyhvitqWu2ugU
    const PREFER_MAP_WHEN_PATCHES_OVER_INDICES_LESS_THAN: f64 = 5.0;

    fn is_map_faster_than_search(&self, take_indices: &PrimitiveArray) -> bool {
        (self.num_patches() as f64 / take_indices.len() as f64)
            < Self::PREFER_MAP_WHEN_PATCHES_OVER_INDICES_LESS_THAN
    }

    /// Take the indices from the patches
    ///
    /// Any nulls in take_indices are added to the resulting patches.
    pub fn take_with_nulls(&self, take_indices: &dyn Array) -> VortexResult<Option<Self>> {
        if take_indices.is_empty() {
            return Ok(None);
        }

        let take_indices = take_indices.to_primitive();
        if self.is_map_faster_than_search(&take_indices) {
            self.take_map(take_indices, true)
        } else {
            self.take_search(take_indices, true)
        }
    }

    /// Take the indices from the patches.
    ///
    /// Any nulls in take_indices are ignored.
    pub fn take(&self, take_indices: &dyn Array) -> VortexResult<Option<Self>> {
        if take_indices.is_empty() {
            return Ok(None);
        }

        let take_indices = take_indices.to_primitive();
        if self.is_map_faster_than_search(&take_indices) {
            self.take_map(take_indices, false)
        } else {
            self.take_search(take_indices, false)
        }
    }

    pub fn take_search(
        &self,
        take_indices: PrimitiveArray,
        include_nulls: bool,
    ) -> VortexResult<Option<Self>> {
        let indices = self.indices.to_primitive();
        let new_length = take_indices.len();

        let Some((new_indices, values_indices)) =
            match_each_unsigned_integer_ptype!(indices.ptype(), |Indices| {
                match_each_integer_ptype!(take_indices.ptype(), |TakeIndices| {
                    take_search::<_, TakeIndices>(
                        indices.as_slice::<Indices>(),
                        take_indices,
                        self.offset(),
                        include_nulls,
                    )?
                })
            })
        else {
            return Ok(None);
        };

        Ok(Some(Self::new(
            new_length,
            0,
            new_indices,
            take(self.values(), &values_indices)?,
        )))
    }

    pub fn take_map(
        &self,
        take_indices: PrimitiveArray,
        include_nulls: bool,
    ) -> VortexResult<Option<Self>> {
        let indices = self.indices.to_primitive();
        let new_length = take_indices.len();

        let Some((new_sparse_indices, value_indices)) =
            match_each_unsigned_integer_ptype!(indices.ptype(), |Indices| {
                match_each_integer_ptype!(take_indices.ptype(), |TakeIndices| {
                    take_map::<_, TakeIndices>(
                        indices.as_slice::<Indices>(),
                        take_indices,
                        self.offset(),
                        self.min_index(),
                        self.max_index(),
                        include_nulls,
                    )?
                })
            })
        else {
            return Ok(None);
        };

        Ok(Some(Patches::new(
            new_length,
            0,
            new_sparse_indices,
            take(self.values(), &value_indices)?,
        )))
    }

    pub fn map_values<F>(self, f: F) -> VortexResult<Self>
    where
        F: FnOnce(ArrayRef) -> VortexResult<ArrayRef>,
    {
        let values = f(self.values)?;
        if self.indices.len() != values.len() {
            vortex_bail!(
                "map_values must preserve length: expected {} received {}",
                self.indices.len(),
                values.len()
            )
        }
        Ok(Self::new(self.array_len, self.offset, self.indices, values))
    }
}

fn take_search<I: NativePType + NumCast + PartialOrd, T: NativePType + NumCast>(
    indices: &[I],
    take_indices: PrimitiveArray,
    indices_offset: usize,
    include_nulls: bool,
) -> VortexResult<Option<(ArrayRef, ArrayRef)>>
where
    usize: TryFrom<T>,
    VortexError: From<<usize as TryFrom<T>>::Error>,
{
    let take_indices_validity = take_indices.validity();
    let indices_offset = I::from(indices_offset).vortex_expect("indices_offset out of range");

    let (values_indices, new_indices): (BufferMut<u64>, BufferMut<u64>) = take_indices
        .as_slice::<T>()
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| {
            I::from(v)
                .and_then(|v| {
                    // If we have to take nulls the take index doesn't matter, make it 0 for consistency
                    if include_nulls && take_indices_validity.is_null(i) {
                        Some(0)
                    } else {
                        indices
                            .search_sorted(&(v + indices_offset), SearchSortedSide::Left)
                            .to_found()
                            .map(|patch_idx| patch_idx as u64)
                    }
                })
                .map(|patch_idx| (patch_idx, i as u64))
        })
        .unzip();

    if new_indices.is_empty() {
        return Ok(None);
    }

    let new_indices = new_indices.into_array();
    let values_validity = take_indices_validity.take(&new_indices)?;
    Ok(Some((
        new_indices,
        PrimitiveArray::new(values_indices, values_validity).into_array(),
    )))
}

fn take_map<I: NativePType + Hash + Eq + TryFrom<usize>, T: NativePType>(
    indices: &[I],
    take_indices: PrimitiveArray,
    indices_offset: usize,
    min_index: usize,
    max_index: usize,
    include_nulls: bool,
) -> VortexResult<Option<(ArrayRef, ArrayRef)>>
where
    usize: TryFrom<T>,
    VortexError: From<<I as TryFrom<usize>>::Error>,
{
    let take_indices_validity = take_indices.validity();
    let take_indices = take_indices.as_slice::<T>();
    let offset_i = I::try_from(indices_offset)?;

    let sparse_index_to_value_index: HashMap<I, usize> = indices
        .iter()
        .copied()
        .map(|idx| idx - offset_i)
        .enumerate()
        .map(|(value_index, sparse_index)| (sparse_index, value_index))
        .collect();

    let (new_sparse_indices, value_indices): (BufferMut<u64>, BufferMut<u64>) = take_indices
        .iter()
        .copied()
        .map(usize::try_from)
        .process_results(|iter| {
            iter.enumerate()
                .filter_map(|(idx_in_take, ti)| {
                    // If we have to take nulls the take index doesn't matter, make it 0 for consistency
                    if include_nulls && take_indices_validity.is_null(idx_in_take) {
                        Some((idx_in_take as u64, 0))
                    } else if ti < min_index || ti > max_index {
                        None
                    } else {
                        sparse_index_to_value_index
                            .get(
                                &I::try_from(ti)
                                    .vortex_expect("take index is between min and max index"),
                            )
                            .map(|value_index| (idx_in_take as u64, *value_index as u64))
                    }
                })
                .unzip()
        })
        .map_err(|_| vortex_err!("Failed to convert index to usize"))?;

    if new_sparse_indices.is_empty() {
        return Ok(None);
    }

    let new_sparse_indices = new_sparse_indices.into_array();
    let values_validity = take_indices_validity.take(&new_sparse_indices)?;
    Ok(Some((
        new_sparse_indices,
        PrimitiveArray::new(value_indices, values_validity).into_array(),
    )))
}

/// Filter patches with the provided mask (in flattened space).
///
/// The filter mask may contain indices that are non-patched. The return value of this function
/// is a new set of `Patches` with the indices relative to the provided `mask` rank, and the
/// patch values.
fn filter_patches_with_mask<T: ToPrimitive + Copy + Ord>(
    patch_indices: &[T],
    offset: usize,
    patch_values: &dyn Array,
    mask_indices: &[usize],
) -> VortexResult<Option<Patches>> {
    let true_count = mask_indices.len();
    let mut new_patch_indices = BufferMut::<u64>::with_capacity(true_count);
    let mut new_mask_indices = Vec::with_capacity(true_count);

    // Attempt to move the window by `STRIDE` elements on each iteration. This assumes that
    // the patches are relatively sparse compared to the overall mask, and so many indices in the
    // mask will end up being skipped.
    const STRIDE: usize = 4;

    let mut mask_idx = 0usize;
    let mut true_idx = 0usize;

    while mask_idx < patch_indices.len() && true_idx < true_count {
        // NOTE: we are searching for overlaps between sorted, unaligned indices in `patch_indices`
        //  and `mask_indices`. We assume that Patches are sparse relative to the global space of
        //  the mask (which covers both patch and non-patch values of the parent array), and so to
        //  quickly jump through regions with no overlap, we attempt to move our pointers by STRIDE
        //  elements on each iteration. If we cannot rule out overlap due to min/max values, we
        //  fallback to performing a two-way iterator merge.
        if (mask_idx + STRIDE) < patch_indices.len() && (true_idx + STRIDE) < mask_indices.len() {
            // Load a vector of each into our registers.
            let left_min = patch_indices[mask_idx].to_usize().vortex_expect("left_min") - offset;
            let left_max = patch_indices[mask_idx + STRIDE]
                .to_usize()
                .vortex_expect("left_max")
                - offset;
            let right_min = mask_indices[true_idx];
            let right_max = mask_indices[true_idx + STRIDE];

            if left_min > right_max {
                // Advance right side
                true_idx += STRIDE;
                continue;
            } else if right_min > left_max {
                mask_idx += STRIDE;
                continue;
            } else {
                // Fallthrough to direct comparison path.
            }
        }

        // Two-way sorted iterator merge:

        let left = patch_indices[mask_idx].to_usize().vortex_expect("left") - offset;
        let right = mask_indices[true_idx];

        match left.cmp(&right) {
            Ordering::Less => {
                mask_idx += 1;
            }
            Ordering::Greater => {
                true_idx += 1;
            }
            Ordering::Equal => {
                // Save the mask index as well as the positional index.
                new_mask_indices.push(mask_idx);
                new_patch_indices.push(true_idx as u64);

                mask_idx += 1;
                true_idx += 1;
            }
        }
    }

    if new_mask_indices.is_empty() {
        return Ok(None);
    }

    let new_patch_indices = new_patch_indices.into_array();
    let new_patch_values = filter(
        patch_values,
        &Mask::from_indices(patch_values.len(), new_mask_indices),
    )?;

    Ok(Some(Patches::new(
        true_count,
        0,
        new_patch_indices,
        new_patch_values,
    )))
}

#[cfg(test)]
mod test {
    use rstest::{fixture, rstest};
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::arrays::PrimitiveArray;
    use crate::patches::Patches;
    use crate::search_sorted::{SearchResult, SearchSortedSide};
    use crate::validity::Validity;
    use crate::{IntoArray, ToCanonical};

    #[test]
    fn test_filter() {
        let patches = Patches::new(
            100,
            0,
            buffer![10u32, 11, 20].into_array(),
            buffer![100, 110, 200].into_array(),
        );

        let filtered = patches
            .filter(&Mask::from_indices(100, vec![10, 20, 30]))
            .unwrap()
            .unwrap();

        let indices = filtered.indices().to_primitive();
        let values = filtered.values().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[0, 1]);
        assert_eq!(values.as_slice::<i32>(), &[100, 200]);
    }

    #[fixture]
    fn patches() -> Patches {
        Patches::new(
            20,
            0,
            buffer![2u64, 9, 15].into_array(),
            PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
        )
    }

    #[rstest]
    fn search_larger_than(patches: Patches) {
        let res = patches.search_sorted(66, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::NotFound(16));
    }

    #[rstest]
    fn search_less_than(patches: Patches) {
        let res = patches.search_sorted(22, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::NotFound(2));
    }

    #[rstest]
    fn search_found(patches: Patches) {
        let res = patches.search_sorted(44, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::Found(9));
    }

    #[rstest]
    fn search_not_found_right(patches: Patches) {
        let res = patches.search_sorted(56, SearchSortedSide::Right).unwrap();
        assert_eq!(res, SearchResult::NotFound(16));
    }

    #[rstest]
    fn search_sliced(patches: Patches) {
        let sliced = patches.slice(7..20).unwrap();
        assert_eq!(
            sliced.search_sorted(22, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(2)
        );
    }

    #[test]
    fn search_right() {
        let patches = Patches::new(
            6,
            0,
            buffer![0u8, 1, 4, 5].into_array(),
            buffer![-128i8, -98, 8, 50].into_array(),
        );

        assert_eq!(
            patches.search_sorted(-98, SearchSortedSide::Right).unwrap(),
            SearchResult::Found(2)
        );
        assert_eq!(
            patches.search_sorted(50, SearchSortedSide::Right).unwrap(),
            SearchResult::Found(6),
        );
        assert_eq!(
            patches.search_sorted(7, SearchSortedSide::Right).unwrap(),
            SearchResult::NotFound(2),
        );
        assert_eq!(
            patches.search_sorted(51, SearchSortedSide::Right).unwrap(),
            SearchResult::NotFound(6)
        );
    }

    #[test]
    fn search_left() {
        let patches = Patches::new(
            20,
            0,
            buffer![0u64, 1, 17, 18, 19].into_array(),
            buffer![11i32, 22, 33, 44, 55].into_array(),
        );
        assert_eq!(
            patches.search_sorted(30, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(2)
        );
        assert_eq!(
            patches.search_sorted(54, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(19)
        );
    }

    #[rstest]
    fn take_with_nulls(patches: Patches) {
        let taken = patches
            .take(
                &PrimitiveArray::new(buffer![9, 0], Validity::from_iter(vec![true, false]))
                    .into_array(),
            )
            .unwrap()
            .unwrap();
        let primitive_values = taken.values().to_primitive();
        assert_eq!(taken.array_len(), 2);
        assert_eq!(primitive_values.as_slice::<i32>(), [44]);
        assert_eq!(
            primitive_values.validity_mask(),
            Mask::from_iter(vec![true])
        );
    }

    #[test]
    fn test_slice() {
        let values = buffer![15_u32, 135, 13531, 42].into_array();
        let indices = buffer![10_u64, 11, 50, 100].into_array();

        let patches = Patches::new(101, 0, indices, values);

        let sliced = patches.slice(15..100).unwrap();
        assert_eq!(sliced.array_len(), 100 - 15);
        let primitive = sliced.values().to_primitive();

        assert_eq!(primitive.as_slice::<u32>(), &[13531]);
    }

    #[test]
    fn doubly_sliced() {
        let values = buffer![15_u32, 135, 13531, 42].into_array();
        let indices = buffer![10_u64, 11, 50, 100].into_array();

        let patches = Patches::new(101, 0, indices, values);

        let sliced = patches.slice(15..100).unwrap();
        assert_eq!(sliced.array_len(), 100 - 15);
        let primitive = sliced.values().to_primitive();

        assert_eq!(primitive.as_slice::<u32>(), &[13531]);

        let doubly_sliced = sliced.slice(35..36).unwrap();
        let primitive_doubly_sliced = doubly_sliced.values().to_primitive();

        assert_eq!(primitive_doubly_sliced.as_slice::<u32>(), &[13531]);
    }

    #[test]
    fn test_mask_all_true() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        let mask = Mask::new_true(10);
        let masked = patches.mask(&mask).unwrap();
        assert!(masked.is_none());
    }

    #[test]
    fn test_mask_all_false() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        let mask = Mask::new_false(10);
        let masked = patches.mask(&mask).unwrap().unwrap();

        // No patch values should be masked
        let masked_values = masked.values().to_primitive();
        assert_eq!(masked_values.as_slice::<i32>(), &[100, 200, 300]);
        assert!(masked_values.is_valid(0));
        assert!(masked_values.is_valid(1));
        assert!(masked_values.is_valid(2));

        // Indices should remain unchanged
        let indices = masked.indices().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[2, 5, 8]);
    }

    #[test]
    fn test_mask_partial() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        // Mask that removes patches at indices 2 and 8 (but not 5)
        let mask = Mask::from_iter([
            false, false, true, false, false, false, false, false, true, false,
        ]);
        let masked = patches.mask(&mask).unwrap().unwrap();

        // Only the patch at index 5 should remain
        let masked_values = masked.values().to_primitive();
        assert_eq!(masked_values.len(), 1);
        assert_eq!(masked_values.as_slice::<i32>(), &[200]);

        // Only index 5 should remain
        let indices = masked.indices().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[5]);
    }

    #[test]
    fn test_mask_with_offset() {
        let patches = Patches::new(
            10,
            5,                                  // offset
            buffer![7u64, 10, 13].into_array(), // actual indices are 2, 5, 8
            buffer![100i32, 200, 300].into_array(),
        );

        // Mask that sets actual index 2 to null
        let mask = Mask::from_iter([
            false, false, true, false, false, false, false, false, false, false,
        ]);

        let masked = patches.mask(&mask).unwrap().unwrap();
        assert_eq!(masked.array_len(), 10);
        assert_eq!(masked.offset(), 5);
        let indices = masked.indices().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[10, 13]);
        let masked_values = masked.values().to_primitive();
        assert_eq!(masked_values.as_slice::<i32>(), &[200, 300]);
    }

    #[test]
    fn test_mask_nullable_values() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            PrimitiveArray::from_option_iter([Some(100i32), None, Some(300)]).into_array(),
        );

        // Test masking removes patch at index 2
        let mask = Mask::from_iter([
            false, false, true, false, false, false, false, false, false, false,
        ]);
        let masked = patches.mask(&mask).unwrap().unwrap();

        // Patches at indices 5 and 8 should remain
        let indices = masked.indices().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[5, 8]);

        // Values should be the null and 300
        let masked_values = masked.values().to_primitive();
        assert_eq!(masked_values.len(), 2);
        assert!(!masked_values.is_valid(0)); // the null value at index 5
        assert!(masked_values.is_valid(1)); // the 300 value at index 8
        assert_eq!(i32::try_from(&masked_values.scalar_at(1)).unwrap(), 300i32);
    }

    #[test]
    fn test_filter_keep_all() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        // Keep all indices (mask with indices 0-9)
        let mask = Mask::from_indices(10, (0..10).collect());
        let filtered = patches.filter(&mask).unwrap().unwrap();

        let indices = filtered.indices().to_primitive();
        let values = filtered.values().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[2, 5, 8]);
        assert_eq!(values.as_slice::<i32>(), &[100, 200, 300]);
    }

    #[test]
    fn test_filter_none() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        // Filter out all (empty mask means keep nothing)
        let mask = Mask::from_indices(10, vec![]);
        let filtered = patches.filter(&mask).unwrap();
        assert!(filtered.is_none());
    }

    #[test]
    fn test_filter_with_indices() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        // Keep indices 2, 5, 9 (so patches at 2 and 5 remain)
        let mask = Mask::from_indices(10, vec![2, 5, 9]);
        let filtered = patches.filter(&mask).unwrap().unwrap();

        let indices = filtered.indices().to_primitive();
        let values = filtered.values().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[0, 1]); // Adjusted indices
        assert_eq!(values.as_slice::<i32>(), &[100, 200]);
    }

    #[test]
    fn test_slice_full_range() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        let sliced = patches.slice(0..10).unwrap();

        let indices = sliced.indices().to_primitive();
        let values = sliced.values().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[2, 5, 8]);
        assert_eq!(values.as_slice::<i32>(), &[100, 200, 300]);
    }

    #[test]
    fn test_slice_partial() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        // Slice from 3 to 8 (includes patch at 5)
        let sliced = patches.slice(3..8).unwrap();

        let indices = sliced.indices().to_primitive();
        let values = sliced.values().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[5]); // Index stays the same
        assert_eq!(values.as_slice::<i32>(), &[200]);
        assert_eq!(sliced.array_len(), 5); // 8 - 3 = 5
        assert_eq!(sliced.offset(), 3); // New offset
    }

    #[test]
    fn test_slice_no_patches() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        // Slice from 6 to 7 (no patches in this range)
        let sliced = patches.slice(6..7);
        assert!(sliced.is_none());
    }

    #[test]
    fn test_slice_with_offset() {
        let patches = Patches::new(
            10,
            5,                                  // offset
            buffer![7u64, 10, 13].into_array(), // actual indices are 2, 5, 8
            buffer![100i32, 200, 300].into_array(),
        );

        // Slice from 3 to 8 (includes patch at actual index 5)
        let sliced = patches.slice(3..8).unwrap();

        let indices = sliced.indices().to_primitive();
        let values = sliced.values().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[10]); // Index stays the same (offset + 5 = 10)
        assert_eq!(values.as_slice::<i32>(), &[200]);
        assert_eq!(sliced.offset(), 8); // New offset = 5 + 3
    }

    #[test]
    fn test_patch_values() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        let values = patches.values().to_primitive();
        assert_eq!(i32::try_from(&values.scalar_at(0)).unwrap(), 100i32);
        assert_eq!(i32::try_from(&values.scalar_at(1)).unwrap(), 200i32);
        assert_eq!(i32::try_from(&values.scalar_at(2)).unwrap(), 300i32);
    }

    #[test]
    fn test_indices_range() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        assert_eq!(patches.min_index(), 2);
        assert_eq!(patches.max_index(), 8);
    }

    #[test]
    fn test_search_index() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        // Search for exact indices
        assert_eq!(patches.search_index(2), SearchResult::Found(0));
        assert_eq!(patches.search_index(5), SearchResult::Found(1));
        assert_eq!(patches.search_index(8), SearchResult::Found(2));

        // Search for non-patch indices
        assert_eq!(patches.search_index(0), SearchResult::NotFound(0));
        assert_eq!(patches.search_index(3), SearchResult::NotFound(1));
        assert_eq!(patches.search_index(6), SearchResult::NotFound(2));
        assert_eq!(patches.search_index(9), SearchResult::NotFound(3));
    }

    #[test]
    fn test_mask_boundary_patches() {
        // Test masking patches at array boundaries
        let patches = Patches::new(
            10,
            0,
            buffer![0u64, 9].into_array(),
            buffer![100i32, 200].into_array(),
        );

        let mask = Mask::from_iter([
            true, false, false, false, false, false, false, false, false, false,
        ]);
        let masked = patches.mask(&mask).unwrap();
        assert!(masked.is_some());
        let masked = masked.unwrap();
        let indices = masked.indices().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[9]);
        let values = masked.values().to_primitive();
        assert_eq!(values.as_slice::<i32>(), &[200]);
    }

    #[test]
    fn test_mask_all_patches_removed() {
        // Test when all patches are masked out
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        // Mask that removes all patches
        let mask = Mask::from_iter([
            false, false, true, false, false, true, false, false, true, false,
        ]);
        let masked = patches.mask(&mask).unwrap();
        assert!(masked.is_none());
    }

    #[test]
    fn test_mask_no_patches_removed() {
        // Test when no patches are masked
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        // Mask that doesn't affect any patches
        let mask = Mask::from_iter([
            true, false, false, true, false, false, true, false, false, true,
        ]);
        let masked = patches.mask(&mask).unwrap().unwrap();

        let indices = masked.indices().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[2, 5, 8]);
        let values = masked.values().to_primitive();
        assert_eq!(values.as_slice::<i32>(), &[100, 200, 300]);
    }

    #[test]
    fn test_mask_single_patch() {
        // Test with a single patch
        let patches = Patches::new(
            5,
            0,
            buffer![2u64].into_array(),
            buffer![42i32].into_array(),
        );

        // Mask that removes the single patch
        let mask = Mask::from_iter([false, false, true, false, false]);
        let masked = patches.mask(&mask).unwrap();
        assert!(masked.is_none());

        // Mask that keeps the single patch
        let mask = Mask::from_iter([true, false, false, true, false]);
        let masked = patches.mask(&mask).unwrap().unwrap();
        let indices = masked.indices().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[2]);
    }

    #[test]
    fn test_mask_contiguous_patches() {
        // Test with contiguous patches
        let patches = Patches::new(
            10,
            0,
            buffer![3u64, 4, 5, 6].into_array(),
            buffer![100i32, 200, 300, 400].into_array(),
        );

        // Mask that removes middle patches
        let mask = Mask::from_iter([
            false, false, false, false, true, true, false, false, false, false,
        ]);
        let masked = patches.mask(&mask).unwrap().unwrap();

        let indices = masked.indices().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[3, 6]);
        let values = masked.values().to_primitive();
        assert_eq!(values.as_slice::<i32>(), &[100, 400]);
    }

    #[test]
    fn test_mask_with_large_offset() {
        // Test with a large offset that shifts all indices
        let patches = Patches::new(
            20,
            15,
            buffer![16u64, 17, 19].into_array(), // actual indices are 1, 2, 4
            buffer![100i32, 200, 300].into_array(),
        );

        // Mask that removes the patch at actual index 2
        let mask = Mask::from_iter([
            false, false, true, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false,
        ]);
        let masked = patches.mask(&mask).unwrap().unwrap();

        let indices = masked.indices().to_primitive();
        assert_eq!(indices.as_slice::<u64>(), &[16, 19]);
        let values = masked.values().to_primitive();
        assert_eq!(values.as_slice::<i32>(), &[100, 300]);
    }

    #[test]
    #[should_panic(expected = "Filter mask length 5 does not match array length 10")]
    fn test_mask_wrong_length() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
        );

        // Mask with wrong length
        let mask = Mask::from_iter([false, false, true, false, false]);
        let _ = patches.mask(&mask).unwrap();
    }
}
