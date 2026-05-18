// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Range;

use num_traits::NumCast;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::LEGACY_SESSION;
#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::match_each_integer_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSorted;
use crate::search_sorted::SearchSortedSide;
use crate::validity::Validity;

/// One patch index offset is stored for each chunk.
/// This allows for constant time patch index lookups.
pub const PATCH_CHUNK_SIZE: usize = 1024;

#[derive(Copy, Clone, prost::Message)]
pub struct PatchesMetadata {
    #[prost(uint64, tag = "1")]
    len: u64,
    #[prost(uint64, tag = "2")]
    offset: u64,
    #[prost(enumeration = "PType", tag = "3")]
    indices_ptype: i32,
    #[prost(uint64, optional, tag = "4")]
    chunk_offsets_len: Option<u64>,
    #[prost(enumeration = "PType", optional, tag = "5")]
    chunk_offsets_ptype: Option<i32>,
    #[prost(uint64, optional, tag = "6")]
    offset_within_chunk: Option<u64>,
}

impl PatchesMetadata {
    #[inline]
    pub fn new(
        len: usize,
        offset: usize,
        indices_ptype: PType,
        chunk_offsets_len: Option<usize>,
        chunk_offsets_ptype: Option<PType>,
        offset_within_chunk: Option<usize>,
    ) -> Self {
        Self {
            len: len as u64,
            offset: offset as u64,
            indices_ptype: indices_ptype as i32,
            chunk_offsets_len: chunk_offsets_len.map(|len| len as u64),
            chunk_offsets_ptype: chunk_offsets_ptype.map(|pt| pt as i32),
            offset_within_chunk: offset_within_chunk.map(|len| len as u64),
        }
    }

    #[inline]
    pub fn len(&self) -> VortexResult<usize> {
        usize::try_from(self.len).map_err(|_| vortex_err!("len does not fit in usize"))
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn offset(&self) -> VortexResult<usize> {
        usize::try_from(self.offset).map_err(|_| vortex_err!("offset does not fit in usize"))
    }

    #[inline]
    pub fn chunk_offsets_dtype(&self) -> VortexResult<Option<DType>> {
        self.chunk_offsets_ptype
            .map(|t| {
                PType::try_from(t)
                    .map_err(|e| vortex_err!("invalid i32 value {t} for PType: {}", e))
                    .map(|ptype| DType::Primitive(ptype, NonNullable))
            })
            .transpose()
    }

    #[inline]
    pub fn indices_dtype(&self) -> VortexResult<DType> {
        let ptype = PType::try_from(self.indices_ptype).map_err(|e| {
            vortex_err!("invalid i32 value {} for PType: {}", self.indices_ptype, e)
        })?;
        vortex_ensure!(
            ptype.is_unsigned_int(),
            "Patch indices must be unsigned integers"
        );
        Ok(DType::Primitive(ptype, NonNullable))
    }
}

/// Metadata stored in an array's data struct for reconstructing [`Patches`] from slots.
///
/// The actual patch arrays (indices, values, chunk_offsets) live in the array's
/// slots. This struct stores only the scalar metadata needed to reassemble them.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PatchesData {
    offset: usize,
    offset_within_chunk: Option<usize>,
}

/// Slot indices for the three patch components within a slot array.
#[derive(Copy, Clone, Debug)]
pub struct PatchSlotIndices {
    pub indices: usize,
    pub values: usize,
    pub chunk_offsets: usize,
}

impl PatchesData {
    /// Extract patch metadata from an existing [`Patches`].
    pub fn from_patches(patches: &Patches) -> Self {
        Self {
            offset: patches.offset(),
            offset_within_chunk: patches.offset_within_chunk(),
        }
    }

    /// Reconstruct patches from the given slot positions.
    ///
    /// Returns `None` if `patches_data` is `None`.
    /// Panics if `patches_data` is `Some` but the indices or values slots are missing.
    pub fn patches_from_slots(
        patches_data: Option<&Self>,
        len: usize,
        slots: &[Option<ArrayRef>],
        slot_idx: PatchSlotIndices,
    ) -> Option<Patches> {
        let data = patches_data?;
        let indices = slots[slot_idx.indices]
            .as_ref()
            .vortex_expect("patches_data is set but patch_indices slot is missing");
        let values = slots[slot_idx.values]
            .as_ref()
            .vortex_expect("patches_data is set but patch_values slot is missing");
        Some(unsafe {
            Patches::new_unchecked(
                len,
                data.offset,
                indices.clone(),
                values.clone(),
                slots[slot_idx.chunk_offsets].clone(),
                data.offset_within_chunk,
            )
        })
    }

    /// Push 3 patch slots (indices, values, chunk_offsets) onto a slot vector.
    ///
    /// If `patches` is `None`, pushes three `None` entries.
    pub fn push_slots(slots: &mut ArraySlots, patches: Option<&Patches>) {
        match patches {
            Some(p) => {
                slots.push(Some(p.indices().clone()));
                slots.push(Some(p.values().clone()));
                slots.push(p.chunk_offsets().clone());
            }
            None => {
                slots.push(None);
                slots.push(None);
                slots.push(None);
            }
        }
    }

    /// Returns the patch offset.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the offset within the first chunk, if chunk offsets are present.
    #[inline]
    pub fn offset_within_chunk(&self) -> Option<usize> {
        self.offset_within_chunk
    }
}

/// A helper for working with patched arrays.
#[derive(Debug, Clone)]
pub struct Patches {
    array_len: usize,
    offset: usize,
    indices: ArrayRef,
    values: ArrayRef,
    /// Stores the patch index offset for each chunk.
    ///
    /// This allows us to lookup the patches for a given chunk in constant time via
    /// `patch_indices[chunk_offsets[i]..chunk_offsets[i+1]]`.
    ///
    /// This is optional for compatibility reasons.
    chunk_offsets: Option<ArrayRef>,
    /// Chunk offsets are only sliced off in case the slice is fully
    /// outside of the chunk range.
    ///
    /// Though the range for indices and values is sliced in terms of
    /// individual elements, not chunks. To account for that we do a
    /// saturating sub when adjusting the indices based on the chunk offset.
    ///
    /// `offset_within_chunk` is necessary in order to keep track of how many
    /// elements were sliced off within the chunk.
    offset_within_chunk: Option<usize>,
}

impl Patches {
    pub fn new(
        array_len: usize,
        offset: usize,
        indices: ArrayRef,
        values: ArrayRef,
        chunk_offsets: Option<ArrayRef>,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            indices.len() == values.len(),
            "Patch indices and values must have the same length"
        );
        vortex_ensure!(
            indices.dtype().is_unsigned_int() && !indices.dtype().is_nullable(),
            "Patch indices must be non-nullable unsigned integers, got {:?}",
            indices.dtype()
        );

        vortex_ensure!(
            indices.len() <= array_len,
            "Patch indices must be shorter than the array length"
        );
        vortex_ensure!(!indices.is_empty(), "Patch indices must not be empty");

        // Perform validation of components when they are host-resident.
        // This is not possible to do eagerly when the data is on GPU memory.
        if indices.is_host() && values.is_host() {
            let max = usize::try_from(&indices.execute_scalar(
                indices.len() - 1,
                &mut LEGACY_SESSION.create_execution_ctx(),
            )?)
            .map_err(|_| vortex_err!("indices must be a number"))?;
            vortex_ensure!(
                max - offset < array_len,
                "Patch indices {max:?}, offset {offset} are longer than the array length {array_len}"
            );

            #[cfg(debug_assertions)]
            {
                use crate::VortexSessionExecute;
                use crate::aggregate_fn::fns::is_sorted::is_sorted;
                let mut ctx = LEGACY_SESSION.create_execution_ctx();
                assert!(
                    is_sorted(&indices, &mut ctx).unwrap_or(false),
                    "Patch indices must be sorted"
                );
            }
        }

        Ok(Self {
            array_len,
            offset,
            indices,
            values,
            chunk_offsets: chunk_offsets.clone(),
            // Initialize with `Some(0)` only if `chunk_offsets` are set.
            offset_within_chunk: chunk_offsets.map(|_| 0),
        })
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
        chunk_offsets: Option<ArrayRef>,
        offset_within_chunk: Option<usize>,
    ) -> Self {
        Self {
            array_len,
            offset,
            indices,
            values,
            chunk_offsets,
            offset_within_chunk,
        }
    }

    #[inline]
    pub fn array_len(&self) -> usize {
        self.array_len
    }

    #[inline]
    pub fn num_patches(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    pub fn dtype(&self) -> &DType {
        self.values.dtype()
    }

    #[inline]
    pub fn indices(&self) -> &ArrayRef {
        &self.indices
    }

    #[inline]
    pub fn into_indices(self) -> ArrayRef {
        self.indices
    }

    #[inline]
    pub fn indices_mut(&mut self) -> &mut ArrayRef {
        &mut self.indices
    }

    #[inline]
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }

    #[inline]
    pub fn into_values(self) -> ArrayRef {
        self.values
    }

    #[inline]
    pub fn values_mut(&mut self) -> &mut ArrayRef {
        &mut self.values
    }

    #[inline]
    // Absolute offset: 0 if the array is unsliced.
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn chunk_offsets(&self) -> &Option<ArrayRef> {
        &self.chunk_offsets
    }

    #[inline]
    pub fn chunk_offset_at(&self, idx: usize) -> VortexResult<usize> {
        let Some(chunk_offsets) = &self.chunk_offsets else {
            vortex_bail!("chunk_offsets must be set to retrieve offset at index")
        };

        chunk_offsets
            .execute_scalar(idx, &mut LEGACY_SESSION.create_execution_ctx())?
            .as_primitive()
            .as_::<usize>()
            .ok_or_else(|| vortex_err!("chunk offset does not fit in usize"))
    }

    /// Returns the number of patches sliced off from the current first chunk.
    ///
    /// When patches are sliced, the chunk offsets array is also sliced to only include
    /// chunks that overlap with the slice range. However, the slice boundary may fall
    /// in the middle of a chunk's patch range. This offset indicates how many patches
    /// at the start of the first chunk should be skipped.
    ///
    /// Returns `None` if chunk offsets are not set.
    #[inline]
    pub fn offset_within_chunk(&self) -> Option<usize> {
        self.offset_within_chunk
    }

    #[inline]
    pub fn indices_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.indices.dtype())
            .map_err(|_| vortex_err!("indices dtype is not primitive"))
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
        let chunk_offsets_len = self.chunk_offsets.as_ref().map(|co| co.len());
        let chunk_offsets_ptype = self.chunk_offsets.as_ref().map(|co| co.dtype().as_ptype());

        Ok(PatchesMetadata::new(
            self.indices.len(),
            self.offset,
            self.indices.dtype().as_ptype(),
            chunk_offsets_len,
            chunk_offsets_ptype,
            self.offset_within_chunk,
        ))
    }

    pub fn cast_values(self, values_dtype: &DType) -> VortexResult<Self> {
        // SAFETY: casting does not affect the relationship between the indices and values
        unsafe {
            Ok(Self::new_unchecked(
                self.array_len,
                self.offset,
                self.indices,
                self.values.cast(values_dtype.clone())?,
                self.chunk_offsets,
                self.offset_within_chunk,
            ))
        }
    }

    /// Get the patched value at a given index if it exists.
    pub fn get_patched(&self, index: usize) -> VortexResult<Option<Scalar>> {
        self.search_index(index)?
            .to_found()
            .map(|patch_idx| {
                self.values()
                    .execute_scalar(patch_idx, &mut LEGACY_SESSION.create_execution_ctx())
            })
            .transpose()
    }

    /// Searches for `index` in the indices array.
    ///
    /// Chooses between chunked search when [`Self::chunk_offsets`] is
    /// available, and binary search otherwise. The `index` parameter is
    /// adjusted by [`Self::offset`] for both.
    ///
    /// # Arguments
    /// * `index` - The index to search for
    ///
    /// # Returns
    /// * [`SearchResult::Found(patch_idx)`] - If a patch exists at this index, returns the
    ///   position in the patches array
    /// * [`SearchResult::NotFound(insertion_point)`] - If no patch exists, returns where
    ///   a patch at this index would be inserted to maintain sorted order
    ///
    /// [`SearchResult::Found(patch_idx)`]: SearchResult::Found
    /// [`SearchResult::NotFound(insertion_point)`]: SearchResult::NotFound
    pub fn search_index(&self, index: usize) -> VortexResult<SearchResult> {
        if self.chunk_offsets.is_some() {
            return self.search_index_chunked(index);
        }

        Self::search_index_binary_search(&self.indices, index + self.offset)
    }

    /// Binary searches for `needle` in the indices array.
    ///
    /// # Returns
    /// [`SearchResult::Found`] with the position if needle exists, or [`SearchResult::NotFound`]
    /// with the insertion point if not found.
    fn search_index_binary_search(indices: &ArrayRef, needle: usize) -> VortexResult<SearchResult> {
        if indices.is_canonical() {
            #[expect(deprecated)]
            let primitive = indices.to_primitive();
            match_each_integer_ptype!(primitive.ptype(), |T| {
                let Ok(needle) = T::try_from(needle) else {
                    // If the needle is not of type T, then it cannot possibly be in this array.
                    //
                    // The needle is a non-negative integer (a usize); therefore, it must be larger
                    // than all values in this array.
                    return Ok(SearchResult::NotFound(primitive.len()));
                };
                return primitive
                    .as_slice::<T>()
                    .search_sorted(&needle, SearchSortedSide::Left);
            });
        }
        indices
            .as_primitive_typed()
            .search_sorted(&PValue::U64(needle as u64), SearchSortedSide::Left)
    }

    /// Constant time searches for `index` in the indices array.
    ///
    /// First determines which chunk the target index falls into, then performs
    /// a binary search within that chunk's range.
    ///
    /// Returns a [`SearchResult`] indicating either the exact patch index if found,
    /// or the insertion point if not found.
    ///
    /// Returns an error if `chunk_offsets` or `offset_within_chunk` are not set.
    fn search_index_chunked(&self, index: usize) -> VortexResult<SearchResult> {
        let Some(chunk_offsets) = &self.chunk_offsets else {
            vortex_bail!("chunk_offsets is required to be set")
        };

        let Some(offset_within_chunk) = self.offset_within_chunk else {
            vortex_bail!("offset_within_chunk is required to be set")
        };

        if index >= self.array_len() {
            return Ok(SearchResult::NotFound(self.indices().len()));
        }

        let chunk_idx = (index + self.offset % PATCH_CHUNK_SIZE) / PATCH_CHUNK_SIZE;

        // Patch index offsets are absolute and need to be offset by the first chunk of the current slice.
        let base_offset = self.chunk_offset_at(0)?;

        let patches_start_idx = (self.chunk_offset_at(chunk_idx)? - base_offset)
            // Chunk offsets are only sliced off in case the slice is fully
            // outside of the chunk range.
            //
            // Though the range for indices and values is sliced in terms of
            // individual elements, not chunks. To account for that we do a
            // saturating sub when adjusting the indices based on the chunk offset.
            .saturating_sub(offset_within_chunk);

        let patches_end_idx = if chunk_idx < chunk_offsets.len() - 1 {
            (self.chunk_offset_at(chunk_idx + 1)? - base_offset)
                .saturating_sub(offset_within_chunk)
                .min(self.indices.len())
        } else {
            self.indices.len()
        };

        let chunk_indices = self.indices.slice(patches_start_idx..patches_end_idx)?;
        let result = Self::search_index_binary_search(&chunk_indices, index + self.offset)?;

        Ok(match result {
            SearchResult::Found(idx) => SearchResult::Found(patches_start_idx + idx),
            SearchResult::NotFound(idx) => SearchResult::NotFound(patches_start_idx + idx),
        })
    }

    /// Batch version of `search_index`.
    ///
    /// In contrast to `search_index`, this function requires `indices` as
    /// well as `chunk_offsets` to be passed as slices. This is to avoid
    /// redundant canonicalization and `scalar_at` lookups across calls.
    fn search_index_chunked_batch<T, O>(
        &self,
        indices: &[T],
        chunk_offsets: &[O],
        index: T,
    ) -> VortexResult<SearchResult>
    where
        T: UnsignedPType,
        O: UnsignedPType,
        usize: TryFrom<T>,
        usize: TryFrom<O>,
    {
        let Some(offset_within_chunk) = self.offset_within_chunk else {
            vortex_bail!("offset_within_chunk is required to be set")
        };

        let chunk_idx = {
            let Ok(index) = usize::try_from(index) else {
                // If the needle cannot be converted to usize, it's larger than all values in this array.
                return Ok(SearchResult::NotFound(indices.len()));
            };

            if index >= self.array_len() {
                return Ok(SearchResult::NotFound(self.indices().len()));
            }

            (index + self.offset % PATCH_CHUNK_SIZE) / PATCH_CHUNK_SIZE
        };

        // Patch index offsets are absolute and need to be offset by the first chunk of the current slice.
        let chunk_offset = usize::try_from(chunk_offsets[chunk_idx] - chunk_offsets[0])
            .map_err(|_| vortex_err!("chunk_offset failed to convert to usize"))?;

        let patches_start_idx = chunk_offset
            // Chunk offsets are only sliced off in case the slice is fully
            // outside of the chunk range.
            //
            // Though the range for indices and values is sliced in terms of
            // individual elements, not chunks. To account for that we do a
            // saturating sub when adjusting the indices based on the chunk offset.
            .saturating_sub(offset_within_chunk);

        let patches_end_idx = if chunk_idx < chunk_offsets.len() - 1 {
            usize::try_from(chunk_offsets[chunk_idx + 1] - chunk_offsets[0])
                .map_err(|_| vortex_err!("patches_end_idx failed to convert to usize"))?
                .saturating_sub(offset_within_chunk)
                .min(indices.len())
        } else {
            self.indices.len()
        };

        let Some(offset) = T::from(self.offset) else {
            // If the offset cannot be converted to T, it's larger than all values in this array.
            return Ok(SearchResult::NotFound(indices.len()));
        };

        let chunk_indices = &indices[patches_start_idx..patches_end_idx];
        let result = chunk_indices.search_sorted(&(index + offset), SearchSortedSide::Left)?;

        Ok(match result {
            SearchResult::Found(idx) => SearchResult::Found(patches_start_idx + idx),
            SearchResult::NotFound(idx) => SearchResult::NotFound(patches_start_idx + idx),
        })
    }

    /// Returns the minimum patch index
    pub fn min_index(&self) -> VortexResult<usize> {
        let first = self
            .indices
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
            .as_primitive()
            .as_::<usize>()
            .ok_or_else(|| vortex_err!("index does not fit in usize"))?;
        Ok(first - self.offset)
    }

    /// Returns the maximum patch index
    pub fn max_index(&self) -> VortexResult<usize> {
        let last = self
            .indices
            .execute_scalar(
                self.indices.len() - 1,
                &mut LEGACY_SESSION.create_execution_ctx(),
            )?
            .as_primitive()
            .as_::<usize>()
            .ok_or_else(|| vortex_err!("index does not fit in usize"))?;
        Ok(last - self.offset)
    }

    /// Filter the patches by a mask, resulting in new patches for the filtered array.
    pub fn filter(&self, mask: &Mask, ctx: &mut ExecutionCtx) -> VortexResult<Option<Self>> {
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
                let flat_indices = self.indices().clone().execute::<PrimitiveArray>(ctx)?;
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
    // TODO(joe): make this lazy and remove the ctx.
    pub fn mask(&self, mask: &Mask, ctx: &mut ExecutionCtx) -> VortexResult<Option<Self>> {
        if mask.len() != self.array_len {
            vortex_bail!(
                "Filter mask length {} does not match array length {}",
                mask.len(),
                self.array_len
            );
        }

        let filter_mask = match mask.bit_buffer() {
            AllOr::All => return Ok(None),
            AllOr::None => return Ok(Some(self.clone())),
            AllOr::Some(masked) => {
                let patch_indices = self.indices().clone().execute::<PrimitiveArray>(ctx)?;
                match_each_unsigned_integer_ptype!(patch_indices.ptype(), |P| {
                    let patch_indices = patch_indices.as_slice::<P>();
                    Mask::from_buffer(BitBuffer::collect_bool(patch_indices.len(), |i| {
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
        let filtered_indices = self.indices.filter(filter_mask.clone())?;
        let filtered_values = self.values.filter(filter_mask)?;

        Ok(Some(Self {
            array_len: self.array_len,
            offset: self.offset,
            indices: filtered_indices,
            values: filtered_values,
            // TODO(0ax1): Chunk offsets are invalid after a filter is applied.
            chunk_offsets: None,
            offset_within_chunk: self.offset_within_chunk,
        }))
    }

    /// Slice the patches by a range of the patched array.
    pub fn slice(&self, range: Range<usize>) -> VortexResult<Option<Self>> {
        let slice_start_idx = self.search_index(range.start)?.to_index();
        let slice_end_idx = self.search_index(range.end)?.to_index();

        if slice_start_idx == slice_end_idx {
            return Ok(None);
        }

        let values = self.values().slice(slice_start_idx..slice_end_idx)?;
        let indices = self.indices().slice(slice_start_idx..slice_end_idx)?;

        let new_chunk_offsets = self
            .chunk_offsets
            .as_ref()
            .map(|chunk_offsets| -> VortexResult<ArrayRef> {
                let chunk_relative_offset = self.offset % PATCH_CHUNK_SIZE;
                let chunk_start_idx = (chunk_relative_offset + range.start) / PATCH_CHUNK_SIZE;
                let chunk_end_idx = (chunk_relative_offset + range.end).div_ceil(PATCH_CHUNK_SIZE);
                chunk_offsets.slice(chunk_start_idx..chunk_end_idx)
            })
            .transpose()?;

        let offset_within_chunk = new_chunk_offsets
            .as_ref()
            .map(|new_chunk_offsets| -> VortexResult<usize> {
                let new_chunk_base = new_chunk_offsets
                    .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
                    .as_primitive()
                    .as_::<usize>()
                    .ok_or_else(|| vortex_err!("chunk offset does not fit in usize"))?;
                let parent_chunk_base = self.chunk_offset_at(0)?;
                let parent_within = self.offset_within_chunk.unwrap_or(0);
                Ok(parent_chunk_base + parent_within + slice_start_idx - new_chunk_base)
            })
            .transpose()?;

        Ok(Some(Self {
            array_len: range.len(),
            offset: range.start + self.offset(),
            indices,
            values,
            chunk_offsets: new_chunk_offsets,
            offset_within_chunk,
        }))
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
    pub fn take_with_nulls(
        &self,
        take_indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self>> {
        if take_indices.is_empty() {
            return Ok(None);
        }

        let take_indices = take_indices.clone().execute::<PrimitiveArray>(ctx)?;
        if self.is_map_faster_than_search(&take_indices) {
            self.take_map(take_indices, true, ctx)
        } else {
            self.take_search(take_indices, true, ctx)
        }
    }

    /// Take the indices from the patches.
    ///
    /// Any nulls in take_indices are ignored.
    pub fn take(
        &self,
        take_indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self>> {
        if take_indices.is_empty() {
            return Ok(None);
        }

        let take_indices = take_indices.clone().execute::<PrimitiveArray>(ctx)?;
        if self.is_map_faster_than_search(&take_indices) {
            self.take_map(take_indices, false, ctx)
        } else {
            self.take_search(take_indices, false, ctx)
        }
    }

    #[expect(
        clippy::cognitive_complexity,
        reason = "complexity is from nested match_each_* macros"
    )]
    pub fn take_search(
        &self,
        take_indices: PrimitiveArray,
        include_nulls: bool,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self>> {
        let take_indices_validity = take_indices.validity()?;
        let patch_indices = self.indices.clone().execute::<PrimitiveArray>(ctx)?;
        let chunk_offsets = self
            .chunk_offsets()
            .as_ref()
            .map(|co| co.clone().execute::<PrimitiveArray>(ctx))
            .transpose()?;

        let (values_indices, new_indices): (BufferMut<u64>, BufferMut<u64>) =
            match_each_unsigned_integer_ptype!(patch_indices.ptype(), |PatchT| {
                let patch_indices_slice = patch_indices.as_slice::<PatchT>();
                match_each_integer_ptype!(take_indices.ptype(), |TakeT| {
                    let take_slice = take_indices.as_slice::<TakeT>();

                    if let Some(chunk_offsets) = chunk_offsets {
                        match_each_unsigned_integer_ptype!(chunk_offsets.ptype(), |OffsetT| {
                            let chunk_offsets = chunk_offsets.as_slice::<OffsetT>();
                            take_indices_with_search_fn(
                                patch_indices_slice,
                                take_slice,
                                take_indices
                                    .as_ref()
                                    .validity()?
                                    .execute_mask(take_indices.as_ref().len(), ctx)?,
                                include_nulls,
                                |take_idx| {
                                    self.search_index_chunked_batch(
                                        patch_indices_slice,
                                        chunk_offsets,
                                        take_idx,
                                    )
                                },
                            )?
                        })
                    } else {
                        take_indices_with_search_fn(
                            patch_indices_slice,
                            take_slice,
                            take_indices
                                .as_ref()
                                .validity()?
                                .execute_mask(take_indices.as_ref().len(), ctx)?,
                            include_nulls,
                            |take_idx| {
                                let Some(offset) = <PatchT as NumCast>::from(self.offset) else {
                                    // If the offset cannot be converted to T, it's larger than all values in this array.
                                    return Ok(SearchResult::NotFound(patch_indices_slice.len()));
                                };

                                patch_indices_slice
                                    .search_sorted(&(take_idx + offset), SearchSortedSide::Left)
                            },
                        )?
                    }
                })
            });

        if new_indices.is_empty() {
            return Ok(None);
        }

        let new_indices = new_indices.into_array();
        let new_array_len = take_indices.len();
        let values_validity = take_indices_validity.take(&new_indices)?;

        Ok(Some(Self {
            array_len: new_array_len,
            offset: 0,
            indices: new_indices,
            values: self
                .values()
                .take(PrimitiveArray::new(values_indices, values_validity).into_array())?,
            chunk_offsets: None,
            offset_within_chunk: Some(0), // Reset when creating new Patches.
        }))
    }

    pub fn take_map(
        &self,
        take_indices: PrimitiveArray,
        include_nulls: bool,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self>> {
        let indices = self.indices.clone().execute::<PrimitiveArray>(ctx)?;
        let new_length = take_indices.len();

        let min_index = self.min_index()?;
        let max_index = self.max_index()?;

        let Some((new_sparse_indices, value_indices)) =
            match_each_unsigned_integer_ptype!(indices.ptype(), |Indices| {
                match_each_integer_ptype!(take_indices.ptype(), |TakeIndices| {
                    let take_validity = take_indices
                        .validity()?
                        .execute_mask(take_indices.len(), ctx)?;
                    let take_nullability = take_indices.validity()?.nullability();
                    let take_slice = take_indices.as_slice::<TakeIndices>();
                    take_map::<_, TakeIndices>(
                        indices.as_slice::<Indices>(),
                        take_slice,
                        take_validity,
                        take_nullability,
                        self.offset(),
                        min_index,
                        max_index,
                        include_nulls,
                    )?
                })
            })
        else {
            return Ok(None);
        };

        let taken_values = self.values().take(value_indices)?;

        Ok(Some(Patches {
            array_len: new_length,
            offset: 0,
            indices: new_sparse_indices,
            values: taken_values,
            // TODO(0ax1): Chunk offsets are invalid after take is applied.
            chunk_offsets: None,
            offset_within_chunk: self.offset_within_chunk,
        }))
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

        Ok(Self {
            array_len: self.array_len,
            offset: self.offset,
            indices: self.indices,
            values,
            chunk_offsets: self.chunk_offsets,
            offset_within_chunk: self.offset_within_chunk,
        })
    }
}

#[expect(clippy::too_many_arguments)] // private function, can clean up one day
fn take_map<I: NativePType + Hash + Eq + TryFrom<usize>, T: NativePType>(
    indices: &[I],
    take_indices: &[T],
    take_validity: Mask,
    take_nullability: Nullability,
    indices_offset: usize,
    min_index: usize,
    max_index: usize,
    include_nulls: bool,
) -> VortexResult<Option<(ArrayRef, ArrayRef)>>
where
    usize: TryFrom<T>,
    VortexError: From<<I as TryFrom<usize>>::Error>,
{
    let offset_i = I::try_from(indices_offset)?;

    let sparse_index_to_value_index: HashMap<I, usize> = indices
        .iter()
        .copied()
        .map(|idx| idx - offset_i)
        .enumerate()
        .map(|(value_index, sparse_index)| (sparse_index, value_index))
        .collect();

    let mut new_sparse_indices = BufferMut::<u64>::with_capacity(take_indices.len());
    let mut value_indices = BufferMut::<u64>::with_capacity(take_indices.len());

    for (idx_in_take, &take_idx) in take_indices.iter().enumerate() {
        let ti = usize::try_from(take_idx)
            .map_err(|_| vortex_err!("Failed to convert index to usize"))?;

        // If we have to take nulls the take index doesn't matter, make it 0 for consistency
        let is_null = match take_validity.bit_buffer() {
            AllOr::All => false,
            AllOr::None => true,
            AllOr::Some(buf) => !buf.value(idx_in_take),
        };
        if is_null {
            if include_nulls {
                new_sparse_indices.push(idx_in_take as u64);
                value_indices.push(0);
            }
        } else if ti >= min_index && ti <= max_index {
            let ti_as_i = I::try_from(ti)
                .map_err(|_| vortex_err!("take index does not fit in index type"))?;
            if let Some(&value_index) = sparse_index_to_value_index.get(&ti_as_i) {
                new_sparse_indices.push(idx_in_take as u64);
                value_indices.push(value_index as u64);
            }
        }
    }

    if new_sparse_indices.is_empty() {
        return Ok(None);
    }

    let new_sparse_indices = new_sparse_indices.into_array();
    let values_validity =
        Validity::from_mask(take_validity, take_nullability).take(&new_sparse_indices)?;
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
fn filter_patches_with_mask<T: IntegerPType>(
    patch_indices: &[T],
    offset: usize,
    patch_values: &ArrayRef,
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
            let left_min = patch_indices[mask_idx]
                .to_usize()
                .ok_or_else(|| vortex_err!("patch index does not fit in usize"))?
                - offset;
            let left_max = patch_indices[mask_idx + STRIDE]
                .to_usize()
                .ok_or_else(|| vortex_err!("patch index does not fit in usize"))?
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

        let left = patch_indices[mask_idx]
            .to_usize()
            .ok_or_else(|| vortex_err!("patch index does not fit in usize"))?
            - offset;
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
    let new_patch_values =
        patch_values.filter(Mask::from_indices(patch_values.len(), new_mask_indices))?;

    Ok(Some(Patches::new(
        true_count,
        0,
        new_patch_indices,
        new_patch_values,
        // TODO(0ax1): Chunk offsets are invalid after a filter is applied.
        None,
    )?))
}

fn take_indices_with_search_fn<
    I: UnsignedPType,
    T: IntegerPType,
    F: Fn(I) -> VortexResult<SearchResult>,
>(
    indices: &[I],
    take_indices: &[T],
    take_validity: Mask,
    include_nulls: bool,
    search_fn: F,
) -> VortexResult<(BufferMut<u64>, BufferMut<u64>)> {
    let mut values_indices = BufferMut::with_capacity(take_indices.len());
    let mut new_indices = BufferMut::with_capacity(take_indices.len());

    for (new_patch_idx, &take_idx) in take_indices.iter().enumerate() {
        if !take_validity.value(new_patch_idx) {
            if include_nulls {
                // For nulls, patch index doesn't matter - use 0 for consistency
                values_indices.push(0u64);
                new_indices.push(new_patch_idx as u64);
            }
            continue;
        } else {
            let search_result = match I::from(take_idx) {
                Some(idx) => search_fn(idx)?,
                None => SearchResult::NotFound(indices.len()),
            };

            if let Some(patch_idx) = search_result.to_found() {
                values_indices.push(patch_idx as u64);
                new_indices.push(new_patch_idx as u64);
            }
        }
    }

    Ok((values_indices, new_indices))
}

#[cfg(test)]
mod test {
    use vortex_buffer::BufferMut;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    #[expect(deprecated)]
    use crate::ToCanonical as _;
    use crate::VortexSessionExecute;
    use crate::assert_arrays_eq;
    use crate::patches::Patches;
    use crate::patches::PrimitiveArray;
    use crate::search_sorted::SearchResult;
    use crate::validity::Validity;

    #[test]
    fn test_filter() {
        let patches = Patches::new(
            100,
            0,
            buffer![10u32, 11, 20].into_array(),
            buffer![100, 110, 200].into_array(),
            None,
        )
        .unwrap();

        let filtered = patches
            .filter(
                &Mask::from_indices(100, vec![10, 20, 30]),
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap()
            .unwrap();

        assert_arrays_eq!(filtered.indices(), PrimitiveArray::from_iter([0u64, 1]));
        assert_arrays_eq!(filtered.values(), PrimitiveArray::from_iter([100i32, 200]));
    }

    #[test]
    fn take_with_nulls() {
        let patches = Patches::new(
            20,
            0,
            buffer![2u64, 9, 15].into_array(),
            PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
            None,
        )
        .unwrap();

        let taken = patches
            .take(
                &PrimitiveArray::new(buffer![9, 0], Validity::from_iter(vec![true, false]))
                    .into_array(),
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap()
            .unwrap();
        #[expect(deprecated)]
        let primitive_values = taken.values().to_primitive();
        #[expect(deprecated)]
        let primitive_indices = taken.indices().to_primitive();
        assert_eq!(taken.array_len(), 2);
        assert_arrays_eq!(
            primitive_values,
            PrimitiveArray::from_option_iter([Some(44i32)])
        );
        assert_arrays_eq!(primitive_indices, PrimitiveArray::from_iter([0u64]));
        assert_eq!(
            primitive_values
                .as_ref()
                .validity()
                .unwrap()
                .execute_mask(
                    primitive_values.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx()
                )
                .unwrap(),
            Mask::from_iter(vec![true])
        );
    }

    #[test]
    fn take_search_with_nulls_chunked() {
        let patches = Patches::new(
            20,
            0,
            buffer![2u64, 9, 15].into_array(),
            buffer![33_i32, 44, 55].into_array(),
            Some(buffer![0u64].into_array()),
        )
        .unwrap();

        let taken = patches
            .take_search(
                PrimitiveArray::new(buffer![9, 0], Validity::from_iter([true, false])),
                true,
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap()
            .unwrap();

        #[expect(deprecated)]
        let primitive_values = taken.values().to_primitive();
        assert_eq!(taken.array_len(), 2);
        assert_arrays_eq!(
            primitive_values,
            PrimitiveArray::from_option_iter([Some(44i32), None])
        );
        assert_arrays_eq!(taken.indices(), PrimitiveArray::from_iter([0u64, 1]));

        assert_eq!(
            primitive_values
                .as_ref()
                .validity()
                .unwrap()
                .execute_mask(
                    primitive_values.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx()
                )
                .unwrap(),
            Mask::from_iter([true, false])
        );
    }

    #[test]
    fn take_search_chunked_multiple_chunks() {
        let patches = Patches::new(
            2048,
            0,
            buffer![100u64, 500, 1200, 1800].into_array(),
            buffer![10_i32, 20, 30, 40].into_array(),
            Some(buffer![0u64, 2].into_array()),
        )
        .unwrap();

        let taken = patches
            .take_search(
                PrimitiveArray::new(buffer![500, 1200, 999], Validity::AllValid),
                true,
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap()
            .unwrap();

        assert_eq!(taken.array_len(), 3);
        assert_arrays_eq!(
            taken.values(),
            PrimitiveArray::from_option_iter([Some(20i32), Some(30)])
        );
    }

    #[test]
    fn take_search_chunked_indices_with_no_patches() {
        let patches = Patches::new(
            20,
            0,
            buffer![2u64, 9, 15].into_array(),
            buffer![33_i32, 44, 55].into_array(),
            Some(buffer![0u64].into_array()),
        )
        .unwrap();

        let taken = patches
            .take_search(
                PrimitiveArray::new(buffer![3, 4, 5], Validity::AllValid),
                true,
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap();

        assert!(taken.is_none());
    }

    #[test]
    fn take_search_chunked_interleaved() {
        let patches = Patches::new(
            30,
            0,
            buffer![5u64, 10, 20, 25].into_array(),
            buffer![100_i32, 200, 300, 400].into_array(),
            Some(buffer![0u64].into_array()),
        )
        .unwrap();

        let taken = patches
            .take_search(
                PrimitiveArray::new(buffer![10, 15, 20, 99], Validity::AllValid),
                true,
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap()
            .unwrap();

        assert_eq!(taken.array_len(), 4);
        assert_arrays_eq!(
            taken.values(),
            PrimitiveArray::from_option_iter([Some(200i32), Some(300)])
        );
    }

    #[test]
    fn test_take_search_multiple_chunk_offsets() {
        let patches = Patches::new(
            1500,
            0,
            BufferMut::from_iter(0..1500u64).into_array(),
            BufferMut::from_iter(0..1500i32).into_array(),
            Some(buffer![0u64, 1024u64].into_array()),
        )
        .unwrap();

        let taken = patches
            .take_search(
                PrimitiveArray::new(BufferMut::from_iter(0..1500u64), Validity::AllValid),
                false,
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap()
            .unwrap();

        assert_eq!(taken.array_len(), 1500);
    }

    #[test]
    fn test_slice() {
        let values = buffer![15_u32, 135, 13531, 42].into_array();
        let indices = buffer![10_u64, 11, 50, 100].into_array();

        let patches = Patches::new(101, 0, indices, values, None).unwrap();

        let sliced = patches.slice(15..100).unwrap().unwrap();
        assert_eq!(sliced.array_len(), 100 - 15);
        assert_arrays_eq!(sliced.values(), PrimitiveArray::from_iter([13531u32]));
    }

    #[test]
    fn doubly_sliced() {
        let values = buffer![15_u32, 135, 13531, 42].into_array();
        let indices = buffer![10_u64, 11, 50, 100].into_array();

        let patches = Patches::new(101, 0, indices, values, None).unwrap();

        let sliced = patches.slice(15..100).unwrap().unwrap();
        assert_eq!(sliced.array_len(), 100 - 15);
        assert_arrays_eq!(sliced.values(), PrimitiveArray::from_iter([13531u32]));

        let doubly_sliced = sliced.slice(35..36).unwrap().unwrap();
        assert_arrays_eq!(
            doubly_sliced.values(),
            PrimitiveArray::from_iter([13531u32])
        );
    }

    #[test]
    fn test_mask_all_true() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        let mask = Mask::new_true(10);
        let masked = patches
            .mask(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        assert!(masked.is_none());
    }

    #[test]
    fn test_mask_all_false() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        let mask = Mask::new_false(10);
        let masked = patches
            .mask(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .unwrap();

        // No patch values should be masked
        assert_arrays_eq!(
            masked.values(),
            PrimitiveArray::from_iter([100i32, 200, 300])
        );
        assert!(
            masked
                .values()
                .is_valid(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert!(
            masked
                .values()
                .is_valid(1, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert!(
            masked
                .values()
                .is_valid(2, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );

        // Indices should remain unchanged
        assert_arrays_eq!(masked.indices(), PrimitiveArray::from_iter([2u64, 5, 8]));
    }

    #[test]
    fn test_mask_partial() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        // Mask that removes patches at indices 2 and 8 (but not 5)
        let mask = Mask::from_iter([
            false, false, true, false, false, false, false, false, true, false,
        ]);
        let masked = patches
            .mask(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .unwrap();

        // Only the patch at index 5 should remain
        assert_eq!(masked.values().len(), 1);
        assert_arrays_eq!(masked.values(), PrimitiveArray::from_iter([200i32]));

        // Only index 5 should remain
        assert_arrays_eq!(masked.indices(), PrimitiveArray::from_iter([5u64]));
    }

    #[test]
    fn test_mask_with_offset() {
        let patches = Patches::new(
            10,
            5,                                  // offset
            buffer![7u64, 10, 13].into_array(), // actual indices are 2, 5, 8
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        // Mask that sets actual index 2 to null
        let mask = Mask::from_iter([
            false, false, true, false, false, false, false, false, false, false,
        ]);

        let masked = patches
            .mask(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .unwrap();
        assert_eq!(masked.array_len(), 10);
        assert_eq!(masked.offset(), 5);
        assert_arrays_eq!(masked.indices(), PrimitiveArray::from_iter([10u64, 13]));
        assert_arrays_eq!(masked.values(), PrimitiveArray::from_iter([200i32, 300]));
    }

    #[test]
    fn test_mask_nullable_values() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            PrimitiveArray::from_option_iter([Some(100i32), None, Some(300)]).into_array(),
            None,
        )
        .unwrap();

        // Test masking removes patch at index 2
        let mask = Mask::from_iter([
            false, false, true, false, false, false, false, false, false, false,
        ]);
        let masked = patches
            .mask(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .unwrap();

        // Patches at indices 5 and 8 should remain
        assert_arrays_eq!(masked.indices(), PrimitiveArray::from_iter([5u64, 8]));

        // Values should be the null and 300
        #[expect(deprecated)]
        let masked_values = masked.values().to_primitive();
        assert_eq!(masked_values.len(), 2);
        assert!(
            !masked_values
                .is_valid(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        ); // the null value at index 5
        assert!(
            masked_values
                .is_valid(1, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        ); // the 300 value at index 8
        assert_eq!(
            i32::try_from(
                &masked_values
                    .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap()
            )
            .unwrap(),
            300i32
        );
    }

    #[test]
    fn test_filter_keep_all() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        // Keep all indices (mask with indices 0-9)
        let mask = Mask::from_indices(10, 0..10);
        let filtered = patches
            .filter(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .unwrap();

        assert_arrays_eq!(filtered.indices(), PrimitiveArray::from_iter([2u64, 5, 8]));
        assert_arrays_eq!(
            filtered.values(),
            PrimitiveArray::from_iter([100i32, 200, 300])
        );
    }

    #[test]
    fn test_filter_none() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        // Filter out all (empty mask means keep nothing)
        let mask = Mask::from_indices(10, vec![]);
        let filtered = patches
            .filter(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        assert!(filtered.is_none());
    }

    #[test]
    fn test_filter_with_indices() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        // Keep indices 2, 5, 9 (so patches at 2 and 5 remain)
        let mask = Mask::from_indices(10, vec![2, 5, 9]);
        let filtered = patches
            .filter(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .unwrap();

        assert_arrays_eq!(filtered.indices(), PrimitiveArray::from_iter([0u64, 1])); // Adjusted indices
        assert_arrays_eq!(filtered.values(), PrimitiveArray::from_iter([100i32, 200]));
    }

    #[test]
    fn test_slice_full_range() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        let sliced = patches.slice(0..10).unwrap().unwrap();

        assert_arrays_eq!(sliced.indices(), PrimitiveArray::from_iter([2u64, 5, 8]));
        assert_arrays_eq!(
            sliced.values(),
            PrimitiveArray::from_iter([100i32, 200, 300])
        );
    }

    #[test]
    fn test_slice_partial() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        // Slice from 3 to 8 (includes patch at 5)
        let sliced = patches.slice(3..8).unwrap().unwrap();

        assert_arrays_eq!(sliced.indices(), PrimitiveArray::from_iter([5u64])); // Index stays the same
        assert_arrays_eq!(sliced.values(), PrimitiveArray::from_iter([200i32]));
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
            None,
        )
        .unwrap();

        // Slice from 6 to 7 (no patches in this range)
        let sliced = patches.slice(6..7).unwrap();
        assert!(sliced.is_none());
    }

    #[test]
    fn test_slice_with_offset() {
        let patches = Patches::new(
            10,
            5,                                  // offset
            buffer![7u64, 10, 13].into_array(), // actual indices are 2, 5, 8
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        // Slice from 3 to 8 (includes patch at actual index 5)
        let sliced = patches.slice(3..8).unwrap().unwrap();

        assert_arrays_eq!(sliced.indices(), PrimitiveArray::from_iter([10u64])); // Index stays the same (offset + 5 = 10)
        assert_arrays_eq!(sliced.values(), PrimitiveArray::from_iter([200i32]));
        assert_eq!(sliced.offset(), 8); // New offset = 5 + 3
    }

    #[test]
    fn test_patch_values() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        #[expect(deprecated)]
        let values = patches.values().to_primitive();
        assert_eq!(
            i32::try_from(
                &values
                    .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap()
            )
            .unwrap(),
            100i32
        );
        assert_eq!(
            i32::try_from(
                &values
                    .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap()
            )
            .unwrap(),
            200i32
        );
        assert_eq!(
            i32::try_from(
                &values
                    .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap()
            )
            .unwrap(),
            300i32
        );
    }

    #[test]
    fn test_indices_range() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        assert_eq!(patches.min_index().unwrap(), 2);
        assert_eq!(patches.max_index().unwrap(), 8);
    }

    #[test]
    fn test_search_index() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        // Search for exact indices
        assert_eq!(patches.search_index(2).unwrap(), SearchResult::Found(0));
        assert_eq!(patches.search_index(5).unwrap(), SearchResult::Found(1));
        assert_eq!(patches.search_index(8).unwrap(), SearchResult::Found(2));

        // Search for non-patch indices
        assert_eq!(patches.search_index(0).unwrap(), SearchResult::NotFound(0));
        assert_eq!(patches.search_index(3).unwrap(), SearchResult::NotFound(1));
        assert_eq!(patches.search_index(6).unwrap(), SearchResult::NotFound(2));
        assert_eq!(patches.search_index(9).unwrap(), SearchResult::NotFound(3));
    }

    #[test]
    fn test_mask_boundary_patches() {
        // Test masking patches at array boundaries
        let patches = Patches::new(
            10,
            0,
            buffer![0u64, 9].into_array(),
            buffer![100i32, 200].into_array(),
            None,
        )
        .unwrap();

        let mask = Mask::from_iter([
            true, false, false, false, false, false, false, false, false, false,
        ]);
        let masked = patches
            .mask(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        assert!(masked.is_some());
        let masked = masked.unwrap();
        assert_arrays_eq!(masked.indices(), PrimitiveArray::from_iter([9u64]));
        assert_arrays_eq!(masked.values(), PrimitiveArray::from_iter([200i32]));
    }

    #[test]
    fn test_mask_all_patches_removed() {
        // Test when all patches are masked out
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        // Mask that removes all patches
        let mask = Mask::from_iter([
            false, false, true, false, false, true, false, false, true, false,
        ]);
        let masked = patches
            .mask(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
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
            None,
        )
        .unwrap();

        // Mask that doesn't affect any patches
        let mask = Mask::from_iter([
            true, false, false, true, false, false, true, false, false, true,
        ]);
        let masked = patches
            .mask(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .unwrap();

        assert_arrays_eq!(masked.indices(), PrimitiveArray::from_iter([2u64, 5, 8]));
        assert_arrays_eq!(
            masked.values(),
            PrimitiveArray::from_iter([100i32, 200, 300])
        );
    }

    #[test]
    fn test_mask_single_patch() {
        // Test with a single patch
        let patches = Patches::new(
            5,
            0,
            buffer![2u64].into_array(),
            buffer![42i32].into_array(),
            None,
        )
        .unwrap();

        // Mask that removes the single patch
        let mask = Mask::from_iter([false, false, true, false, false]);
        let masked = patches
            .mask(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        assert!(masked.is_none());

        // Mask that keeps the single patch
        let mask = Mask::from_iter([true, false, false, true, false]);
        let masked = patches
            .mask(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .unwrap();
        assert_arrays_eq!(masked.indices(), PrimitiveArray::from_iter([2u64]));
    }

    #[test]
    fn test_mask_contiguous_patches() {
        // Test with contiguous patches
        let patches = Patches::new(
            10,
            0,
            buffer![3u64, 4, 5, 6].into_array(),
            buffer![100i32, 200, 300, 400].into_array(),
            None,
        )
        .unwrap();

        // Mask that removes middle patches
        let mask = Mask::from_iter([
            false, false, false, false, true, true, false, false, false, false,
        ]);
        let masked = patches
            .mask(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .unwrap();

        assert_arrays_eq!(masked.indices(), PrimitiveArray::from_iter([3u64, 6]));
        assert_arrays_eq!(masked.values(), PrimitiveArray::from_iter([100i32, 400]));
    }

    #[test]
    fn test_mask_with_large_offset() {
        // Test with a large offset that shifts all indices
        let patches = Patches::new(
            20,
            15,
            buffer![16u64, 17, 19].into_array(), // actual indices are 1, 2, 4
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        // Mask that removes the patch at actual index 2
        let mask = Mask::from_iter([
            false, false, true, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false,
        ]);
        let masked = patches
            .mask(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .unwrap();

        assert_arrays_eq!(masked.indices(), PrimitiveArray::from_iter([16u64, 19]));
        assert_arrays_eq!(masked.values(), PrimitiveArray::from_iter([100i32, 300]));
    }

    #[test]
    #[should_panic(expected = "Filter mask length 5 does not match array length 10")]
    fn test_mask_wrong_length() {
        let patches = Patches::new(
            10,
            0,
            buffer![2u64, 5, 8].into_array(),
            buffer![100i32, 200, 300].into_array(),
            None,
        )
        .unwrap();

        // Mask with wrong length
        let mask = Mask::from_iter([false, false, true, false, false]);
        patches
            .mask(&mask, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
    }

    #[test]
    fn test_chunk_offsets_search() {
        let indices = buffer![100u64, 200, 3000, 3100].into_array();
        let values = buffer![10i32, 20, 30, 40].into_array();
        let chunk_offsets = buffer![0u64, 2, 2, 3].into_array();
        let patches = Patches::new(4096, 0, indices, values, Some(chunk_offsets)).unwrap();

        assert!(patches.chunk_offsets.is_some());

        // chunk 0: patches at 100, 200
        assert_eq!(patches.search_index(100).unwrap(), SearchResult::Found(0));
        assert_eq!(patches.search_index(200).unwrap(), SearchResult::Found(1));

        // chunks 1, 2: no patches
        assert_eq!(
            patches.search_index(1500).unwrap(),
            SearchResult::NotFound(2)
        );
        assert_eq!(
            patches.search_index(2000).unwrap(),
            SearchResult::NotFound(2)
        );

        // chunk 3: patches at 3000, 3100
        assert_eq!(patches.search_index(3000).unwrap(), SearchResult::Found(2));
        assert_eq!(patches.search_index(3100).unwrap(), SearchResult::Found(3));

        assert_eq!(
            patches.search_index(1024).unwrap(),
            SearchResult::NotFound(2)
        );
    }

    #[test]
    fn test_chunk_offsets_with_slice() {
        let indices = buffer![100u64, 500, 1200, 1300, 1500, 1800, 2100, 2500].into_array();
        let values = buffer![10i32, 20, 30, 35, 40, 45, 50, 60].into_array();
        let chunk_offsets = buffer![0u64, 2, 6].into_array();
        let patches = Patches::new(3000, 0, indices, values, Some(chunk_offsets)).unwrap();

        let sliced = patches.slice(1000..2200).unwrap().unwrap();

        assert!(sliced.chunk_offsets.is_some());
        assert_eq!(sliced.offset(), 1000);

        assert_eq!(sliced.search_index(200).unwrap(), SearchResult::Found(0));
        assert_eq!(sliced.search_index(500).unwrap(), SearchResult::Found(2));
        assert_eq!(sliced.search_index(1100).unwrap(), SearchResult::Found(4));

        assert_eq!(sliced.search_index(250).unwrap(), SearchResult::NotFound(1));
    }

    #[test]
    fn test_chunk_offsets_with_slice_after_first_chunk() {
        let indices = buffer![100u64, 500, 1200, 1300, 1500, 1800, 2100, 2500].into_array();
        let values = buffer![10i32, 20, 30, 35, 40, 45, 50, 60].into_array();
        let chunk_offsets = buffer![0u64, 2, 6].into_array();
        let patches = Patches::new(3000, 0, indices, values, Some(chunk_offsets)).unwrap();

        let sliced = patches.slice(1300..2200).unwrap().unwrap();

        assert!(sliced.chunk_offsets.is_some());
        assert_eq!(sliced.offset(), 1300);

        assert_eq!(sliced.search_index(0).unwrap(), SearchResult::Found(0));
        assert_eq!(sliced.search_index(200).unwrap(), SearchResult::Found(1));
        assert_eq!(sliced.search_index(500).unwrap(), SearchResult::Found(2));
        assert_eq!(sliced.search_index(250).unwrap(), SearchResult::NotFound(2));
        assert_eq!(sliced.search_index(900).unwrap(), SearchResult::NotFound(4));
    }

    #[test]
    fn test_chunk_offsets_slice_empty_result() {
        let indices = buffer![100u64, 200, 3000, 3100].into_array();
        let values = buffer![10i32, 20, 30, 40].into_array();
        let chunk_offsets = buffer![0u64, 2].into_array();
        let patches = Patches::new(4000, 0, indices, values, Some(chunk_offsets)).unwrap();

        let sliced = patches.slice(1000..2000).unwrap();
        assert!(sliced.is_none());
    }

    #[test]
    fn test_chunk_offsets_slice_single_patch() {
        let indices = buffer![100u64, 1200, 1300, 2500].into_array();
        let values = buffer![10i32, 20, 30, 40].into_array();
        let chunk_offsets = buffer![0u64, 1, 3].into_array();
        let patches = Patches::new(3000, 0, indices, values, Some(chunk_offsets)).unwrap();

        let sliced = patches.slice(1100..1250).unwrap().unwrap();

        assert_eq!(sliced.num_patches(), 1);
        assert_eq!(sliced.offset(), 1100);
        assert_eq!(sliced.search_index(100).unwrap(), SearchResult::Found(0)); // 1200 - 1100 = 100
        assert_eq!(sliced.search_index(50).unwrap(), SearchResult::NotFound(0));
        assert_eq!(sliced.search_index(150).unwrap(), SearchResult::NotFound(1));
    }

    #[test]
    fn test_chunk_offsets_slice_across_chunks() {
        let indices = buffer![100u64, 200, 1100, 1200, 2100, 2200].into_array();
        let values = buffer![10i32, 20, 30, 40, 50, 60].into_array();
        let chunk_offsets = buffer![0u64, 2, 4].into_array();
        let patches = Patches::new(3000, 0, indices, values, Some(chunk_offsets)).unwrap();

        let sliced = patches.slice(150..2150).unwrap().unwrap();

        assert_eq!(sliced.num_patches(), 4);
        assert_eq!(sliced.offset(), 150);

        assert_eq!(sliced.search_index(50).unwrap(), SearchResult::Found(0)); // 200 - 150 = 50
        assert_eq!(sliced.search_index(950).unwrap(), SearchResult::Found(1)); // 1100 - 150 = 950
        assert_eq!(sliced.search_index(1050).unwrap(), SearchResult::Found(2)); // 1200 - 150 = 1050
        assert_eq!(sliced.search_index(1950).unwrap(), SearchResult::Found(3)); // 2100 - 150 = 1950
    }

    #[test]
    fn test_chunk_offsets_boundary_searches() {
        let indices = buffer![1023u64, 1024, 1025, 2047, 2048].into_array();
        let values = buffer![10i32, 20, 30, 40, 50].into_array();
        let chunk_offsets = buffer![0u64, 1, 4].into_array();
        let patches = Patches::new(3000, 0, indices, values, Some(chunk_offsets)).unwrap();

        assert_eq!(patches.search_index(1023).unwrap(), SearchResult::Found(0));
        assert_eq!(patches.search_index(1024).unwrap(), SearchResult::Found(1));
        assert_eq!(patches.search_index(1025).unwrap(), SearchResult::Found(2));
        assert_eq!(patches.search_index(2047).unwrap(), SearchResult::Found(3));
        assert_eq!(patches.search_index(2048).unwrap(), SearchResult::Found(4));

        assert_eq!(
            patches.search_index(1022).unwrap(),
            SearchResult::NotFound(0)
        );
        assert_eq!(
            patches.search_index(2046).unwrap(),
            SearchResult::NotFound(3)
        );
    }

    #[test]
    fn test_chunk_offsets_slice_edge_cases() {
        let indices = buffer![0u64, 1, 1023, 1024, 2047, 2048].into_array();
        let values = buffer![10i32, 20, 30, 40, 50, 60].into_array();
        let chunk_offsets = buffer![0u64, 3, 5].into_array();
        let patches = Patches::new(3000, 0, indices, values, Some(chunk_offsets)).unwrap();

        // Slice at the very beginning
        let sliced = patches.slice(0..10).unwrap().unwrap();
        assert_eq!(sliced.num_patches(), 2);
        assert_eq!(sliced.search_index(0).unwrap(), SearchResult::Found(0));
        assert_eq!(sliced.search_index(1).unwrap(), SearchResult::Found(1));

        // Slice at the very end
        let sliced = patches.slice(2040..3000).unwrap().unwrap();
        assert_eq!(sliced.num_patches(), 2); // patches at 2047 and 2048
        assert_eq!(sliced.search_index(7).unwrap(), SearchResult::Found(0)); // 2047 - 2040
        assert_eq!(sliced.search_index(8).unwrap(), SearchResult::Found(1)); // 2048 - 2040
    }

    #[test]
    fn test_chunk_offsets_slice_nested() {
        let indices = buffer![100u64, 200, 300, 400, 500, 600].into_array();
        let values = buffer![10i32, 20, 30, 40, 50, 60].into_array();
        let chunk_offsets = buffer![0u64].into_array();
        let patches = Patches::new(1000, 0, indices, values, Some(chunk_offsets)).unwrap();

        let sliced1 = patches.slice(150..550).unwrap().unwrap();
        assert_eq!(sliced1.num_patches(), 4); // 200, 300, 400, 500

        let sliced2 = sliced1.slice(100..250).unwrap().unwrap();
        assert_eq!(sliced2.num_patches(), 1); // 300
        assert_eq!(sliced2.offset(), 250);

        assert_eq!(sliced2.search_index(50).unwrap(), SearchResult::Found(0)); // 300 - 250
        assert_eq!(
            sliced2.search_index(150).unwrap(),
            SearchResult::NotFound(1)
        );
    }

    #[test]
    fn test_nested_slice_with_dropped_first_chunk() {
        // PATCH_CHUNK_SIZE = 1024, so the two patches land in different chunks.
        let indices = buffer![0u64, 1024].into_array();
        let values = buffer![1i32, 2].into_array();
        let chunk_offsets = buffer![0u64, 1].into_array();
        let patches = Patches::new(2048, 0, indices, values, Some(chunk_offsets)).unwrap();

        // Drop chunk 0, then re-slice the result.
        let dropped_first = patches.slice(1024..2048).unwrap().unwrap();
        let resliced = dropped_first.slice(0..1024).unwrap().unwrap();
        assert_eq!(resliced.num_patches(), 1);
    }

    #[test]
    fn test_index_larger_than_length() {
        let chunk_offsets = buffer![0u64].into_array();
        let indices = buffer![1023u64].into_array();
        let values = buffer![42i32].into_array();
        let patches = Patches::new(1024, 0, indices, values, Some(chunk_offsets)).unwrap();
        assert_eq!(
            patches.search_index(2048).unwrap(),
            SearchResult::NotFound(1)
        );
    }
}
