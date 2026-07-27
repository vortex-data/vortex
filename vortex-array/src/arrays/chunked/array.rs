// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! First-class chunked arrays.
//!
//! Vortex is a chunked array library that's able to

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

use futures::stream;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::arrays::Chunked;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::iter::ArrayIterator;
use crate::iter::ArrayIteratorAdapter;
use crate::search_sorted::SearchSorted;
use crate::search_sorted::SearchSortedSide;
use crate::stream::ArrayStream;
use crate::stream::ArrayStreamAdapter;
use crate::validity::Validity;

/// Slot layout of a [`Chunked`] array: `[chunk_offsets, chunks...]`.
#[array_slots(Chunked)]
pub struct ChunkedSlots {
    /// The non-nullable `u64` array of cumulative chunk offsets.
    pub chunk_offsets: ArrayRef,
    /// The chunk arrays, each sharing the outer dtype.
    pub chunks: Vec<ArrayRef>,
}

pub(super) const CHUNK_OFFSETS_SLOT: usize = ChunkedSlots::CHUNK_OFFSETS;
pub(super) const CHUNKS_OFFSET: usize = ChunkedSlots::CHUNKS_OFFSET;

#[derive(Clone, Debug)]
pub struct ChunkedData {
    pub(super) chunk_offsets: Vec<usize>,
    /// This is used to find the next child to execute when in executing into a builder.
    pub(super) next_builder_slot: usize,
}

impl Display for ChunkedData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "nchunks: {}", self.chunk_offsets.len().saturating_sub(1))
    }
}

pub trait ChunkedArrayExt: TypedArrayRef<Chunked> {
    fn chunk_offsets_array(&self) -> &ArrayRef {
        self.as_ref().slots()[CHUNK_OFFSETS_SLOT]
            .as_ref()
            .vortex_expect("validated chunk offsets slot")
    }

    fn nchunks(&self) -> usize {
        self.as_ref().slots().len().saturating_sub(CHUNKS_OFFSET)
    }

    fn chunk(&self, idx: usize) -> &ArrayRef {
        self.as_ref().slots()[CHUNKS_OFFSET + idx]
            .as_ref()
            .vortex_expect("validated chunk slot")
    }

    fn iter_chunks<'a>(&'a self) -> Box<dyn Iterator<Item = &'a ArrayRef> + 'a> {
        Box::new(
            self.as_ref().slots()[CHUNKS_OFFSET..]
                .iter()
                .map(|slot| slot.as_ref().vortex_expect("validated chunk slot")),
        )
    }

    fn chunks(&self) -> Vec<ArrayRef> {
        self.iter_chunks().cloned().collect()
    }

    fn non_empty_chunks<'a>(&'a self) -> Box<dyn Iterator<Item = &'a ArrayRef> + 'a> {
        Box::new(self.iter_chunks().filter(|chunk| !chunk.is_empty()))
    }

    /// Returns the cached chunk boundary offsets.
    fn chunk_offset_values(&self) -> &[usize] {
        &self.chunk_offsets
    }

    fn find_chunk_idx(&self, index: usize) -> VortexResult<(usize, usize)> {
        assert!(
            index <= self.as_ref().len(),
            "Index out of bounds of the array"
        );
        let chunk_offset_values = self.chunk_offset_values();
        let index_chunk = chunk_offset_values
            .search_sorted(&index, SearchSortedSide::Right)?
            .to_ends_index(self.nchunks() + 1)
            .saturating_sub(1);
        let chunk_start = chunk_offset_values[index_chunk];
        let index_in_chunk = index - chunk_start;
        Ok((index_chunk, index_in_chunk))
    }

    fn array_iterator(&self) -> impl ArrayIterator + '_ {
        ArrayIteratorAdapter::new(
            self.as_ref().dtype().clone(),
            self.iter_chunks().map(|chunk| Ok(chunk.clone())),
        )
    }

    fn array_stream(&self) -> impl ArrayStream + '_ {
        ArrayStreamAdapter::new(
            self.as_ref().dtype().clone(),
            stream::iter(self.iter_chunks().map(|chunk| Ok(chunk.clone()))),
        )
    }
}
impl<T: TypedArrayRef<Chunked>> ChunkedArrayExt for T {}

impl ChunkedData {
    pub(super) fn new(chunk_offsets: Vec<usize>) -> Self {
        Self {
            chunk_offsets,
            next_builder_slot: CHUNKS_OFFSET,
        }
    }

    pub(super) fn make_chunk_offsets_array(chunk_offsets: &[usize]) -> ArrayRef {
        let mut chunk_offsets_buf = BufferMut::<u64>::with_capacity(chunk_offsets.len());
        for &offset in chunk_offsets {
            let offset = u64::try_from(offset)
                .vortex_expect("chunk offset must fit in u64 for serialization");
            unsafe { chunk_offsets_buf.push_unchecked(offset) }
        }
        PrimitiveArray::new(chunk_offsets_buf.freeze(), Validity::NonNullable).into_array()
    }

    /// Validates the components that would be used to create a `ChunkedArray`.
    ///
    /// This function checks all the invariants required by `ChunkedArray::new_unchecked`.
    pub fn validate(chunks: &[ArrayRef], dtype: &DType) -> VortexResult<()> {
        for chunk in chunks {
            if chunk.dtype() != dtype {
                vortex_bail!(MismatchedTypes: dtype, chunk.dtype());
            }
        }

        Ok(())
    }
}

impl Array<Chunked> {
    fn parts_from_chunks<const VALIDATE: bool>(
        chunks: impl IntoIterator<Item = ArrayRef>,
        dtype: DType,
    ) -> VortexResult<ArrayParts<Chunked>> {
        let chunks = chunks.into_iter();
        let (lower, _) = chunks.size_hint();
        let mut slots = ArraySlots::with_capacity(CHUNKS_OFFSET + lower);
        slots.push(None);
        let mut chunk_offsets = Vec::with_capacity(lower + 1);
        chunk_offsets.push(0);
        let mut len = 0usize;

        for chunk in chunks {
            if VALIDATE && chunk.dtype() != &dtype {
                vortex_bail!(MismatchedTypes: &dtype, chunk.dtype());
            }
            len += chunk.len();
            chunk_offsets.push(len);
            slots.push(Some(chunk));
        }

        slots[CHUNK_OFFSETS_SLOT] = Some(ChunkedData::make_chunk_offsets_array(&chunk_offsets));
        Ok(ArrayParts::new(Chunked, dtype, len, ChunkedData::new(chunk_offsets)).with_slots(slots))
    }

    pub(super) fn with_next_builder_slot(mut self, next_builder_slot: usize) -> Self {
        if let Some(data) = self.data_mut() {
            data.next_builder_slot = next_builder_slot;
            return self;
        }
        // This is the slow path that will be hit at most once per execution since the second one
        // *MUST* have execlusive access due to this copy.
        let stats = self.statistics().to_owned();
        let mut data = self.data().clone();
        data.next_builder_slot = next_builder_slot;
        // SAFETY: we only modified next_builder_slot which doesn't affect array invariants.
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Chunked, self.dtype().clone(), self.len(), data)
                    .with_slots(self.slots().iter().cloned().collect::<ArraySlots>()),
            )
        }
        .with_stats_set(stats)
    }

    /// Constructs a new `ChunkedArray`.
    pub fn try_new(chunks: impl IntoIterator<Item = ArrayRef>, dtype: DType) -> VortexResult<Self> {
        Array::try_from_parts(Self::parts_from_chunks::<true>(chunks, dtype)?)
    }

    pub fn rechunk(
        &self,
        target_bytesize: u64,
        target_rowsize: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let mut new_chunks = Vec::new();
        let mut chunks_to_combine = Vec::new();
        let mut new_chunk_n_bytes = 0;
        let mut new_chunk_n_elements = 0;
        for chunk in self.iter_chunks() {
            let n_bytes = chunk.nbytes();
            let n_elements = chunk.len();

            if (new_chunk_n_bytes + n_bytes > target_bytesize
                || new_chunk_n_elements + n_elements > target_rowsize)
                && !chunks_to_combine.is_empty()
            {
                let canonical = unsafe {
                    Array::<Chunked>::new_unchecked(chunks_to_combine, self.dtype().clone())
                }
                .into_array()
                .execute::<Canonical>(ctx)?
                .into_array();
                new_chunks.push(canonical);

                new_chunk_n_bytes = 0;
                new_chunk_n_elements = 0;
                chunks_to_combine = Vec::new();
            }

            if n_bytes > target_bytesize || n_elements > target_rowsize {
                new_chunks.push(chunk.clone());
            } else {
                new_chunk_n_bytes += n_bytes;
                new_chunk_n_elements += n_elements;
                chunks_to_combine.push(chunk.clone());
            }
        }

        if !chunks_to_combine.is_empty() {
            let canonical =
                unsafe { Array::<Chunked>::new_unchecked(chunks_to_combine, self.dtype().clone()) }
                    .into_array()
                    .execute::<Canonical>(ctx)?
                    .into_array();
            new_chunks.push(canonical);
        }

        unsafe { Ok(Self::new_unchecked(new_chunks, self.dtype().clone())) }
    }

    /// Creates a new `ChunkedArray` without validation.
    ///
    /// # Safety
    ///
    /// All chunks must have exactly the same [`DType`] as the provided `dtype`.
    pub unsafe fn new_unchecked(chunks: impl IntoIterator<Item = ArrayRef>, dtype: DType) -> Self {
        let parts = Self::parts_from_chunks::<true>(chunks, dtype)
            .vortex_expect("unchecked chunked construction cannot fail");
        unsafe { Array::from_parts_unchecked(parts) }
    }
}

impl FromIterator<ArrayRef> for Array<Chunked> {
    fn from_iter<T: IntoIterator<Item = ArrayRef>>(iter: T) -> Self {
        let chunks: Vec<ArrayRef> = iter.into_iter().collect();
        let dtype = chunks
            .first()
            .map(|c| c.dtype().clone())
            .vortex_expect("Cannot infer DType from an empty iterator");
        Array::<Chunked>::try_new(chunks, dtype)
            .vortex_expect("Failed to create chunked array from iterator")
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::chunked::ChunkedArrayExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    #[test]
    fn test_rechunk_one_chunk() {
        let mut ctx = array_session().create_execution_ctx();
        let chunked = ChunkedArray::try_new(
            vec![buffer![0u64].into_array()],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();

        let rechunked = chunked.rechunk(1 << 16, 1 << 16, &mut ctx).unwrap();

        assert_arrays_eq!(chunked, rechunked, &mut ctx);
    }

    #[test]
    fn test_rechunk_two_chunks() {
        let mut ctx = array_session().create_execution_ctx();
        let chunked = ChunkedArray::try_new(
            vec![buffer![0u64].into_array(), buffer![5u64].into_array()],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();

        let rechunked = chunked.rechunk(1 << 16, 1 << 16, &mut ctx).unwrap();

        assert_eq!(rechunked.nchunks(), 1);
        assert_arrays_eq!(chunked, rechunked, &mut ctx);
    }

    #[test]
    fn test_rechunk_tiny_target_chunks() {
        let mut ctx = array_session().create_execution_ctx();
        let chunked = ChunkedArray::try_new(
            vec![
                buffer![0u64, 1, 2, 3].into_array(),
                buffer![4u64, 5].into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();

        let rechunked = chunked.rechunk(1 << 16, 5, &mut ctx).unwrap();

        assert_eq!(rechunked.nchunks(), 2);
        assert!(rechunked.iter_chunks().all(|c| c.len() < 5));
        assert_arrays_eq!(chunked, rechunked, &mut ctx);
    }

    #[test]
    fn test_rechunk_with_too_big_chunk() {
        let mut ctx = array_session().create_execution_ctx();
        let chunked = ChunkedArray::try_new(
            vec![
                buffer![0u64, 1, 2].into_array(),
                buffer![42_u64; 6].into_array(),
                buffer![4u64, 5].into_array(),
                buffer![6u64, 7].into_array(),
                buffer![8u64, 9].into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();

        let rechunked = chunked.rechunk(1 << 16, 5, &mut ctx).unwrap();
        // greedy so should be: [0, 1, 2] [42, 42, 42, 42, 42, 42] [4, 5, 6, 7] [8, 9]

        assert_eq!(rechunked.nchunks(), 4);
        assert_arrays_eq!(chunked, rechunked, &mut ctx);
    }

    #[test]
    fn test_empty_chunks_all_valid() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // Create chunks where some are empty but all non-empty chunks have all valid values
        let chunks = vec![
            PrimitiveArray::new(buffer![1u64, 2, 3], Validity::AllValid).into_array(),
            PrimitiveArray::new(buffer![0u64; 0], Validity::AllValid).into_array(), // empty chunk
            PrimitiveArray::new(buffer![4u64, 5], Validity::AllValid).into_array(),
            PrimitiveArray::new(buffer![0u64; 0], Validity::AllValid).into_array(), // empty chunk
        ];

        let chunked =
            ChunkedArray::try_new(chunks, DType::Primitive(PType::U64, Nullability::Nullable))?;

        // Should be all_valid since all non-empty chunks are all_valid
        assert!(chunked.all_valid(&mut ctx)?);
        assert!(!chunked.into_array().all_invalid(&mut ctx)?);

        Ok(())
    }

    #[test]
    fn test_empty_chunks_all_invalid() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // Create chunks where some are empty but all non-empty chunks have all invalid values
        let chunks = vec![
            PrimitiveArray::new(buffer![1u64, 2], Validity::AllInvalid).into_array(),
            PrimitiveArray::new(buffer![0u64; 0], Validity::AllInvalid).into_array(), /* empty chunk */
            PrimitiveArray::new(buffer![3u64, 4, 5], Validity::AllInvalid).into_array(),
            PrimitiveArray::new(buffer![0u64; 0], Validity::AllInvalid).into_array(), /* empty chunk */
        ];

        let chunked =
            ChunkedArray::try_new(chunks, DType::Primitive(PType::U64, Nullability::Nullable))?;

        // Should be all_invalid since all non-empty chunks are all_invalid
        assert!(!chunked.all_valid(&mut ctx)?);
        assert!(chunked.into_array().all_invalid(&mut ctx)?);

        Ok(())
    }

    #[test]
    fn test_empty_chunks_mixed_validity() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // Create chunks with mixed validity including empty chunks
        let chunks = vec![
            PrimitiveArray::new(buffer![1u64, 2], Validity::AllValid).into_array(),
            PrimitiveArray::new(buffer![0u64; 0], Validity::AllValid).into_array(), // empty chunk
            PrimitiveArray::new(buffer![3u64, 4], Validity::AllInvalid).into_array(),
            PrimitiveArray::new(buffer![0u64; 0], Validity::AllInvalid).into_array(), /* empty chunk */
        ];

        let chunked =
            ChunkedArray::try_new(chunks, DType::Primitive(PType::U64, Nullability::Nullable))?;

        // Should be neither all_valid nor all_invalid
        assert!(!chunked.all_valid(&mut ctx)?);
        assert!(!chunked.into_array().all_invalid(&mut ctx)?);

        Ok(())
    }
}
