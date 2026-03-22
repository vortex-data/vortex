// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! First-class chunked arrays.
//!
//! Vortex is a chunked array library that's able to

use std::fmt::Debug;

use futures::stream;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::DynArray;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::iter::ArrayIterator;
use crate::iter::ArrayIteratorAdapter;
use crate::search_sorted::SearchSorted;
use crate::search_sorted::SearchSortedSide;
use crate::stats::ArrayStats;
use crate::stream::ArrayStream;
use crate::stream::ArrayStreamAdapter;
use crate::validity::Validity;

#[derive(Clone, Debug)]
pub struct ChunkedArray {
    pub(super) dtype: DType,
    pub(super) len: usize,
    pub(super) chunk_offsets: PrimitiveArray,
    pub(super) chunks: Vec<ArrayRef>,
    pub(super) stats_set: ArrayStats,
}

impl ChunkedArray {
    /// Constructs a new `ChunkedArray`.
    ///
    /// See [`ChunkedArray::new_unchecked`] for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented in
    /// [`ChunkedArray::new_unchecked`].
    pub fn try_new(chunks: Vec<ArrayRef>, dtype: DType) -> VortexResult<Self> {
        Self::validate(&chunks, &dtype)?;

        // SAFETY: validation done above.
        unsafe { Ok(Self::new_unchecked(chunks, dtype)) }
    }

    /// Creates a new [`ChunkedArray`] without validation from these components:
    ///
    /// * `chunks` is a vector of arrays to be concatenated logically.
    /// * `dtype` is the common data type of all chunks.
    ///
    /// # Safety
    ///
    /// All chunks must have exactly the same [`DType`] as the provided `dtype`.
    pub unsafe fn new_unchecked(chunks: Vec<ArrayRef>, dtype: DType) -> Self {
        #[cfg(debug_assertions)]
        Self::validate(&chunks, &dtype)
            .vortex_expect("[Debug Assertion]: Invalid `ChunkedArray` parameters");

        let nchunks = chunks.len();

        let mut chunk_offsets_buf = BufferMut::<u64>::with_capacity(nchunks + 1);
        // SAFETY: nchunks + 1
        unsafe { chunk_offsets_buf.push_unchecked(0) }
        let mut curr_offset = 0;
        for c in &chunks {
            curr_offset += c.len() as u64;
            // SAFETY: nchunks + 1
            unsafe { chunk_offsets_buf.push_unchecked(curr_offset) }
        }

        let chunk_offsets = PrimitiveArray::new(chunk_offsets_buf.freeze(), Validity::NonNullable);

        Self {
            dtype,
            len: curr_offset
                .try_into()
                .vortex_expect("chunk offset must fit in usize"),
            chunk_offsets,
            chunks,
            stats_set: Default::default(),
        }
    }

    /// Validates the components that would be used to create a [`ChunkedArray`].
    ///
    /// This function checks all the invariants required by [`ChunkedArray::new_unchecked`].
    pub fn validate(chunks: &[ArrayRef], dtype: &DType) -> VortexResult<()> {
        for chunk in chunks {
            if chunk.dtype() != dtype {
                vortex_bail!(MismatchedTypes: dtype, chunk.dtype());
            }
        }

        Ok(())
    }

    #[inline]
    pub fn chunk(&self, idx: usize) -> &ArrayRef {
        assert!(idx < self.nchunks(), "chunk index {idx} out of bounds");
        // SAFETY: bounds checked by the assert above.
        unsafe { self.chunks.get_unchecked(idx) }
    }

    pub fn nchunks(&self) -> usize {
        self.chunks.len()
    }

    #[inline]
    pub fn chunk_offsets(&self) -> Buffer<u64> {
        self.chunk_offsets.to_buffer()
    }

    pub(crate) fn find_chunk_idx(&self, index: usize) -> VortexResult<(usize, usize)> {
        assert!(index <= self.len(), "Index out of bounds of the array");
        let index = index as u64;

        // Since there might be duplicate values in offsets because of empty chunks we want to search from right
        // and take the last chunk (we subtract 1 since there's a leading 0)
        let index_chunk = self
            .chunk_offsets()
            .search_sorted(&index, SearchSortedSide::Right)?
            .to_ends_index(self.nchunks() + 1)
            .saturating_sub(1);
        let chunk_start = self.chunk_offsets()[index_chunk];

        let index_in_chunk =
            usize::try_from(index - chunk_start).vortex_expect("Index is too large for usize");
        Ok((index_chunk, index_in_chunk))
    }

    pub fn chunks(&self) -> &[ArrayRef] {
        &self.chunks
    }

    pub fn non_empty_chunks(&self) -> impl Iterator<Item = &ArrayRef> + '_ {
        self.chunks().iter().filter(|c| !c.is_empty())
    }

    pub fn array_iterator(&self) -> impl ArrayIterator + '_ {
        ArrayIteratorAdapter::new(self.dtype().clone(), self.chunks().iter().cloned().map(Ok))
    }

    pub fn array_stream(&self) -> impl ArrayStream + '_ {
        ArrayStreamAdapter::new(
            self.dtype().clone(),
            stream::iter(self.chunks().iter().cloned().map(Ok)),
        )
    }

    pub fn rechunk(&self, target_bytesize: u64, target_rowsize: usize) -> VortexResult<Self> {
        let mut new_chunks = Vec::new();
        let mut chunks_to_combine = Vec::new();
        let mut new_chunk_n_bytes = 0;
        let mut new_chunk_n_elements = 0;
        for chunk in self.chunks() {
            let n_bytes = chunk.nbytes();
            let n_elements = chunk.len();

            if (new_chunk_n_bytes + n_bytes > target_bytesize
                || new_chunk_n_elements + n_elements > target_rowsize)
                && !chunks_to_combine.is_empty()
            {
                new_chunks.push(
                    // SAFETY: chunks_to_combine contains valid chunks of the same dtype as self.
                    // All chunks are guaranteed to be valid arrays matching self.dtype().
                    unsafe {
                        ChunkedArray::new_unchecked(chunks_to_combine, self.dtype().clone())
                            .to_canonical()?
                            .into_array()
                    },
                );

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
            new_chunks.push(unsafe {
                // SAFETY: chunks_to_combine contains valid chunks of the same dtype as self.
                // All chunks are guaranteed to be valid arrays matching self.dtype().
                ChunkedArray::new_unchecked(chunks_to_combine, self.dtype().clone())
                    .to_canonical()?
                    .into_array()
            });
        }

        // SAFETY: new_chunks contains valid arrays of the same dtype as self.
        // All chunks were either taken from self or created from self's chunks.
        unsafe { Ok(Self::new_unchecked(new_chunks, self.dtype().clone())) }
    }
}

impl FromIterator<ArrayRef> for ChunkedArray {
    fn from_iter<T: IntoIterator<Item = ArrayRef>>(iter: T) -> Self {
        let chunks: Vec<ArrayRef> = iter.into_iter().collect();
        let dtype = chunks
            .first()
            .map(|c| c.dtype().clone())
            .vortex_expect("Cannot infer DType from an empty iterator");
        Self::try_new(chunks, dtype).vortex_expect("Failed to create chunked array from iterator")
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::array::DynArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    #[test]
    fn test_rechunk_one_chunk() {
        let chunked = ChunkedArray::try_new(
            vec![buffer![0u64].into_array()],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();

        let rechunked = chunked.rechunk(1 << 16, 1 << 16).unwrap();

        assert_arrays_eq!(chunked, rechunked);
    }

    #[test]
    fn test_rechunk_two_chunks() {
        let chunked = ChunkedArray::try_new(
            vec![buffer![0u64].into_array(), buffer![5u64].into_array()],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();

        let rechunked = chunked.rechunk(1 << 16, 1 << 16).unwrap();

        assert_eq!(rechunked.nchunks(), 1);
        assert_arrays_eq!(chunked, rechunked);
    }

    #[test]
    fn test_rechunk_tiny_target_chunks() {
        let chunked = ChunkedArray::try_new(
            vec![
                buffer![0u64, 1, 2, 3].into_array(),
                buffer![4u64, 5].into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();

        let rechunked = chunked.rechunk(1 << 16, 5).unwrap();

        assert_eq!(rechunked.nchunks(), 2);
        assert!(rechunked.chunks().iter().all(|c| c.len() < 5));
        assert_arrays_eq!(chunked, rechunked);
    }

    #[test]
    fn test_rechunk_with_too_big_chunk() {
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

        let rechunked = chunked.rechunk(1 << 16, 5).unwrap();
        // greedy so should be: [0, 1, 2] [42, 42, 42, 42, 42, 42] [4, 5, 6, 7] [8, 9]

        assert_eq!(rechunked.nchunks(), 4);
        assert_arrays_eq!(chunked, rechunked);
    }

    #[test]
    fn test_empty_chunks_all_valid() -> VortexResult<()> {
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
        assert!(chunked.all_valid().unwrap());
        assert!(!chunked.all_invalid().unwrap());

        Ok(())
    }

    #[test]
    fn test_empty_chunks_all_invalid() -> VortexResult<()> {
        // Create chunks where some are empty but all non-empty chunks have all invalid values
        let chunks = vec![
            PrimitiveArray::new(buffer![1u64, 2], Validity::AllInvalid).into_array(),
            PrimitiveArray::new(buffer![0u64; 0], Validity::AllInvalid).into_array(), // empty chunk
            PrimitiveArray::new(buffer![3u64, 4, 5], Validity::AllInvalid).into_array(),
            PrimitiveArray::new(buffer![0u64; 0], Validity::AllInvalid).into_array(), // empty chunk
        ];

        let chunked =
            ChunkedArray::try_new(chunks, DType::Primitive(PType::U64, Nullability::Nullable))?;

        // Should be all_invalid since all non-empty chunks are all_invalid
        assert!(!chunked.all_valid().unwrap());
        assert!(chunked.all_invalid().unwrap());

        Ok(())
    }

    #[test]
    fn test_empty_chunks_mixed_validity() -> VortexResult<()> {
        // Create chunks with mixed validity including empty chunks
        let chunks = vec![
            PrimitiveArray::new(buffer![1u64, 2], Validity::AllValid).into_array(),
            PrimitiveArray::new(buffer![0u64; 0], Validity::AllValid).into_array(), // empty chunk
            PrimitiveArray::new(buffer![3u64, 4], Validity::AllInvalid).into_array(),
            PrimitiveArray::new(buffer![0u64; 0], Validity::AllInvalid).into_array(), // empty chunk
        ];

        let chunked =
            ChunkedArray::try_new(chunks, DType::Primitive(PType::U64, Nullability::Nullable))?;

        // Should be neither all_valid nor all_invalid
        assert!(!chunked.all_valid().unwrap());
        assert!(!chunked.all_invalid().unwrap());

        Ok(())
    }
}
