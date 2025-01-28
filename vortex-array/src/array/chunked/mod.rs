//! First-class chunked arrays.
//!
//! Vortex is a chunked array library that's able to

use std::fmt::{Debug, Display};

use futures_util::stream;
use itertools::Itertools;
use rkyv::{access, to_bytes};
use serde::{Deserialize, Serialize};
use vortex_buffer::BufferMut;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{vortex_bail, vortex_panic, VortexExpect as _, VortexResult, VortexUnwrap};

use crate::array::primitive::PrimitiveArray;
use crate::compute::{scalar_at, search_sorted_usize, SearchSortedSide};
use crate::encoding::{ids, EncodingVTable};
use crate::iter::{ArrayIterator, ArrayIteratorAdapter};
use crate::stats::StatsSet;
use crate::stream::{ArrayStream, ArrayStreamAdapter};
use crate::validate::ValidateVTable;
use crate::validity::Validity::NonNullable;
use crate::validity::{ArrayValidity, LogicalValidity, Validity, ValidityVTable};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, DeserializeMetadata, IntoArrayData,
    IntoCanonical, RkyvMetadata,
};

mod canonical;
mod compute;
mod stats;
mod variants;

impl_encoding!(
    "vortex.chunked",
    ids::CHUNKED,
    Chunked,
    RkyvMetadata<ChunkedMetadata>
);

#[derive(
    Clone, Debug, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct ChunkedMetadata {
    pub(crate) nchunks: usize,
}

impl Display for ChunkedMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl ChunkedArray {
    const ENDS_DTYPE: DType = DType::Primitive(PType::U64, Nullability::NonNullable);

    pub fn try_new(chunks: Vec<ArrayData>, dtype: DType) -> VortexResult<Self> {
        for chunk in &chunks {
            if chunk.dtype() != &dtype {
                vortex_bail!(MismatchedTypes: dtype, chunk.dtype());
            }
        }

        let mut chunk_offsets = BufferMut::<u64>::with_capacity(chunks.len() + 1);
        chunk_offsets.extend(
            [0u64]
                .into_iter()
                .chain(chunks.iter().map(|c| c.len() as u64))
                .scan(0, |acc, c| {
                    *acc += c;
                    Some(*acc)
                }),
        );

        let nchunks = chunk_offsets.len() - 1;
        let length = *chunk_offsets
            .last()
            .vortex_expect("Chunk ends is guaranteed to have at least one element");

        let mut children = Vec::with_capacity(chunks.len() + 1);
        children.push(PrimitiveArray::new(chunk_offsets, NonNullable).into_array());
        children.extend(chunks);

        Self::try_from_parts(
            dtype,
            length.try_into().vortex_unwrap(),
            RkyvMetadata(ChunkedMetadata { nchunks }),
            None,
            Some(children.into()),
            StatsSet::default(),
        )
    }

    #[inline]
    pub fn chunk(&self, idx: usize) -> VortexResult<ArrayData> {
        if idx >= self.nchunks() {
            vortex_bail!("chunk index {} > num chunks ({})", idx, self.nchunks());
        }

        let chunk_offsets = self.chunk_offsets();
        let chunk_start = usize::try_from(&scalar_at(&chunk_offsets, idx)?)?;
        let chunk_end = usize::try_from(&scalar_at(&chunk_offsets, idx + 1)?)?;

        // Offset the index since chunk_ends is child 0.
        self.as_ref()
            .child(idx + 1, self.as_ref().dtype(), chunk_end - chunk_start)
    }

    pub fn nchunks(&self) -> usize {
        self.metadata().nchunks
    }

    #[inline]
    pub fn chunk_offsets(&self) -> ArrayData {
        self.as_ref()
            .child(0, &Self::ENDS_DTYPE, self.nchunks() + 1)
            .vortex_expect("Missing chunk ends in ChunkedArray")
    }

    fn find_chunk_idx(&self, index: usize) -> (usize, usize) {
        assert!(index <= self.len(), "Index out of bounds of the array");

        // Since there might be duplicate values in offsets because of empty chunks we want to search from right
        // and take the last chunk (we subtract 1 since there's a leading 0)
        let index_chunk =
            search_sorted_usize(&self.chunk_offsets(), index, SearchSortedSide::Right)
                .vortex_expect("Search sorted failed in find_chunk_idx")
                .to_ends_index(self.nchunks() + 1)
                .saturating_sub(1);
        let chunk_start = scalar_at(self.chunk_offsets(), index_chunk)
            .and_then(|s| usize::try_from(&s))
            .vortex_expect("Failed to find chunk start in find_chunk_idx");

        let index_in_chunk = index - chunk_start;
        (index_chunk, index_in_chunk)
    }

    pub fn chunks(&self) -> impl Iterator<Item = ArrayData> + '_ {
        (0..self.nchunks()).map(|c| {
            self.chunk(c).unwrap_or_else(|e| {
                vortex_panic!(
                    e,
                    "ChunkedArray: chunks: chunk {} should exist (num chunks: {})",
                    c,
                    self.nchunks()
                )
            })
        })
    }

    pub fn array_iterator(&self) -> impl ArrayIterator + '_ {
        ArrayIteratorAdapter::new(self.dtype().clone(), self.chunks().map(Ok))
    }

    pub fn array_stream(&self) -> impl ArrayStream + '_ {
        ArrayStreamAdapter::new(self.dtype().clone(), stream::iter(self.chunks().map(Ok)))
    }

    pub fn rechunk(&self, target_bytesize: usize, target_rowsize: usize) -> VortexResult<Self> {
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
                    ChunkedArray::try_new(chunks_to_combine, self.dtype().clone())?
                        .into_canonical()?
                        .into(),
                );

                new_chunk_n_bytes = 0;
                new_chunk_n_elements = 0;
                chunks_to_combine = Vec::new();
            }

            if n_bytes > target_bytesize || n_elements > target_rowsize {
                new_chunks.push(chunk);
            } else {
                new_chunk_n_bytes += n_bytes;
                new_chunk_n_elements += n_elements;
                chunks_to_combine.push(chunk);
            }
        }

        if !chunks_to_combine.is_empty() {
            new_chunks.push(
                ChunkedArray::try_new(chunks_to_combine, self.dtype().clone())?
                    .into_canonical()?
                    .into(),
            );
        }

        Self::try_new(new_chunks, self.dtype().clone())
    }
}

impl ValidateVTable<ChunkedArray> for ChunkedEncoding {}

impl FromIterator<ArrayData> for ChunkedArray {
    fn from_iter<T: IntoIterator<Item = ArrayData>>(iter: T) -> Self {
        let chunks: Vec<ArrayData> = iter.into_iter().collect();
        let dtype = chunks
            .first()
            .map(|c| c.dtype().clone())
            .vortex_expect("Cannot infer DType from an empty iterator");
        Self::try_new(chunks, dtype).vortex_expect("Failed to create chunked array from iterator")
    }
}

impl VisitorVTable<ChunkedArray> for ChunkedEncoding {
    fn accept(&self, array: &ChunkedArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("chunk_ends", &array.chunk_offsets())?;
        for (idx, chunk) in array.chunks().enumerate() {
            visitor.visit_child(format!("chunks[{}]", idx).as_str(), &chunk)?;
        }
        Ok(())
    }
}

impl ValidityVTable<ChunkedArray> for ChunkedEncoding {
    fn is_valid(&self, array: &ChunkedArray, index: usize) -> VortexResult<bool> {
        let (chunk, offset_in_chunk) = array.find_chunk_idx(index);
        array.chunk(chunk)?.is_valid(offset_in_chunk)
    }

    fn logical_validity(&self, array: &ChunkedArray) -> VortexResult<LogicalValidity> {
        // TODO(ngates): implement FromIterator<LogicalValidity> for LogicalValidity.
        let validity: Validity = array.chunks().map(|a| a.logical_validity()).try_collect()?;
        validity.to_logical(array.len())
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_error::VortexResult;

    use crate::array::chunked::ChunkedArray;
    use crate::compute::test_harness::test_binary_numeric;
    use crate::compute::{scalar_at, sub_scalar, try_cast};
    use crate::{assert_arrays_eq, ArrayDType, IntoArrayData, IntoArrayVariant};

    fn chunked_array() -> ChunkedArray {
        ChunkedArray::try_new(
            vec![
                buffer![1u64, 2, 3].into_array(),
                buffer![4u64, 5, 6].into_array(),
                buffer![7u64, 8, 9].into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap()
    }

    #[test]
    fn test_scalar_subtract() {
        let chunked = chunked_array().into_array();
        let to_subtract = 1u64;
        let array = sub_scalar(&chunked, to_subtract.into()).unwrap();

        let chunked = ChunkedArray::try_from(array).unwrap();
        let mut chunks_out = chunked.chunks();

        let results = chunks_out
            .next()
            .unwrap()
            .into_primitive()
            .unwrap()
            .as_slice::<u64>()
            .to_vec();
        assert_eq!(results, &[0u64, 1, 2]);
        let results = chunks_out
            .next()
            .unwrap()
            .into_primitive()
            .unwrap()
            .as_slice::<u64>()
            .to_vec();
        assert_eq!(results, &[3u64, 4, 5]);
        let results = chunks_out
            .next()
            .unwrap()
            .into_primitive()
            .unwrap()
            .as_slice::<u64>()
            .to_vec();
        assert_eq!(results, &[6u64, 7, 8]);
    }

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
        assert!(rechunked.chunks().all(|c| c.len() < 5));
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
    fn test_chunked_binary_numeric() {
        let array = chunked_array().into_array();
        // The tests test both X - 1 and 1 - X, so we need signed values
        let signed_dtype = DType::from(PType::try_from(array.dtype()).unwrap().to_signed());
        let array = try_cast(array, &signed_dtype).unwrap();
        test_binary_numeric::<u64>(array)
    }
}
