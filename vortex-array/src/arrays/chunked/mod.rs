//! First-class chunked arrays.
//!
//! Vortex is a chunked array library that's able to

use std::any::Any;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use arrow_array::builder::ArrayBuilder;
use futures_util::stream;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{vortex_bail, vortex_panic, VortexExpect as _, VortexResult, VortexUnwrap};
use vortex_mask::Mask;

use crate::arrays::primitive::PrimitiveArray;
use crate::compute::{scalar_at, search_sorted_usize, SearchSortedSide};
use crate::encoding::encoding_ids;
use crate::iter::{ArrayIterator, ArrayIteratorAdapter};
use crate::stats::StatsSet;
use crate::stream::{ArrayStream, ArrayStreamAdapter};
use crate::validity::Validity;
use crate::validity::Validity::NonNullable;
use crate::variants::PrimitiveArrayTrait;
use crate::visitor::ArrayVisitor;
use crate::{
    impl_encoding, Array, ArrayCanonicalImpl, ArrayImpl, ArrayRef, ArrayValidityImpl,
    ArrayVariantsImpl, Canonical, EmptyMetadata, Encoding, EncodingId, IntoArray, IntoCanonical,
    RkyvMetadata,
};

mod canonical;
// mod compute;
// // mod stats;
mod variants;

#[derive(Clone)]
pub struct ChunkedArray {
    dtype: DType,
    len: usize,
    chunk_offsets: Buffer<u64>,
    chunks: Vec<ArrayRef>,
    stats: StatsSet,
}

pub struct ChunkedEncoding;
impl Encoding for ChunkedEncoding {
    const ID: EncodingId = EncodingId("vortex.chunked", encoding_ids::CHUNKED);
    type Array = ChunkedArray;
    type Metadata = EmptyMetadata;
}

impl ChunkedArray {
    const ENDS_DTYPE: DType = DType::Primitive(PType::U64, Nullability::NonNullable);

    pub fn try_new(chunks: Vec<ArrayRef>, dtype: DType) -> VortexResult<Self> {
        for chunk in &chunks {
            if chunk.dtype() != &dtype {
                vortex_bail!(MismatchedTypes: dtype, chunk.dtype());
            }
        }

        Ok(Self::try_new_unchecked(chunks, dtype))
    }

    pub fn try_new_unchecked(chunks: Vec<ArrayRef>, dtype: DType) -> Self {
        let nchunks = chunks.len();

        let mut chunk_offsets = BufferMut::<u64>::with_capacity(nchunks + 1);
        unsafe { chunk_offsets.push_unchecked(0) }
        let mut curr_offset = 0u64;
        for c in &chunks {
            curr_offset += c.len() as u64;
            unsafe { chunk_offsets.push_unchecked(curr_offset) }
        }

        Self {
            dtype,
            len: curr_offset.try_into().vortex_unwrap(),
            chunk_offsets: chunk_offsets.freeze(),
            chunks,
            stats: Default::default(),
        }
    }

    #[inline]
    pub fn chunk(&self, idx: usize) -> VortexResult<ArrayRef> {
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
        self.chunks.len()
    }

    #[inline]
    pub fn chunk_offsets(&self) -> ArrayRef {
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

    pub fn chunks(&self) -> &[ArrayRef] {
        &self.chunks
    }

    pub fn non_empty_chunks(&self) -> impl Iterator<Item = ArrayRef> + '_ {
        self.chunks().filter(|c| !c.is_empty())
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
                    ChunkedArray::try_new_unchecked(chunks_to_combine, self.dtype().clone())
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
                ChunkedArray::try_new_unchecked(chunks_to_combine, self.dtype().clone())
                    .into_array()
                    .into_canonical()?
                    .into(),
            );
        }

        Ok(Self::try_new_unchecked(new_chunks, self.dtype().clone()))
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

impl ArrayCanonicalImpl for ChunkedArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        todo!()
    }

    fn _append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        todo!()
    }
}

impl ArrayValidityImpl for ChunkedArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        todo!()
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        todo!()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        todo!()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        todo!()
    }
}

impl ArrayVariantsImpl for ChunkedArray {}

impl ArrayImpl for ChunkedArray {
    fn _len(&self) -> usize {
        self.len
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
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

    fn all_valid(&self, array: &ChunkedArray) -> VortexResult<bool> {
        for chunk in array.chunks() {
            if !chunk.all_valid()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn all_invalid(&self, array: &ChunkedArray) -> VortexResult<bool> {
        for chunk in array.chunks() {
            if !chunk.all_invalid()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn validity_mask(&self, array: &ChunkedArray) -> VortexResult<Mask> {
        // TODO(ngates): implement FromIterator<LogicalValidity> for LogicalValidity.
        let validity: Validity = array.chunks().map(|a| a.validity_mask()).try_collect()?;
        validity.to_logical(array.len())
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_error::VortexResult;

    use crate::arrays::chunked::ChunkedArray;
    use crate::compute::test_harness::test_binary_numeric;
    use crate::compute::{scalar_at, sub_scalar, try_cast};
    use crate::{assert_arrays_eq, IntoArray, IntoArrayVariant};

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
