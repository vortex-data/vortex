//! First-class chunked arrays.
//!
//! Vortex is a chunked array library that's able to

use std::fmt::Debug;

use futures_util::stream;
use itertools::Itertools;
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::DType;
use vortex_error::{VortexExpect as _, VortexResult, VortexUnwrap, vortex_bail};
use vortex_mask::Mask;

use crate::array::ArrayValidityImpl;
use crate::compute::{ComputeFn, InvocationArgs, Output, SearchSorted, SearchSortedSide};
use crate::iter::{ArrayIterator, ArrayIteratorAdapter};
use crate::nbytes::NBytes;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::stream::{ArrayStream, ArrayStreamAdapter};
use crate::vtable::VTableRef;
use crate::{Array, ArrayImpl, ArrayRef, ArrayStatisticsImpl, EmptyMetadata, Encoding, IntoArray};

mod canonical;
mod compute;
mod ops;
mod serde;
mod variants;

#[derive(Clone, Debug)]
pub struct ChunkedArray {
    dtype: DType,
    len: usize,
    chunk_offsets: Buffer<u64>,
    chunks: Vec<ArrayRef>,
    stats_set: ArrayStats,
}

#[derive(Debug)]
pub struct ChunkedEncoding;
impl Encoding for ChunkedEncoding {
    type Array = ChunkedArray;
    type Metadata = EmptyMetadata;
}

impl ChunkedArray {
    pub fn try_new(chunks: Vec<ArrayRef>, dtype: DType) -> VortexResult<Self> {
        for chunk in &chunks {
            if chunk.dtype() != &dtype {
                vortex_bail!(MismatchedTypes: dtype, chunk.dtype());
            }
        }

        Ok(Self::new_unchecked(chunks, dtype))
    }

    pub fn new_unchecked(chunks: Vec<ArrayRef>, dtype: DType) -> Self {
        let nchunks = chunks.len();

        let mut chunk_offsets = BufferMut::<u64>::with_capacity(nchunks + 1);
        unsafe { chunk_offsets.push_unchecked(0) }
        let mut curr_offset = 0;
        for c in &chunks {
            curr_offset += c.len() as u64;
            unsafe { chunk_offsets.push_unchecked(curr_offset) }
        }
        assert_eq!(chunk_offsets.len(), nchunks + 1);

        Self {
            dtype,
            len: curr_offset.try_into().vortex_unwrap(),
            chunk_offsets: chunk_offsets.freeze(),
            chunks,
            stats_set: Default::default(),
        }
    }

    // TODO(ngates): remove result
    #[inline]
    pub fn chunk(&self, idx: usize) -> VortexResult<&ArrayRef> {
        if idx >= self.nchunks() {
            vortex_bail!("chunk index {} > num chunks ({})", idx, self.nchunks());
        }
        Ok(&self.chunks[idx])
    }

    pub fn nchunks(&self) -> usize {
        self.chunks.len()
    }

    #[inline]
    pub fn chunk_offsets(&self) -> &[u64] {
        &self.chunk_offsets
    }

    fn find_chunk_idx(&self, index: usize) -> (usize, usize) {
        assert!(index <= self.len(), "Index out of bounds of the array");
        let index = index as u64;

        // Since there might be duplicate values in offsets because of empty chunks we want to search from right
        // and take the last chunk (we subtract 1 since there's a leading 0)
        let index_chunk = self
            .chunk_offsets()
            .search_sorted(&index, SearchSortedSide::Right)
            .to_ends_index(self.nchunks() + 1)
            .saturating_sub(1);
        let chunk_start = self.chunk_offsets()[index_chunk];

        let index_in_chunk =
            usize::try_from(index - chunk_start).vortex_expect("Index is too large for usize");
        (index_chunk, index_in_chunk)
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
                    ChunkedArray::new_unchecked(chunks_to_combine, self.dtype().clone())
                        .to_canonical()?
                        .into_array(),
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
            new_chunks.push(
                ChunkedArray::new_unchecked(chunks_to_combine, self.dtype().clone())
                    .to_canonical()?
                    .into_array(),
            );
        }

        Ok(Self::new_unchecked(new_chunks, self.dtype().clone()))
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

impl ArrayImpl for ChunkedArray {
    type Encoding = ChunkedEncoding;

    fn _len(&self) -> usize {
        self.len
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&ChunkedEncoding)
    }

    fn _with_children(&self, children: &[ArrayRef]) -> VortexResult<Self> {
        Ok(ChunkedArray::new_unchecked(
            // We skip the first child as it contains the offsets buffer.
            children[1..].to_vec(),
            self.dtype.clone(),
        ))
    }

    fn _invoke(
        &self,
        compute_fn: &ComputeFn,
        args: &InvocationArgs,
    ) -> VortexResult<Option<Output>> {
        if compute_fn.is_elementwise() {
            return self.invoke_elementwise(compute_fn, args);
        }
        Ok(None)
    }
}

impl ArrayStatisticsImpl for ChunkedArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayValidityImpl for ChunkedArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        if !self.dtype.is_nullable() {
            return Ok(true);
        }
        let (chunk, offset_in_chunk) = self.find_chunk_idx(index);
        self.chunk(chunk)?.is_valid(offset_in_chunk)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        if !self.dtype().is_nullable() {
            return Ok(true);
        }
        for chunk in self.chunks() {
            if !chunk.all_valid()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        if !self.dtype().is_nullable() {
            return Ok(false);
        }
        for chunk in self.chunks() {
            if !chunk.all_invalid()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.chunks()
            .iter()
            .map(|a| a.validity_mask())
            .try_collect()
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_error::VortexResult;

    use crate::array::Array;
    use crate::arrays::chunked::ChunkedArray;
    use crate::compute::conformance::binary_numeric::test_numeric;
    use crate::compute::{cast, sub_scalar};
    use crate::{ArrayExt, IntoArray, ToCanonical, assert_arrays_eq};

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

        let chunked = array.as_::<ChunkedArray>();
        let chunks_out = chunked.chunks();

        let results = chunks_out[0]
            .to_primitive()
            .unwrap()
            .as_slice::<u64>()
            .to_vec();
        assert_eq!(results, &[0u64, 1, 2]);
        let results = chunks_out[1]
            .to_primitive()
            .unwrap()
            .as_slice::<u64>()
            .to_vec();
        assert_eq!(results, &[3u64, 4, 5]);
        let results = chunks_out[2]
            .to_primitive()
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
    fn test_chunked_binary_numeric() {
        let array = chunked_array();
        // The tests test both X - 1 and 1 - X, so we need signed values
        let signed_dtype = DType::from(PType::try_from(array.dtype()).unwrap().to_signed());
        let array = cast(&array, &signed_dtype).unwrap();
        test_numeric::<u64>(array)
    }
}
