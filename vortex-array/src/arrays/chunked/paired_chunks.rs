// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::chunked::ChunkedArrayExt;

pub(crate) struct AlignedPair {
    pub left: ArrayRef,
    pub right: ArrayRef,
    pub pos: Range<usize>,
}

/// Cursor over a chunk list that maintains the invariant: `idx` always
/// points at a non-empty chunk or is past the end.
struct ChunkCursor {
    chunks: Vec<ArrayRef>,
    idx: usize,
    offset: usize,
}

impl ChunkCursor {
    fn new(chunks: Vec<ArrayRef>) -> Self {
        let mut cursor = Self {
            chunks,
            idx: 0,
            offset: 0,
        };
        cursor.skip_empty();
        cursor
    }

    fn skip_empty(&mut self) {
        while self.idx < self.chunks.len() && self.chunks[self.idx].is_empty() {
            self.idx += 1;
        }
    }

    fn is_exhausted(&self) -> bool {
        self.idx >= self.chunks.len()
    }

    fn remaining(&self) -> usize {
        self.chunks[self.idx].len() - self.offset
    }

    fn take(&mut self, n: usize) -> VortexResult<ArrayRef> {
        let chunk = &self.chunks[self.idx];
        let slice = chunk.slice(self.offset..self.offset + n)?;
        self.offset += n;
        if self.offset == chunk.len() {
            self.idx += 1;
            self.offset = 0;
            self.skip_empty();
        }
        Ok(slice)
    }
}

pub(crate) struct PairedChunks {
    left: ChunkCursor,
    right: ChunkCursor,
    pos: usize,
    total_len: usize,
}

pub(crate) trait PairedChunksExt: ChunkedArrayExt {
    fn paired_chunks<T: ChunkedArrayExt>(&self, other: &T) -> PairedChunks {
        assert_eq!(
            self.as_ref().len(),
            other.as_ref().len(),
            "paired_chunks requires arrays of equal length"
        );
        PairedChunks {
            left: ChunkCursor::new(self.chunks()),
            right: ChunkCursor::new(other.chunks()),
            pos: 0,
            total_len: self.as_ref().len(),
        }
    }
}

impl<T: ChunkedArrayExt> PairedChunksExt for T {}

impl Iterator for PairedChunks {
    type Item = VortexResult<AlignedPair>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.total_len || self.left.is_exhausted() || self.right.is_exhausted() {
            return None;
        }

        let take = self.left.remaining().min(self.right.remaining());

        let (lhs_slice, rhs_slice) = match self
            .left
            .take(take)
            .and_then(|l| self.right.take(take).map(|r| (l, r)))
        {
            Ok(pair) => pair,
            Err(e) => return Some(Err(e)),
        };

        let start = self.pos;
        self.pos += take;

        Some(Ok(AlignedPair {
            left: lhs_slice,
            right: rhs_slice,
            pos: start..self.pos,
        }))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    #[expect(deprecated)]
    use crate::ToCanonical as _;
    use crate::arrays::ChunkedArray;
    use crate::arrays::chunked::paired_chunks::PairedChunksExt;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    fn i32_dtype() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    #[expect(clippy::type_complexity)]
    fn collect_pairs(
        left: &ChunkedArray,
        right: &ChunkedArray,
    ) -> VortexResult<Vec<(Vec<i32>, Vec<i32>, std::ops::Range<usize>)>> {
        let mut result = Vec::new();
        for pair in left.paired_chunks(right) {
            let pair = pair?;
            #[expect(deprecated)]
            let l: Vec<i32> = pair.left.to_primitive().as_slice::<i32>().to_vec();
            #[expect(deprecated)]
            let r: Vec<i32> = pair.right.to_primitive().as_slice::<i32>().to_vec();
            result.push((l, r, pair.pos));
        }
        Ok(result)
    }

    #[test]
    fn test_aligned_chunks() -> VortexResult<()> {
        let left = ChunkedArray::try_new(
            vec![buffer![1i32, 2].into_array(), buffer![3i32, 4].into_array()],
            i32_dtype(),
        )?;
        let right = ChunkedArray::try_new(
            vec![
                buffer![10i32, 20].into_array(),
                buffer![30i32, 40].into_array(),
            ],
            i32_dtype(),
        )?;

        let pairs = collect_pairs(&left, &right)?;
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], (vec![1, 2], vec![10, 20], 0..2));
        assert_eq!(pairs[1], (vec![3, 4], vec![30, 40], 2..4));
        Ok(())
    }

    #[test]
    fn test_misaligned_chunks() -> VortexResult<()> {
        let left = ChunkedArray::try_new(
            vec![
                buffer![1i32, 2].into_array(),
                buffer![3i32].into_array(),
                buffer![4i32, 5].into_array(),
            ],
            i32_dtype(),
        )?;
        let right = ChunkedArray::try_new(
            vec![
                buffer![10i32].into_array(),
                buffer![20i32, 30].into_array(),
                buffer![40i32, 50].into_array(),
            ],
            i32_dtype(),
        )?;

        let pairs = collect_pairs(&left, &right)?;
        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[0], (vec![1], vec![10], 0..1));
        assert_eq!(pairs[1], (vec![2], vec![20], 1..2));
        assert_eq!(pairs[2], (vec![3], vec![30], 2..3));
        assert_eq!(pairs[3], (vec![4, 5], vec![40, 50], 3..5));
        Ok(())
    }

    #[test]
    fn test_empty_chunks() -> VortexResult<()> {
        let left = ChunkedArray::try_new(
            vec![
                buffer![0i32; 0].into_array(),
                buffer![1i32, 2, 3].into_array(),
            ],
            i32_dtype(),
        )?;
        let right = ChunkedArray::try_new(
            vec![
                buffer![10i32, 20, 30].into_array(),
                buffer![0i32; 0].into_array(),
            ],
            i32_dtype(),
        )?;

        let pairs = collect_pairs(&left, &right)?;
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (vec![1, 2, 3], vec![10, 20, 30], 0..3));
        Ok(())
    }

    #[test]
    fn test_single_element_chunks() -> VortexResult<()> {
        let left = ChunkedArray::try_new(
            vec![
                buffer![1i32].into_array(),
                buffer![2i32].into_array(),
                buffer![3i32].into_array(),
            ],
            i32_dtype(),
        )?;
        let right = ChunkedArray::try_new(vec![buffer![10i32, 20, 30].into_array()], i32_dtype())?;

        let pairs = collect_pairs(&left, &right)?;
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0], (vec![1], vec![10], 0..1));
        assert_eq!(pairs[1], (vec![2], vec![20], 1..2));
        assert_eq!(pairs[2], (vec![3], vec![30], 2..3));
        Ok(())
    }

    #[test]
    fn test_both_empty() -> VortexResult<()> {
        let left = ChunkedArray::try_new(vec![], i32_dtype())?;
        let right = ChunkedArray::try_new(vec![], i32_dtype())?;

        let pairs = collect_pairs(&left, &right)?;
        assert!(pairs.is_empty());
        Ok(())
    }

    #[test]
    #[should_panic(expected = "paired_chunks requires arrays of equal length")]
    fn test_length_mismatch_panics() {
        let left = ChunkedArray::try_new(vec![buffer![1i32, 2].into_array()], i32_dtype()).unwrap();
        let right =
            ChunkedArray::try_new(vec![buffer![10i32, 20, 30].into_array()], i32_dtype()).unwrap();

        left.paired_chunks(&right).for_each(drop);
    }
}
