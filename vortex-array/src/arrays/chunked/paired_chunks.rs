// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::ChunkedArray;

/// A pre-sliced, aligned pair of array chunks from two `ChunkedArray`s.
pub(crate) struct AlignedPair {
    pub left: ArrayRef,
    pub right: ArrayRef,
    pub pos: Range<usize>,
}

/// An iterator that walks two equally-sized `ChunkedArray`s in lockstep,
/// yielding aligned `(left, right)` slices at every chunk boundary of either
/// input. Empty chunks are skipped automatically.
pub(crate) struct PairedChunks<'a> {
    left: &'a ChunkedArray,
    right: &'a ChunkedArray,
    lhs_idx: usize,
    rhs_idx: usize,
    lhs_offset: usize,
    rhs_offset: usize,
    pos: usize,
    total_len: usize,
}

impl ChunkedArray {
    /// Returns an iterator that walks `self` and `other` in lockstep, yielding
    /// [`AlignedPair`]s sliced at every chunk boundary of either input.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != other.len()`.
    pub(crate) fn paired_chunks<'a>(&'a self, other: &'a ChunkedArray) -> PairedChunks<'a> {
        assert_eq!(
            self.len(),
            other.len(),
            "paired_chunks requires arrays of equal length"
        );
        PairedChunks {
            left: self,
            right: other,
            lhs_idx: 0,
            rhs_idx: 0,
            lhs_offset: 0,
            rhs_offset: 0,
            pos: 0,
            total_len: self.len(),
        }
    }
}

impl Iterator for PairedChunks<'_> {
    type Item = VortexResult<AlignedPair>;

    fn next(&mut self) -> Option<Self::Item> {
        // Skip empty chunks on either side.
        while self.lhs_idx < self.left.nchunks() && self.left.chunk(self.lhs_idx).is_empty() {
            self.lhs_idx += 1;
        }
        while self.rhs_idx < self.right.nchunks() && self.right.chunk(self.rhs_idx).is_empty() {
            self.rhs_idx += 1;
        }

        if self.pos >= self.total_len {
            return None;
        }

        let lhs_chunk = self.left.chunk(self.lhs_idx);
        let rhs_chunk = self.right.chunk(self.rhs_idx);

        let lhs_rem = lhs_chunk.len() - self.lhs_offset;
        let rhs_rem = rhs_chunk.len() - self.rhs_offset;
        let take = lhs_rem.min(rhs_rem);

        let lhs_slice = match lhs_chunk.slice(self.lhs_offset..self.lhs_offset + take) {
            Ok(s) => s,
            Err(e) => return Some(Err(e)),
        };
        let rhs_slice = match rhs_chunk.slice(self.rhs_offset..self.rhs_offset + take) {
            Ok(s) => s,
            Err(e) => return Some(Err(e)),
        };

        let start = self.pos;
        self.pos += take;
        self.lhs_offset += take;
        self.rhs_offset += take;

        if self.lhs_offset == lhs_chunk.len() {
            self.lhs_idx += 1;
            self.lhs_offset = 0;
        }
        if self.rhs_offset == rhs_chunk.len() {
            self.rhs_idx += 1;
            self.rhs_offset = 0;
        }

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
    use crate::arrays::ChunkedArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    fn i32_dtype() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    #[allow(clippy::type_complexity)]
    fn collect_pairs(
        left: &ChunkedArray,
        right: &ChunkedArray,
    ) -> VortexResult<Vec<(Vec<i32>, Vec<i32>, std::ops::Range<usize>)>> {
        use crate::ToCanonical;
        let mut result = Vec::new();
        for pair in left.paired_chunks(right) {
            let pair = pair?;
            let l: Vec<i32> = pair.left.to_primitive().as_slice::<i32>().to_vec();
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
        // Left:  [1,2] [3] [4,5]  →  boundaries at 0,2,3,5
        // Right: [10]  [20,30] [40,50]  →  boundaries at 0,1,3,5
        // Aligned at: 0,1,2,3,5
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

        // Should panic.
        drop(left.paired_chunks(&right).collect::<Vec<_>>());
    }
}
