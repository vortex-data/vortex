// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

pub use compress::rle_decompress;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, VTable, ValidityChild, ValidityChildSliceHelper,
    ValidityVTableFromChildSliceHelper,
};
use vortex_array::{
    Array, ArrayEq, ArrayHash, ArrayRef, Canonical, EncodingId, EncodingRef, Precision, vtable,
};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_ensure};

use crate::FL_CHUNK_SIZE;

mod compress;
mod compute;
mod ops;
mod serde;

vtable!(RLE);

impl VTable for RLEVTable {
    type Array = RLEArray;
    type Encoding = RLEEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChildSliceHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;
    type PipelineVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("fastlanes.rle")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(RLEEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct RLEArray {
    dtype: DType,
    /// Run value in the dictionary.
    values: ArrayRef,
    /// Chunk-local indices from all chunks. The start of each chunk is looked up in `values_idx_offsets`.
    indices: ArrayRef,
    /// Index start positions of each value chunk.
    ///
    /// # Example
    /// ```
    /// // Chunk 0: [10, 20] (starts at index 0)
    /// // Chunk 1: [30, 40] (starts at index 2)
    /// let values = [10, 20, 30, 40];           // Global values array
    /// let values_idx_offsets = [0, 2];         // Chunk 0 starts at index 0, Chunk 1 starts at index2
    /// ```
    values_idx_offsets: ArrayRef,

    stats_set: ArrayStats,
    // Offset relative to the start of the chunk.
    offset: usize,
    length: usize,
}

#[derive(Clone, Debug)]
pub struct RLEEncoding;

impl RLEArray {
    fn validate(
        values: &dyn Array,
        indices: &dyn Array,
        value_idx_offsets: &dyn Array,
        offset: usize,
    ) -> VortexResult<()> {
        vortex_ensure!(
            offset < 1024,
            "Offset must be smaller than 1024, got {}",
            offset
        );

        vortex_ensure!(
            values.dtype().is_primitive(),
            "RLE values must be a primitive type, got {}",
            values.dtype()
        );

        vortex_ensure!(
            matches!(indices.dtype().as_ptype(), PType::U8 | PType::U16),
            "RLE indices must be u8 or u16, got {}",
            indices.dtype()
        );

        vortex_ensure!(
            value_idx_offsets.dtype().is_unsigned_int() && !value_idx_offsets.dtype().is_nullable(),
            "RLE value idx offsets must be non-nullable unsigned integer, got {}",
            value_idx_offsets.dtype()
        );

        vortex_ensure!(
            indices.len().div_ceil(FL_CHUNK_SIZE) == value_idx_offsets.len(),
            "RLE must have one value idx offset per chunk, got {}",
            value_idx_offsets.len()
        );

        vortex_ensure!(
            indices.len() >= values.len(),
            "RLE must have at least as many indices as values, got {} indices and {} values",
            indices.len(),
            values.len()
        );

        Ok(())
    }

    /// Create a new chunk-based RLE array from its components.
    ///
    /// # Arguments
    ///
    /// * `values` - Unique values from all chunks
    /// * `indices` - Chunk-local indices from all chunks
    /// * `values_idx_offsets` - Start indices for each value chunk.
    /// * `offset` - Offset into the first chunk
    /// * `length` - Array length
    pub fn try_new(
        values: ArrayRef,
        indices: ArrayRef,
        values_idx_offsets: ArrayRef,
        offset: usize,
        length: usize,
    ) -> VortexResult<Self> {
        assert_eq!(indices.len() % FL_CHUNK_SIZE, 0);
        Self::validate(&values, &indices, &values_idx_offsets, offset)?;

        // Ensure that the DType has the same nullability as the indices array.
        let dtype = DType::Primitive(values.dtype().as_ptype(), indices.dtype().nullability());

        Ok(Self {
            dtype,
            values,
            indices,
            values_idx_offsets,
            stats_set: ArrayStats::default(),
            offset,
            length,
        })
    }

    /// Create a new RLEArray without validation.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `offset + length` does not exceed the length of the indices array
    /// - The `dtype` is consistent with the values array's primitive type and validity nullability
    /// - The `indices` array contains valid indices into chunks of the `values` array
    /// - The `values_idx_offsets` array contains valid chunk start offsets
    /// - The `validity` array has the same length as `length`
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_unchecked(
        values: ArrayRef,
        indices: ArrayRef,
        values_idx_offsets: ArrayRef,
        dtype: DType,
        offset: usize,
        length: usize,
    ) -> Self {
        Self {
            dtype,
            values,
            indices,
            values_idx_offsets,
            stats_set: ArrayStats::default(),
            offset,
            length,
        }
    }

    #[inline]
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }

    #[inline]
    pub fn indices(&self) -> &ArrayRef {
        &self.indices
    }

    #[inline]
    pub fn values_idx_offsets(&self) -> &ArrayRef {
        &self.values_idx_offsets
    }

    /// Values index offset relative to the first chunk.
    ///
    /// Offsets in `values_idx_offsets` are absolute and need to be shifted
    /// by the offset of the first chunk, respective the current slice, in
    /// order to make them relative.
    #[allow(clippy::expect_used)]
    pub(crate) fn values_idx_offset(&self, chunk_idx: usize) -> usize {
        self.values_idx_offsets
            .scalar_at(chunk_idx)
            .as_primitive()
            .as_::<usize>()
            .expect("index must be of type usize")
            - self
                .values_idx_offsets
                .scalar_at(0)
                .as_primitive()
                .as_::<usize>()
                .expect("index must be of type usize")
    }

    /// Index offset into the array
    pub fn offset(&self) -> usize {
        self.offset
    }
}

impl ValidityChild<RLEVTable> for RLEVTable {
    fn validity_child(array: &RLEArray) -> &dyn Array {
        array.indices().as_ref()
    }
}

impl ArrayVTable<RLEVTable> for RLEVTable {
    fn len(array: &RLEArray) -> usize {
        array.length
    }

    fn dtype(array: &RLEArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &RLEArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &RLEArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.values.array_hash(state, precision);
        array.indices.array_hash(state, precision);
        array.values_idx_offsets.array_hash(state, precision);
        array.offset.hash(state);
        array.length.hash(state);
    }

    fn array_eq(array: &RLEArray, other: &RLEArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.values.array_eq(&other.values, precision)
            && array.indices.array_eq(&other.indices, precision)
            && array
                .values_idx_offsets
                .array_eq(&other.values_idx_offsets, precision)
            && array.offset == other.offset
            && array.length == other.length
    }
}

impl CanonicalVTable<RLEVTable> for RLEVTable {
    fn canonicalize(array: &RLEArray) -> Canonical {
        Canonical::Primitive(rle_decompress(array))
    }
}

impl ValidityChildSliceHelper for RLEArray {
    fn unsliced_child_and_slice(&self) -> (&ArrayRef, usize, usize) {
        let (start, len) = (self.offset(), self.len());
        (self.indices(), start, start + len)
    }
}

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;

    use super::*;
    use crate::RLEArray;

    #[test]
    fn test_try_new() {
        let values = PrimitiveArray::from_iter([10u32, 20, 30]).into_array();

        // Pad indices to 1024 chunk.
        let indices =
            PrimitiveArray::from_iter([0u16, 0, 1, 1, 2].iter().cycle().take(1024).copied())
                .into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();
        let rle_array = RLEArray::try_new(values, indices, values_idx_offsets, 0, 5).unwrap();

        assert_eq!(rle_array.len(), 5);
        assert_eq!(rle_array.values.len(), 3);
        assert_eq!(rle_array.values.dtype().as_ptype(), PType::U32);
    }

    #[test]
    fn test_try_new_with_validity() {
        let values = PrimitiveArray::from_iter([10u32, 20]).into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();

        let indices_pattern = [0u16, 1, 0];
        let validity_pattern = [true, false, true];

        // Pad indices to 1024 chunk.
        let indices_with_validity = PrimitiveArray::new(
            indices_pattern
                .iter()
                .cycle()
                .take(1024)
                .copied()
                .collect::<Buffer<u16>>(),
            Validity::from_iter(validity_pattern.iter().cycle().take(1024).copied()),
        )
        .into_array();

        let rle_array = RLEArray::try_new(
            values.clone(),
            indices_with_validity,
            values_idx_offsets,
            0,
            3,
        )
        .unwrap();

        assert_eq!(rle_array.len(), 3);
        assert_eq!(rle_array.values.len(), 2);
        assert!(rle_array.is_valid(0));
        assert!(!rle_array.is_valid(1));
        assert!(rle_array.is_valid(2));
    }

    #[test]
    fn test_all_valid() {
        let values = PrimitiveArray::from_iter([10u32, 20, 30]).into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();

        let indices_pattern = [0u16, 1, 2, 0, 1];
        let validity_pattern = [true, true, true, false, false];

        // Pad indices to 1024 chunk.
        let indices_with_validity = PrimitiveArray::new(
            indices_pattern
                .iter()
                .cycle()
                .take(1024)
                .copied()
                .collect::<Buffer<u16>>(),
            Validity::from_iter(validity_pattern.iter().cycle().take(1024).copied()),
        )
        .into_array();

        let rle_array = RLEArray::try_new(
            values.clone(),
            indices_with_validity,
            values_idx_offsets,
            0,
            5,
        )
        .unwrap();

        let valid_slice = rle_array.slice(0..3);
        assert!(valid_slice.all_valid());

        let mixed_slice = rle_array.slice(1..5);
        assert!(!mixed_slice.all_valid());
    }

    #[test]
    fn test_all_invalid() {
        let values = PrimitiveArray::from_iter([10u32, 20, 30]).into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();

        // Pad indices to 1024 chunk.
        let indices_pattern = [0u16, 1, 2, 0, 1];
        let validity_pattern = [true, true, false, false, false];

        let indices_with_validity = PrimitiveArray::new(
            indices_pattern
                .iter()
                .cycle()
                .take(1024)
                .copied()
                .collect::<Buffer<u16>>(),
            Validity::from_iter(validity_pattern.iter().cycle().take(1024).copied()),
        )
        .into_array();

        let rle_array = RLEArray::try_new(
            values.clone(),
            indices_with_validity,
            values_idx_offsets,
            0,
            5,
        )
        .unwrap();

        let invalid_slice = rle_array.slice(2..5);
        assert!(invalid_slice.all_invalid());

        let mixed_slice = rle_array.slice(1..4);
        assert!(!mixed_slice.all_invalid());
    }

    #[test]
    fn test_validity_mask() {
        let values = PrimitiveArray::from_iter([10u32, 20, 30]).into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();

        // Pad indices to 1024 chunk.
        let indices_pattern = [0u16, 1, 2, 0];
        let validity_pattern = [true, false, true, false];

        let indices_with_validity = PrimitiveArray::new(
            indices_pattern
                .iter()
                .cycle()
                .take(1024)
                .copied()
                .collect::<Buffer<u16>>(),
            Validity::from_iter(validity_pattern.iter().cycle().take(1024).copied()),
        )
        .into_array();

        let rle_array = RLEArray::try_new(
            values.clone(),
            indices_with_validity,
            values_idx_offsets,
            0,
            4,
        )
        .unwrap();

        let sliced_array = rle_array.slice(1..4);
        let validity_mask = sliced_array.validity_mask();

        let expected_mask = Validity::from_iter([false, true, false]).to_mask(3);
        assert_eq!(validity_mask.len(), expected_mask.len());
        assert_eq!(validity_mask, expected_mask);
    }

    #[test]
    fn test_try_new_empty() {
        let values = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        let indices = PrimitiveArray::from_iter(Vec::<u16>::new()).into_array();
        let values_idx_offsets = PrimitiveArray::from_iter(Vec::<u64>::new()).into_array();
        let rle_array = RLEArray::try_new(
            values,
            indices.clone(),
            values_idx_offsets,
            0,
            indices.len(),
        )
        .unwrap();

        assert_eq!(rle_array.len(), 0);
        assert_eq!(rle_array.values.len(), 0);
    }

    #[test]
    fn test_multi_chunk_two_chunks() {
        let values = PrimitiveArray::from_iter([10u32, 20, 30, 40]).into_array();
        let indices = PrimitiveArray::from_iter([0u16, 1].repeat(1024)).into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([0u64, 2]).into_array();
        let rle_array = RLEArray::try_new(values, indices, values_idx_offsets, 0, 2048).unwrap();

        assert_eq!(rle_array.len(), 2048);
        assert_eq!(rle_array.values.len(), 4);

        assert_eq!(rle_array.values_idx_offset(0), 0);
        assert_eq!(rle_array.values_idx_offset(1), 2);
    }
}
