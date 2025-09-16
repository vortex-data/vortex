// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

pub use compress::rle_decompress;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::validity::Validity;
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper,
};
use vortex_array::{Array, ArrayRef, Canonical, EncodingId, EncodingRef, vtable};
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_ensure};

use crate::FL_CHUNK_SIZE;

mod compress;
mod ops;
mod serde;

vtable!(RLE);

impl VTable for RLEVTable {
    type Array = RLEArray;
    type Encoding = RLEEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
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
    /// Chunk-local indices from all chunks. The start of each chunk is looked up in the `value_chunk_offsets`.
    indices: ArrayRef,
    /// Index start positions of each value chunk.
    ///
    /// # Example
    /// ```
    /// // Chunk 0: [10, 20] (starts at index 0)
    /// // Chunk 1: [30, 40] (starts at index 2)
    /// let values = [10, 20, 30, 40];           // Global values array
    /// let value_chunk_offsets = [0, 2];        // Chunk 0 starts at index 0, Chunk 1 starts at index2
    /// ```
    value_chunk_offsets: ArrayRef,
    validity: Validity,
    stats_set: ArrayStats,
    offset: usize,
    length: usize,
}

#[derive(Clone, Debug)]
pub struct RLEEncoding;

impl RLEArray {
    fn validate(
        values: &dyn Array,
        indices: &dyn Array,
        value_chunks_offsets: &dyn Array,
        validity: Validity,
    ) -> VortexResult<()> {
        if let Some(validity_length) = validity.maybe_len() {
            vortex_ensure!(
                validity_length == indices.len(),
                "RLE validity length must match indices length, got {} and {}",
                validity_length,
                values.len()
            );
        }

        vortex_ensure!(
            values.dtype().is_primitive(),
            "RLE values must be a primitive type, got {}",
            values.dtype()
        );

        vortex_ensure!(
            *indices.dtype() == DType::Primitive(PType::U16, NonNullable),
            "RLE indices must be non-nullable u16, got {}",
            indices.dtype()
        );

        vortex_ensure!(
            *value_chunks_offsets.dtype() == DType::Primitive(PType::U64, NonNullable),
            "RLE value chunk offsets must be non-nullable u64, got {}",
            value_chunks_offsets.dtype()
        );

        Ok(())
    }

    /// Create a new chunk-based RLE array from its components.
    ///
    /// # Arguments
    ///
    /// * `values` - Unique values from all chunks
    /// * `indices` - Chunk-local indices from all chunks
    /// * `value_chunk_offsets` - Start indices for each value chunk.
    /// * `validity` - Array validity
    /// * `length` - Array length
    pub fn try_new(
        values: ArrayRef,
        indices: ArrayRef,
        value_chunk_offsets: ArrayRef,
        validity: Validity,
        length: usize,
    ) -> VortexResult<Self> {
        Self::validate(&values, &indices, &value_chunk_offsets, validity.clone())?;
        let dtype = DType::Primitive(values.dtype().as_ptype(), validity.nullability());

        Ok(Self {
            dtype,
            values,
            indices,
            value_chunk_offsets,
            validity,
            stats_set: ArrayStats::default(),
            offset: 0,
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
    /// - The `value_chunk_offsets` array contains valid chunk start offsets
    /// - The `validity` array has the same length as `length`
    pub unsafe fn new_unchecked(
        values: ArrayRef,
        indices: ArrayRef,
        value_chunk_offsets: ArrayRef,
        validity: Validity,
        dtype: DType,
        offset: usize,
        length: usize,
    ) -> Self {
        Self {
            dtype,
            values,
            indices,
            value_chunk_offsets,
            validity,
            stats_set: ArrayStats::default(),
            offset,
            length,
        }
    }

    pub(crate) fn values(&self) -> &dyn Array {
        &self.values
    }

    pub(crate) fn indices(&self) -> &dyn Array {
        &self.indices
    }

    pub(crate) fn value_chunk_offsets(&self) -> &dyn Array {
        &self.value_chunk_offsets
    }

    pub fn ptype(&self) -> PType {
        self.dtype.as_ptype()
    }

    /// Returns the offset within a chunk for an absolute position.
    pub(crate) fn offset_in_chunk(&self, abs_position: usize) -> usize {
        abs_position & (FL_CHUNK_SIZE - 1) // Equivalent to % 1024
    }

    /// Start index offset in the values array for a given chunk.
    #[allow(clippy::expect_used)]
    pub(crate) fn value_chunk_offset(&self, chunk_idx: usize) -> usize {
        self.value_chunk_offsets
            .scalar_at(chunk_idx)
            .as_primitive()
            .as_::<usize>()
            .expect("index must be of type usize")
    }

    /// Returns the chunk index for an absolute scalar index.
    pub(crate) fn chunk_idx(&self, abs_position: usize) -> usize {
        abs_position / FL_CHUNK_SIZE
    }

    /// Index offset into the array
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }
}

impl ValidityHelper for RLEArray {
    fn validity(&self) -> &Validity {
        &self.validity
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
}

impl CanonicalVTable<RLEVTable> for RLEVTable {
    fn canonicalize(array: &RLEArray) -> Canonical {
        Canonical::Primitive(rle_decompress(array))
    }
}

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;

    use super::*;
    use crate::RLEArray;

    #[test]
    fn test_try_new_valid() {
        use vortex_array::arrays::PrimitiveArray;

        let values = PrimitiveArray::from_iter([10u32, 20, 30]).into_array();
        let indices = PrimitiveArray::from_iter([0u16, 0, 1, 1, 2]).into_array();
        let value_chunk_offsets = PrimitiveArray::from_iter([3u64]).into_array();
        let validity = Validity::NonNullable;

        let rle_array = RLEArray::try_new(
            values,
            indices.clone(),
            value_chunk_offsets,
            validity,
            indices.len(),
        )
        .unwrap();

        assert_eq!(rle_array.len(), 5);
        assert_eq!(rle_array.values.len(), 3);
        assert_eq!(rle_array.ptype(), PType::U32);
    }

    #[test]
    fn test_try_new_with_validity() {
        use vortex_array::arrays::PrimitiveArray;

        let values = PrimitiveArray::from_iter([10u32, 20]).into_array();
        let indices = PrimitiveArray::from_iter([0u16, 1, 0]).into_array();
        let value_chunk_offsets = PrimitiveArray::from_iter([2u64]).into_array();
        let validity = Validity::from_iter([true, false, true]);

        let rle_array = RLEArray::try_new(
            values,
            indices.clone(),
            value_chunk_offsets,
            validity,
            indices.len(),
        )
        .unwrap();

        assert_eq!(rle_array.len(), 3);
        assert_eq!(rle_array.values.len(), 2);
        assert!(rle_array.is_valid(0));
        assert!(!rle_array.is_valid(1));
        assert!(rle_array.is_valid(2));
    }

    #[test]
    fn test_try_new_empty() {
        use vortex_array::arrays::PrimitiveArray;

        let values = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        let indices = PrimitiveArray::from_iter(Vec::<u16>::new()).into_array();
        let value_chunk_offsets = PrimitiveArray::from_iter(Vec::<u64>::new()).into_array();
        let validity = Validity::NonNullable;

        let rle_array = RLEArray::try_new(
            values,
            indices.clone(),
            value_chunk_offsets,
            validity,
            indices.len(),
        )
        .unwrap();

        assert_eq!(rle_array.len(), 0);
        assert_eq!(rle_array.values.len(), 0);
    }

    #[test]
    fn test_multi_chunk_two_chunks() {
        use vortex_array::arrays::PrimitiveArray;

        let values = PrimitiveArray::from_iter([10u32, 20, 30, 40]).into_array();
        let indices = PrimitiveArray::from_iter([0u16, 1].repeat(1024)).into_array();

        let value_chunk_offsets = PrimitiveArray::from_iter([0u64, 2]).into_array();
        let validity = Validity::NonNullable;

        let rle_array =
            RLEArray::try_new(values, indices, value_chunk_offsets, validity, 2048).unwrap();

        assert_eq!(rle_array.len(), 2048);
        assert_eq!(rle_array.values.len(), 4);

        assert_eq!(rle_array.chunk_idx(0), 0);
        assert_eq!(rle_array.chunk_idx(1024), 1);
        assert_eq!(rle_array.chunk_idx(2047), 1);

        assert_eq!(rle_array.value_chunk_offset(0), 0);
        assert_eq!(rle_array.value_chunk_offset(1), 2);
    }
}
