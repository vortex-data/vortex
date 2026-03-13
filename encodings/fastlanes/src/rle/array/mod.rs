// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::stats::ArrayStats;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::FL_CHUNK_SIZE;

pub mod rle_compress;
pub mod rle_decompress;

#[derive(Clone, Debug)]
pub struct RLEArray {
    pub(super) dtype: DType,
    /// Run value in the dictionary.
    pub(super) values: ArrayRef,
    /// Chunk-local indices from all chunks. The start of each chunk is looked up in `values_idx_offsets`.
    pub(super) indices: ArrayRef,
    /// Index start positions of each value chunk.
    ///
    /// # Example
    /// ```
    /// // Chunk 0: [10, 20] (starts at index 0)
    /// // Chunk 1: [30, 40] (starts at index 2)
    /// let values = [10, 20, 30, 40];           // Global values array
    /// let values_idx_offsets = [0, 2];         // Chunk 0 starts at index 0, Chunk 1 starts at index 2
    /// ```
    pub(super) values_idx_offsets: ArrayRef,

    pub(super) stats_set: ArrayStats,
    // Offset relative to the start of the chunk.
    pub(super) offset: usize,
    pub(super) length: usize,
}

impl RLEArray {
    fn validate(
        values: &ArrayRef,
        indices: &ArrayRef,
        value_idx_offsets: &ArrayRef,
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
    pub fn len(&self) -> usize {
        self.length
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    #[inline]
    pub fn dtype(&self) -> &DType {
        &self.dtype
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
    #[expect(
        clippy::expect_used,
        reason = "expect is safe here as scalar_at returns a valid primitive"
    )]
    pub(crate) fn values_idx_offset(&self, chunk_idx: usize) -> usize {
        self.values_idx_offsets
            .scalar_at(chunk_idx)
            .expect("index must be in bounds")
            .as_primitive()
            .as_::<usize>()
            .expect("index must be of type usize")
            - self
                .values_idx_offsets
                .scalar_at(0)
                .expect("index must be in bounds")
                .as_primitive()
                .as_::<usize>()
                .expect("index must be of type usize")
    }

    /// Index offset into the array
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub(crate) fn stats_set(&self) -> &ArrayStats {
        &self.stats_set
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayContext;
    use vortex_array::DynArray;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::serde::ArrayParts;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_session::registry::ReadContext;

    use crate::RLEArray;
    use crate::test::SESSION;

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
        assert_eq!(rle_array.values().len(), 3);
        assert_eq!(rle_array.values().dtype().as_ptype(), PType::U32);
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
        assert_eq!(rle_array.values().len(), 2);
        assert!(rle_array.is_valid(0).unwrap());
        assert!(!rle_array.is_valid(1).unwrap());
        assert!(rle_array.is_valid(2).unwrap());
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

        let valid_slice = rle_array.slice(0..3).unwrap().to_primitive();
        // TODO(joe): replace with compute null count
        assert!(valid_slice.all_valid().unwrap());

        let mixed_slice = rle_array.slice(1..5).unwrap();
        assert!(!mixed_slice.all_valid().unwrap());
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

        // TODO(joe): replace with compute null count
        let invalid_slice = rle_array
            .slice(2..5)
            .unwrap()
            .to_canonical()
            .unwrap()
            .into_primitive();
        assert!(invalid_slice.all_invalid().unwrap());

        let mixed_slice = rle_array.slice(1..4).unwrap();
        assert!(!mixed_slice.all_invalid().unwrap());
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

        let sliced_array = rle_array.slice(1..4).unwrap();
        let validity_mask = sliced_array.validity_mask().unwrap();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let expected_mask = Validity::from_iter([false, true, false])
            .execute_mask(3, &mut ctx)
            .unwrap();
        assert_eq!(validity_mask.len(), expected_mask.len());
        assert_eq!(validity_mask, expected_mask);
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
        assert_eq!(rle_array.values().len(), 0);
    }

    #[test]
    fn test_multi_chunk_two_chunks() {
        let values = PrimitiveArray::from_iter([10u32, 20, 30, 40]).into_array();
        let indices = PrimitiveArray::from_iter([0u16, 1].repeat(1024)).into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([0u64, 2]).into_array();
        let rle_array = RLEArray::try_new(values, indices, values_idx_offsets, 0, 2048).unwrap();

        assert_eq!(rle_array.len(), 2048);
        assert_eq!(rle_array.values().len(), 4);

        assert_eq!(rle_array.values_idx_offset(0), 0);
        assert_eq!(rle_array.values_idx_offset(1), 2);
    }

    #[test]
    fn test_rle_serialization() {
        let primitive = PrimitiveArray::from_iter((0..2048).map(|i| (i / 100) as u32));
        let rle_array = RLEArray::encode(&primitive).unwrap();
        assert_eq!(rle_array.len(), 2048);

        let original_data = rle_array.to_primitive();

        let ctx = ArrayContext::empty();
        let serialized = rle_array
            .into_array()
            .serialize(&ctx, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = ArrayParts::try_from(concat).unwrap();
        let decoded = parts
            .decode(
                &DType::Primitive(PType::U32, Nullability::NonNullable),
                2048,
                &ReadContext::new(ctx.to_ids()),
                &SESSION,
            )
            .unwrap();

        let decoded_data = decoded.to_primitive();

        assert_arrays_eq!(original_data, decoded_data);
    }

    #[test]
    fn test_rle_serialization_slice() {
        let primitive = PrimitiveArray::from_iter((0..2048).map(|i| (i / 100) as u32));
        let rle_array = RLEArray::encode(&primitive).unwrap();

        let sliced = RLEArray::try_new(
            rle_array.values().clone(),
            rle_array.indices().clone(),
            rle_array.values_idx_offsets().clone(),
            100,
            100,
        )
        .unwrap();
        assert_eq!(sliced.len(), 100);

        let ctx = ArrayContext::empty();
        let serialized = sliced
            .clone()
            .into_array()
            .serialize(&ctx, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = ArrayParts::try_from(concat).unwrap();
        let decoded = parts
            .decode(
                sliced.dtype(),
                sliced.len(),
                &ReadContext::new(ctx.to_ids()),
                &SESSION,
            )
            .unwrap();

        let original_data = sliced.to_primitive();
        let decoded_data = decoded.to_primitive();

        assert_arrays_eq!(original_data, decoded_data);
    }
}
