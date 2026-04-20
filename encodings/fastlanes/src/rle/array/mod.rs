// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::TypedArrayRef;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

pub mod rle_compress;
pub mod rle_decompress;

/// Run values in the dictionary.
pub(super) const VALUES_SLOT: usize = 0;
/// Chunk-local indices from all chunks. The start of each chunk is looked up in `values_idx_offsets`.
pub(super) const INDICES_SLOT: usize = 1;
/// Index start positions of each value chunk.
///
/// # Example
/// ```text
/// // Chunk 0: [10, 20] (starts at index 0)
/// // Chunk 1: [30, 40] (starts at index 2)
/// let values = [10, 20, 30, 40];           // Global values array
/// let values_idx_offsets = [0, 2];         // Chunk 0 starts at index 0, Chunk 1 starts at index 2
/// ```
pub(super) const VALUES_IDX_OFFSETS_SLOT: usize = 2;
pub(super) const NUM_SLOTS: usize = 3;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["values", "indices", "values_idx_offsets"];

#[derive(Clone, Debug)]
pub struct RLEData {
    // Offset relative to the start of the chunk.
    pub(super) offset: usize,
}

impl Display for RLEData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "offset: {}", self.offset)
    }
}

impl RLEData {
    /// Create a new chunk-based RLE array from its components.
    ///
    /// # Arguments
    ///
    /// * `values` - Unique values from all chunks
    /// * `indices` - Chunk-local indices from all chunks
    /// * `values_idx_offsets` - Start indices for each value chunk.
    /// * `offset` - Offset into the first chunk
    /// * `length` - Array length
    pub fn try_new(offset: usize) -> VortexResult<Self> {
        vortex_ensure!(
            offset < 1024,
            "Offset must be smaller than 1024, got {}",
            offset
        );
        Ok(Self { offset })
    }

    /// Create a new RLEArray without validation.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `offset + length` does not exceed the length of the indices array
    /// - The `indices` array contains valid indices into chunks of the `values` array
    /// - The `values_idx_offsets` array contains valid chunk start offsets
    pub unsafe fn new_unchecked(offset: usize) -> Self {
        Self { offset }
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }
}

pub trait RLEArrayExt: TypedArrayRef<crate::RLE> {
    #[inline]
    fn values(&self) -> &ArrayRef {
        self.as_ref().slots()[VALUES_SLOT]
            .as_ref()
            .vortex_expect("RLEArray values slot must be populated")
    }

    #[inline]
    fn indices(&self) -> &ArrayRef {
        self.as_ref().slots()[INDICES_SLOT]
            .as_ref()
            .vortex_expect("RLEArray indices slot must be populated")
    }

    #[inline]
    fn values_idx_offsets(&self) -> &ArrayRef {
        self.as_ref().slots()[VALUES_IDX_OFFSETS_SLOT]
            .as_ref()
            .vortex_expect("RLEArray values_idx_offsets slot must be populated")
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
    fn values_idx_offset(&self, chunk_idx: usize, ctx: &mut ExecutionCtx) -> usize {
        self.values_idx_offsets()
            .execute_scalar(chunk_idx, ctx)
            .expect("index must be in bounds")
            .as_primitive()
            .as_::<usize>()
            .expect("index must be of type usize")
            - self
                .values_idx_offsets()
                .execute_scalar(0, ctx)
                .expect("index must be in bounds")
                .as_primitive()
                .as_::<usize>()
                .expect("index must be of type usize")
    }

    /// Index offset into the array
    #[inline]
    fn offset(&self) -> usize {
        self.offset
    }
}

impl<T: TypedArrayRef<crate::RLE>> RLEArrayExt for T {}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayContext;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::primitive::PrimitiveArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use crate::FL_CHUNK_SIZE;
    use crate::RLE;
    use crate::RLEData;
    use crate::rle::array::RLEArrayExt;
    use crate::test::SESSION;

    #[test]
    fn test_try_new() {
        let values = PrimitiveArray::from_iter([10u32, 20, 30]).into_array();

        // Pad indices to 1024 chunk.
        let indices =
            PrimitiveArray::from_iter([0u16, 0, 1, 1, 2].iter().cycle().take(1024).copied())
                .into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();
        let rle_array = RLE::try_new(values, indices, values_idx_offsets, 0, 5)
            .vortex_expect("RLEData is always valid");

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

        let rle_array = RLE::try_new(values, indices_with_validity, values_idx_offsets, 0, 3)
            .vortex_expect("RLEData is always valid");

        assert_eq!(rle_array.len(), 3);
        assert_eq!(rle_array.values().len(), 2);
        let mut ctx = SESSION.create_execution_ctx();
        assert!(rle_array.is_valid(0, &mut ctx).unwrap());
        assert!(!rle_array.is_valid(1, &mut ctx).unwrap());
        assert!(rle_array.is_valid(2, &mut ctx).unwrap());
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

        let rle_array = RLE::try_new(values, indices_with_validity, values_idx_offsets, 0, 5)
            .vortex_expect("RLEData is always valid");

        let mut ctx = SESSION.create_execution_ctx();
        let valid_slice = rle_array
            .slice(0..3)
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        // TODO(joe): replace with compute null count
        assert!(valid_slice.all_valid(&mut ctx).unwrap());

        let mixed_slice = rle_array.slice(1..5).unwrap();
        assert!(!mixed_slice.all_valid(&mut ctx).unwrap());
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

        let rle_array = RLE::try_new(values, indices_with_validity, values_idx_offsets, 0, 5)
            .vortex_expect("RLEData is always valid");

        // TODO(joe): replace with compute null count
        let invalid_slice = rle_array
            .slice(2..5)
            .unwrap()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .into_primitive();
        let mut ctx = SESSION.create_execution_ctx();
        assert!(invalid_slice.all_invalid(&mut ctx).unwrap());

        let mixed_slice = rle_array.slice(1..4).unwrap();
        assert!(!mixed_slice.all_invalid(&mut ctx).unwrap());
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

        let rle_array = RLE::try_new(values, indices_with_validity, values_idx_offsets, 0, 4)
            .vortex_expect("RLEData is always valid");

        let sliced_array = rle_array.slice(1..4).unwrap();
        let validity_mask = sliced_array
            .validity()
            .unwrap()
            .execute_mask(
                sliced_array.len(),
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap();

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
        let rle_array = RLE::try_new(
            values,
            indices.clone(),
            values_idx_offsets,
            0,
            indices.len(),
        )
        .vortex_expect("RLEData is always valid");

        assert_eq!(rle_array.len(), 0);
        assert_eq!(rle_array.values().len(), 0);
    }

    #[test]
    fn test_multi_chunk_two_chunks() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter([10u32, 20, 30, 40]).into_array();
        let indices = PrimitiveArray::from_iter([0u16, 1].repeat(1024)).into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([0u64, 2]).into_array();
        let rle_array = RLE::try_new(values, indices, values_idx_offsets, 0, 2048)
            .vortex_expect("RLEData is always valid");

        assert_eq!(rle_array.len(), 2048);
        assert_eq!(rle_array.values().len(), 4);

        assert_eq!(rle_array.values_idx_offset(0, &mut ctx), 0);
        assert_eq!(rle_array.values_idx_offset(1, &mut ctx), 2);
    }

    #[test]
    fn test_rle_serialization() -> VortexResult<()> {
        let mut exec_ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_iter((0..2048).map(|i| (i / 100) as u32));
        let rle_array = RLEData::encode(primitive.as_view(), &mut exec_ctx).unwrap();
        assert_eq!(rle_array.len(), 2048);

        let original_data = rle_array
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut exec_ctx)?;

        let ctx = ArrayContext::empty();
        let serialized = rle_array
            .into_array()
            .serialize(&ctx, &SESSION, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = SerializedArray::try_from(concat).unwrap();
        let decoded = parts
            .decode(
                &DType::Primitive(PType::U32, Nullability::NonNullable),
                2048,
                &ReadContext::new(ctx.to_ids()),
                &SESSION,
            )
            .unwrap();

        let decoded_data = decoded.execute::<PrimitiveArray>(&mut exec_ctx)?;

        assert_arrays_eq!(original_data, decoded_data);
        Ok(())
    }

    #[test]
    fn test_rle_serialization_slice() -> VortexResult<()> {
        let mut exec_ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_iter((0..2048).map(|i| (i / 100) as u32));
        let rle_array = RLEData::encode(primitive.as_view(), &mut exec_ctx).unwrap();

        let sliced = RLE::try_new(
            rle_array.values().clone(),
            rle_array.indices().clone(),
            rle_array.values_idx_offsets().clone(),
            100,
            100,
        )
        .vortex_expect("RLEData is always valid");
        assert_eq!(sliced.len(), 100);

        let ctx = ArrayContext::empty();
        let serialized = sliced
            .clone()
            .into_array()
            .serialize(&ctx, &SESSION, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = SerializedArray::try_from(concat).unwrap();
        let decoded = parts
            .decode(
                sliced.dtype(),
                sliced.len(),
                &ReadContext::new(ctx.to_ids()),
                &SESSION,
            )
            .unwrap();

        let original_data = sliced
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut exec_ctx)?;
        let decoded_data = decoded.execute::<PrimitiveArray>(&mut exec_ctx)?;

        assert_arrays_eq!(original_data, decoded_data);
        Ok(())
    }

    /// Regression test: re-encoding RLE indices with RLE must not corrupt
    /// chunk-local index values via cross-chunk fill-forward.
    ///
    /// The scenario: an array spanning 2 chunks where chunk 0 has 2 distinct
    /// non-null values (producing chunk-local indices 0 and 1) and chunk 1 is
    /// entirely null. When fill_forward_nulls propagated the last valid index
    /// (1) from chunk 0 into chunk 1 during re-encoding, decoding panicked
    /// because chunk 1 only had 1 unique value and index 1 was out of bounds.
    #[test]
    fn test_recompress_indices_no_cross_chunk_leak() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = FL_CHUNK_SIZE + 100;
        let mut values: Vec<Option<i16>> = vec![None; len];
        // Two distinct values in chunk 0 → indices 0 and 1.
        values[0] = Some(10);
        values[500] = Some(20);
        // Chunk 1 (positions 1024..) is all-null.

        let original = PrimitiveArray::from_option_iter(values);
        let rle = RLEData::encode(original.as_view(), &mut ctx)?;

        // Simulate cascading compression: narrow u16->u8 then re-encode with RLE,
        // matching the path taken by the BtrBlocks compressor.
        let indices_prim = rle
            .indices()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .narrow()?;
        let re_encoded = RLEData::encode(indices_prim.as_view(), &mut ctx)?;

        // Reconstruct the outer RLE with re-encoded indices.
        // SAFETY: we only replace the indices child; all other invariants hold.
        let reconstructed = unsafe {
            RLE::new_unchecked(
                rle.values().clone(),
                re_encoded.into_array(),
                rle.values_idx_offsets().clone(),
                rle.offset(),
                rle.len(),
            )
        };

        // Decompress — panicked before the fill_forward_nulls chunk-boundary fix.
        let decoded = reconstructed
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, original);
        Ok(())
    }
}
