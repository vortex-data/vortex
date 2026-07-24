// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::RLE;
use num_traits::AsPrimitive;
use num_traits::NumCast;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_native_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::FL_CHUNK_SIZE;
use crate::RLEArray;
use crate::rle::RLEArrayExt;

/// Decompresses an RLE array back into a primitive array.
pub fn rle_decompress(array: &RLEArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    // The per-chunk value-index offsets are tiny (one entry per 1024-element chunk), so cast them
    // to `u64` once here instead of monomorphizing the whole decode loop over the offset width.
    let values_idx_offsets = array
        .values_idx_offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let values_idx_offsets: Vec<u64> =
        match_each_unsigned_integer_ptype!(values_idx_offsets.ptype(), |O| {
            values_idx_offsets
                .as_slice::<O>()
                .iter()
                .map(|&o| o.as_())
                .collect()
        });

    match_each_native_ptype!(array.values().dtype().as_ptype(), |V| {
        // RLE indices are always u16 (or u8 if downcasted).
        match array.indices().dtype().as_ptype() {
            PType::U8 => rle_decode_typed::<V, u8>(array, &values_idx_offsets, ctx),
            PType::U16 => rle_decode_typed::<V, u16>(array, &values_idx_offsets, ctx),
            _ => vortex_panic!(
                "Unsupported index type for RLE decoding: {}",
                array.indices().dtype().as_ptype()
            ),
        }
    })
}

/// Decompresses an `RLEArray` into to a primitive array of unsigned integers.
fn rle_decode_typed<V, I>(
    array: &RLEArray,
    values_idx_offsets: &[u64],
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    V: NativePType + RLE + Clone + Copy,
    I: NativePType + Into<usize>,
{
    let values = array.values().clone().execute::<PrimitiveArray>(ctx)?;
    let values = values.as_slice::<V>();

    // The offsets come from (possibly untrusted) storage. Validate them once here so
    // that the per-chunk slicing and the unchecked decodes below stay within `values`:
    // offsets must be non-decreasing and span at most `values.len()` values.
    vortex_ensure!(
        values_idx_offsets.is_sorted(),
        "RLE values_idx_offsets must be non-decreasing"
    );
    if let (Some(&first), Some(&last)) = (values_idx_offsets.first(), values_idx_offsets.last()) {
        vortex_ensure!(
            last - first <= values.len() as u64,
            "RLE values_idx_offsets span {} values but only {} are present",
            last - first,
            values.len()
        );
    }

    let indices = array.indices().clone().execute::<PrimitiveArray>(ctx)?;
    assert!(indices.len().is_multiple_of(FL_CHUNK_SIZE));
    let has_invalid = !indices.all_valid(ctx)?;
    let (indices_sl, _) = indices.as_slice::<I>().as_chunks::<FL_CHUNK_SIZE>();

    let chunk_start_idx = array.offset() / FL_CHUNK_SIZE;
    let chunk_end_idx = (array.offset() + array.len()).div_ceil(FL_CHUNK_SIZE);
    let num_chunks = chunk_end_idx - chunk_start_idx;

    let mut buffer = BufferMut::<V>::with_capacity(num_chunks * FL_CHUNK_SIZE);
    let (out_buf, _) = buffer.spare_capacity_mut().as_chunks_mut::<FL_CHUNK_SIZE>();

    for (chunk_idx, (chunk_indices, chunk_out)) in
        indices_sl.iter().zip(out_buf.iter_mut()).enumerate()
    {
        // Offsets in `values_idx_offsets` are absolute and need to be shifted
        // by the offset of the first chunk, respective of the current slice,
        // to make them relative.
        let value_idx_offset = (values_idx_offsets[chunk_idx] - values_idx_offsets[0]) as usize;

        let next_value_idx_offset = if chunk_idx + 1 < num_chunks {
            (values_idx_offsets[chunk_idx + 1] - values_idx_offsets[0]) as usize
        } else {
            values.len()
        };
        let num_chunk_values = u16::try_from(next_value_idx_offset - value_idx_offset)
            .vortex_expect("There can be at most 1024 values in RLE chunk");
        vortex_ensure!(
            num_chunk_values > 0,
            "RLE chunk {chunk_idx} references no values"
        );

        // SAFETY: `MaybeUninit<T>` and `T` have the same layout.
        let buffer_values: &mut [V; FL_CHUNK_SIZE] = unsafe { std::mem::transmute(chunk_out) };
        let chunk_values = &values[value_idx_offset..];
        if num_chunk_values == 1 {
            // Single-value chunk: fill directly to avoid out-of-bounds index
            // access. The indices may contain values other than 0 when they
            // have been further compressed (e.g., as a masked constant).
            buffer_values.fill(chunk_values[0]);
        } else if has_invalid {
            // When the indices array has invalid (null) positions, those
            // positions may contain arbitrary garbage values after further
            // compression. Clamp all indices into [0, num_chunk_values) to
            // prevent out-of-bounds access in the fastlanes decoder.
            let mut sanitized: [u16; FL_CHUNK_SIZE] = [0; FL_CHUNK_SIZE];
            for (idx_out, idx) in sanitized.iter_mut().zip(chunk_indices) {
                let idx: u16 =
                    NumCast::from(*idx).vortex_expect("RLE indices are always less than u16");
                *idx_out = idx.min(num_chunk_values - 1);
            }
            // SAFETY: every sanitized index is clamped below `num_chunk_values`, which the
            // offset validation above bounds by `chunk_values.len()`.
            unsafe { V::decode_unchecked(chunk_values, &sanitized, buffer_values) };
        } else {
            // The indices also come from (possibly untrusted) storage, so bound-check the
            // whole chunk up front. A single max-reduction vectorizes and keeps the check
            // out of the per-element decode loop.
            let max_index: usize = chunk_indices
                .iter()
                .map(|idx| (*idx).into())
                .max()
                .unwrap_or_default();
            vortex_ensure!(
                max_index < num_chunk_values as usize,
                "RLE index {max_index} out of bounds for chunk {chunk_idx} with {num_chunk_values} values"
            );
            // SAFETY: just checked that every index in the chunk is below
            // `num_chunk_values`, which the offset validation above bounds by
            // `chunk_values.len()`.
            unsafe { V::decode_unchecked(chunk_values, chunk_indices, buffer_values) };
        }
    }

    unsafe {
        buffer.set_len(num_chunks * FL_CHUNK_SIZE);
    }

    let offset_within_chunk = array.offset();

    Ok(PrimitiveArray::new(
        buffer
            .freeze()
            .slice(offset_within_chunk..(offset_within_chunk + array.len())),
        array.validity()?,
    ))
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;

    use crate::FL_CHUNK_SIZE;
    use crate::RLE;
    use crate::rle::array::rle_decompress::rle_decompress;
    use crate::test::SESSION;

    fn indices_with_oob(oob: u16) -> vortex_array::ArrayRef {
        let mut indices = [0u16, 1]
            .iter()
            .cycle()
            .take(FL_CHUNK_SIZE)
            .copied()
            .collect::<Vec<_>>();
        indices[100] = oob;
        PrimitiveArray::from_iter(indices).into_array()
    }

    #[test]
    fn test_decode_rejects_out_of_bounds_index() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter([10u32, 20]).into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();

        // Index 999 points far beyond the 2 values of the only chunk.
        let rle = RLE::try_new(
            values,
            indices_with_oob(999),
            values_idx_offsets,
            0,
            FL_CHUNK_SIZE,
        )?;
        assert!(rle_decompress(&rle, &mut ctx).is_err());
        Ok(())
    }

    #[test]
    fn test_decode_rejects_index_into_next_chunk() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Two chunks with 2 values each: an index of 2 in chunk 0 is within `values`
        // but out of bounds for the chunk, and must be rejected rather than silently
        // reading chunk 1's values.
        let values = PrimitiveArray::from_iter([10u32, 20, 30, 40]).into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([0u64, 2]).into_array();
        let mut indices = [0u16, 1].repeat(FL_CHUNK_SIZE);
        indices[100] = 2;
        let indices = PrimitiveArray::from_iter(indices).into_array();

        let rle = RLE::try_new(values, indices, values_idx_offsets, 0, 2 * FL_CHUNK_SIZE)?;
        assert!(rle_decompress(&rle, &mut ctx).is_err());
        Ok(())
    }

    #[test]
    fn test_decode_rejects_non_monotonic_offsets() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter([10u32, 20, 30, 40]).into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([2u64, 0]).into_array();
        let indices = PrimitiveArray::from_iter([0u16, 1].repeat(FL_CHUNK_SIZE)).into_array();

        let rle = RLE::try_new(values, indices, values_idx_offsets, 0, 2 * FL_CHUNK_SIZE)?;
        assert!(rle_decompress(&rle, &mut ctx).is_err());
        Ok(())
    }

    #[test]
    fn test_decode_rejects_offsets_beyond_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter([10u32, 20, 30, 40]).into_array();
        // The offset span (100) exceeds the number of values present (4).
        let values_idx_offsets = PrimitiveArray::from_iter([0u64, 100]).into_array();
        let indices = PrimitiveArray::from_iter([0u16, 1].repeat(FL_CHUNK_SIZE)).into_array();

        let rle = RLE::try_new(values, indices, values_idx_offsets, 0, 2 * FL_CHUNK_SIZE)?;
        assert!(rle_decompress(&rle, &mut ctx).is_err());
        Ok(())
    }

    #[test]
    fn test_decode_rejects_empty_chunk() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter([10u32, 20]).into_array();
        // Chunk 0 references no values: offsets [0, 0].
        let values_idx_offsets = PrimitiveArray::from_iter([0u64, 0]).into_array();
        let indices = PrimitiveArray::from_iter([0u16, 1].repeat(FL_CHUNK_SIZE)).into_array();

        let rle = RLE::try_new(values, indices, values_idx_offsets, 0, 2 * FL_CHUNK_SIZE)?;
        assert!(rle_decompress(&rle, &mut ctx).is_err());
        Ok(())
    }

    #[test]
    fn test_decode_in_bounds_indices_roundtrip() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter([10u32, 20]).into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();
        let indices =
            PrimitiveArray::from_iter([0u16, 1].iter().cycle().take(FL_CHUNK_SIZE).copied())
                .into_array();

        let rle = RLE::try_new(values, indices, values_idx_offsets, 0, FL_CHUNK_SIZE)?;
        let decoded = rle_decompress(&rle, &mut ctx)?;
        let expected =
            PrimitiveArray::from_iter([10u32, 20].iter().cycle().take(FL_CHUNK_SIZE).copied());
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }
}
