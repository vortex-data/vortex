// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::RLE;
use num_traits::AsPrimitive;
use num_traits::NumCast;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_native_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::RLEArray;
use crate::rle::RLEArrayExt;
use crate::rle::RLEArraySlotsExt;

/// Decompresses an RLE array back into a primitive array.
pub fn rle_decompress(array: &RLEArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let values_idx_offsets = load_values_idx_offsets(array.values_idx_offsets(), ctx)?;

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

/// Load the per-chunk value-index offsets as `u64`.
///
/// They are tiny (one entry per 1024-element chunk), so they are cast once here instead of
/// monomorphizing the whole decode loop over the offset width.
pub(crate) fn load_values_idx_offsets(
    values_idx_offsets: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<u64>> {
    let values_idx_offsets = values_idx_offsets.clone().execute::<PrimitiveArray>(ctx)?;
    Ok(match_each_unsigned_integer_ptype!(
        values_idx_offsets.ptype(),
        |O| {
            values_idx_offsets
                .as_slice::<O>()
                .iter()
                .map(|&o| o.as_())
                .collect()
        }
    ))
}

/// Decompresses an `RLEArray` into to a primitive array of unsigned integers.
fn rle_decode_typed<V, I>(
    array: &RLEArray,
    values_idx_offsets: &[u64],
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    V: NativePType + RLE + Clone + Copy,
    I: NativePType + Ord + Into<usize>,
{
    let values = array.values().clone().execute::<PrimitiveArray>(ctx)?;
    let values = values.as_slice::<V>();

    let index_bound = bitpacked_index_bound(array.indices());
    let indices = array.indices().clone().execute::<PrimitiveArray>(ctx)?;
    assert!(indices.len().is_multiple_of(FL_CHUNK_SIZE));
    let indices_validity = indices.validity()?.execute_mask(indices.len(), ctx)?;
    // `None` means every position is valid.
    let validity_bits = (!indices_validity.all_true()).then(|| indices_validity.to_bit_buffer());
    let (indices_sl, _) = indices.as_slice::<I>().as_chunks::<FL_CHUNK_SIZE>();

    let chunk_start_idx = array.offset() / FL_CHUNK_SIZE;
    let chunk_end_idx = (array.offset() + array.len()).div_ceil(FL_CHUNK_SIZE);
    let num_chunks = chunk_end_idx - chunk_start_idx;

    let decoder = ChunkDecoder::try_new(
        values,
        values_idx_offsets,
        num_chunks,
        validity_bits.as_ref(),
    )?
    .with_index_bound(index_bound);

    let mut buffer = BufferMut::<V>::with_capacity(num_chunks * FL_CHUNK_SIZE);
    let (out_buf, _) = buffer.spare_capacity_mut().as_chunks_mut::<FL_CHUNK_SIZE>();

    for (chunk_idx, (chunk_indices, chunk_out)) in
        indices_sl.iter().zip(out_buf.iter_mut()).enumerate()
    {
        // SAFETY: `MaybeUninit<T>` and `T` have the same layout.
        let buffer_values: &mut [V; FL_CHUNK_SIZE] = unsafe { std::mem::transmute(chunk_out) };
        decoder.decode(chunk_idx, chunk_indices, buffer_values)?;
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

/// The bound on indices that their encoding guarantees: bit-packed indices without patches are
/// below `1 << bit_width`.
pub(crate) fn bitpacked_index_bound(indices: &ArrayRef) -> usize {
    indices
        .as_opt::<BitPacked>()
        .filter(|bp| bp.patches().is_none())
        .map_or(usize::MAX, |bp| 1 << bp.bit_width())
}

/// Decodes RLE chunks from their indices: the run values, where each chunk's values start, and
/// which indices are valid.
pub(crate) struct ChunkDecoder<'a, V> {
    values: &'a [V],
    values_idx_offsets: &'a [u64],
    num_chunks: usize,
    /// `None` means every index is valid.
    validity_bits: Option<&'a BitBuffer>,
    /// Every index is below this, as their encoding guarantees, e.g. bit-packed indices are below
    /// `1 << bit_width`. Chunks with at least this many values need no bounds check.
    index_bound: usize,
}

impl<'a, V: NativePType + RLE> ChunkDecoder<'a, V> {
    /// Validate the value-index offsets against `values`.
    ///
    /// The offsets come from (possibly untrusted) storage. Validating them once here keeps the
    /// per-chunk slicing and the unchecked decodes within `values`: offsets must be
    /// non-decreasing and span at most `values.len()` values.
    pub(crate) fn try_new(
        values: &'a [V],
        values_idx_offsets: &'a [u64],
        num_chunks: usize,
        validity_bits: Option<&'a BitBuffer>,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            values_idx_offsets.is_sorted(),
            "RLE values_idx_offsets must be non-decreasing"
        );
        if let (Some(&first), Some(&last)) = (values_idx_offsets.first(), values_idx_offsets.last())
        {
            vortex_ensure!(
                last - first <= values.len() as u64,
                "RLE values_idx_offsets span {} values but only {} are present",
                last - first,
                values.len()
            );
        }
        Ok(Self {
            values,
            values_idx_offsets,
            num_chunks,
            validity_bits,
            index_bound: usize::MAX,
        })
    }

    /// Declare that every index is below `index_bound`, so chunks with at least that many values
    /// skip their bounds check.
    pub(crate) fn with_index_bound(mut self, index_bound: usize) -> Self {
        self.index_bound = index_bound;
        self
    }

    /// Decode chunk `chunk_idx`, counted from the first chunk of the indices, into `out`.
    pub(crate) fn decode<I: NativePType + Ord + Into<usize>>(
        &self,
        chunk_idx: usize,
        chunk_indices: &[I; FL_CHUNK_SIZE],
        out: &mut [V; FL_CHUNK_SIZE],
    ) -> VortexResult<()> {
        let values_idx_offsets = self.values_idx_offsets;
        let values = self.values;
        // Offsets in `values_idx_offsets` are absolute and need to be shifted
        // by the offset of the first chunk, respective of the current slice,
        // to make them relative.
        let value_idx_offset = (values_idx_offsets[chunk_idx] - values_idx_offsets[0]) as usize;

        let next_value_idx_offset = if chunk_idx + 1 < self.num_chunks {
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

        let chunk_values = &values[value_idx_offset..];
        if num_chunk_values == 1 {
            // Single-value chunk: fill directly to avoid out-of-bounds index
            // access. The indices may contain values other than 0 when they
            // have been further compressed (e.g., as a masked constant).
            out.fill(chunk_values[0]);
            return Ok(());
        }

        let chunk_start = chunk_idx * FL_CHUNK_SIZE;
        let chunk_valid_count = self.validity_bits.map_or(FL_CHUNK_SIZE, |bits| {
            bits.count_range(chunk_start, chunk_start + FL_CHUNK_SIZE)
        });
        if chunk_valid_count == FL_CHUNK_SIZE {
            decode_chunk_checked(
                chunk_values,
                chunk_indices,
                num_chunk_values,
                self.index_bound,
                chunk_idx,
                out,
            )
        } else if chunk_valid_count == 0 {
            // Entirely-null chunk: every position is masked out, so the indices are
            // meaningless (possibly garbage after further compression) — skip the
            // gather and emit an arbitrary in-bounds value.
            out.fill(chunk_values[0]);
            Ok(())
        } else {
            // Null positions may contain arbitrary garbage indices after further
            // compression, so zero them out (their decoded values are masked by the
            // validity). Valid positions must be genuinely in bounds and are still
            // bound-checked before the unchecked gather: a corrupt index at a valid
            // position is an error, never silently remapped.
            let bits = self
                .validity_bits
                .vortex_expect("mixed-validity chunk implies a materialized mask");
            let mut sanitized: [u16; FL_CHUNK_SIZE] = [0; FL_CHUNK_SIZE];
            bits.slice(chunk_start..chunk_start + FL_CHUNK_SIZE)
                .for_each_set_index(|i| {
                    sanitized[i] = NumCast::from(chunk_indices[i])
                        .vortex_expect("RLE indices are always less than u16");
                });
            decode_chunk_checked(
                chunk_values,
                &sanitized,
                num_chunk_values,
                self.index_bound,
                chunk_idx,
                out,
            )
        }
    }
}

/// The largest index in the chunk.
///
/// The check guards every decoded chunk, so it runs with AVX2 where available: the baseline x86
/// target has no unsigned 16-bit max, which the portable reduction emulates in several
/// instructions per vector.
#[inline]
fn max_index<I: Copy + Ord + Default>(chunk_indices: &[I; FL_CHUNK_SIZE]) -> I {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is available.
        return unsafe { max_index_avx2(chunk_indices) };
    }
    max_index_portable(chunk_indices)
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2")]
fn max_index_avx2<I: Copy + Ord + Default>(chunk_indices: &[I; FL_CHUNK_SIZE]) -> I {
    max_index_portable(chunk_indices)
}

/// Reduce in the index type, which is unsigned: widening every index to `usize` first is several
/// times slower.
#[allow(clippy::inline_always)]
#[inline(always)]
fn max_index_portable<I: Copy + Ord + Default>(chunk_indices: &[I; FL_CHUNK_SIZE]) -> I {
    chunk_indices.iter().copied().fold(I::default(), Ord::max)
}

/// Bound-checks every index in the chunk, then runs the unchecked fastlanes gather.
///
/// The indices come from (possibly untrusted) storage: a single max-reduction over the
/// chunk vectorizes and keeps the bounds check out of the per-element decode loop.
fn decode_chunk_checked<V, I>(
    chunk_values: &[V],
    chunk_indices: &[I; FL_CHUNK_SIZE],
    num_chunk_values: u16,
    index_bound: usize,
    chunk_idx: usize,
    out: &mut [V; FL_CHUNK_SIZE],
) -> VortexResult<()>
where
    V: RLE,
    I: Copy + Ord + Default + Into<usize>,
{
    if index_bound > num_chunk_values as usize {
        let max_index: usize = max_index(chunk_indices).into();
        vortex_ensure!(
            max_index < num_chunk_values as usize,
            "RLE index {max_index} out of bounds for chunk {chunk_idx} with {num_chunk_values} values"
        );
    }
    // SAFETY: every index in the chunk is below `num_chunk_values`, checked above unless their
    // encoding bounds them by at most that, and the caller's offset validation bounds it by
    // `chunk_values.len()`.
    unsafe { V::decode_unchecked(chunk_values, chunk_indices, out) };
    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
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
    fn test_decode_rejects_out_of_bounds_index_at_valid_position() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter([10u32, 20]).into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();

        // Position 100 is VALID but holds an out-of-bounds index: this is corruption
        // and must fail, even though other positions are null.
        let mut indices = [0u16, 1].repeat(FL_CHUNK_SIZE / 2);
        indices[100] = 999;
        let mut validity = vec![true; FL_CHUNK_SIZE];
        validity[200] = false;
        let indices = PrimitiveArray::new(
            indices.into_iter().collect::<Buffer<u16>>(),
            Validity::from_iter(validity),
        )
        .into_array();

        let rle = RLE::try_new(values, indices, values_idx_offsets, 0, FL_CHUNK_SIZE)?;
        assert!(rle_decompress(&rle, &mut ctx).is_err());
        Ok(())
    }

    #[test]
    fn test_decode_tolerates_garbage_index_at_null_position() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter([10u32, 20]).into_array();
        let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();

        // Position 100 is NULL and holds a garbage out-of-bounds index (e.g. after
        // further compression of the indices): its value is masked, so decoding
        // must succeed and every valid position must decode correctly.
        let mut indices = [0u16, 1].repeat(FL_CHUNK_SIZE / 2);
        indices[100] = 999;
        let mut validity = vec![true; FL_CHUNK_SIZE];
        validity[100] = false;
        let indices = PrimitiveArray::new(
            indices.into_iter().collect::<Buffer<u16>>(),
            Validity::from_iter(validity),
        )
        .into_array();

        let rle = RLE::try_new(values, indices, values_idx_offsets, 0, FL_CHUNK_SIZE)?;
        let decoded = rle_decompress(&rle, &mut ctx)?;
        let expected = PrimitiveArray::from_option_iter(
            (0..FL_CHUNK_SIZE).map(|i| (i != 100).then_some(if i % 2 == 0 { 10u32 } else { 20 })),
        );
        assert_arrays_eq!(decoded, expected, &mut ctx);
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
