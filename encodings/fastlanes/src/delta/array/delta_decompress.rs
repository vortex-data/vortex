// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::mem::MaybeUninit;
use std::slice;

use fastlanes::BitPacking;
use fastlanes::Delta;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::DeltaArray;
use crate::FL_CHUNK_SIZE;
use crate::bit_transpose::untranspose_validity;
use crate::delta::array::DeltaArrayExt;

pub fn delta_decompress(
    array: &DeltaArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    // Fast path: unpack the bit-packed deltas one FastLanes chunk at a time, straight into the
    // un-delta + un-transpose pipeline. Falls back to the general path when its preconditions
    // do not hold.
    if let Some(decoded) = try_fused_delta_decompress(array, ctx)? {
        return Ok(decoded);
    }

    let bases = array.bases().clone().execute::<PrimitiveArray>(ctx)?;
    let deltas = array.deltas().clone().execute::<PrimitiveArray>(ctx)?;

    let start = array.offset();
    let end = start + array.len();

    let validity = untranspose_validity(&deltas.validity()?, ctx)?;
    let validity = validity.slice(start..end)?;

    let original_ptype = deltas.ptype();
    // Signed inputs are processed through their unsigned counterpart; `wrapping_add` on the
    // raw bytes inverts the `wrapping_sub` done at compress time regardless of signedness.
    let bases = bases.reinterpret_cast(original_ptype.to_unsigned());
    let deltas = deltas.reinterpret_cast(original_ptype.to_unsigned());

    let decoded = match_each_unsigned_integer_ptype!(deltas.ptype(), |T| {
        const LANES: usize = T::LANES;

        let buffer = decompress_primitive::<T, LANES>(bases.as_slice(), deltas.as_slice());
        let buffer = buffer.slice(start..end);

        PrimitiveArray::new(buffer, validity)
    });

    Ok(decoded.reinterpret_cast(original_ptype))
}

/// Attempt the fused decode path, which avoids materializing the full-length unpacked deltas.
///
/// Returns `Ok(None)` (so the caller falls back to the general path) unless every precondition
/// holds: the array is non-nullable, and the deltas child is a [`BitPacked`] array with no
/// patches and no physical offset. In that case the deltas are unpacked one chunk at a time into
/// a cache-resident scratch buffer that feeds directly into `undelta` + `untranspose`, removing a
/// full-length intermediate allocation and one pass over the data.
fn try_fused_delta_decompress(
    array: &DeltaArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<PrimitiveArray>> {
    let deltas = array.deltas();

    // Validity is stored on the deltas child. The fused path only handles the all-valid case so
    // it can skip the validity un-transpose entirely.
    if deltas.dtype().is_nullable() {
        return Ok(None);
    }

    // The deltas must be a bare bit-packed array. A `FoR` (or any other) wrapper is left to the
    // general path, which decodes the child first.
    let Some(bp) = deltas.as_opt::<BitPacked>() else {
        return Ok(None);
    };

    // A sliced bit-packed array (non-zero physical offset) does not map cleanly onto whole-chunk
    // unpacking; defer it to the general path.
    if bp.offset() != 0 {
        return Ok(None);
    }

    let original_ptype = PType::try_from(deltas.dtype())?;
    let unsigned_ptype = original_ptype.to_unsigned();
    let bit_width = bp.bit_width() as usize;

    // `delta_compress` always pads the deltas to a multiple of the FastLanes chunk size.
    let deltas_len = bp.len();
    debug_assert_eq!(deltas_len % FL_CHUNK_SIZE, 0);
    let num_chunks = deltas_len / FL_CHUNK_SIZE;

    let start = array.offset();
    let end = start + array.len();

    // Bases are tiny (`num_chunks * LANES` values); decoding them up front is cheap.
    let bases = array.bases().clone().execute::<PrimitiveArray>(ctx)?;
    let bases = bases.reinterpret_cast(unsigned_ptype);

    // Bit-packing may store out-of-width deltas as patches (sparse exceptions). They correct the
    // *unpacked* deltas, so they must be applied before `undelta`. Decode the index/value children
    // up front; both are small (one entry per exception).
    let (patch_indices, patch_values) = match bp.patches() {
        Some(patches) => {
            let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
            let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
            if !values.all_valid(ctx)? {
                return Ok(None);
            }
            let patch_offset = patches.offset();
            let indices: Vec<usize> = match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
                indices
                    .as_slice::<P>()
                    .iter()
                    .map(|&i| <P as AsPrimitive<usize>>::as_(i) - patch_offset)
                    .collect()
            });
            (indices, Some(values.reinterpret_cast(unsigned_ptype)))
        }
        None => (Vec::new(), None),
    };

    // The packed buffer is reinterpreted to the unsigned native type, mirroring the unsigned
    // domain used by the general path's `reinterpret_cast`.
    let packed: ByteBuffer = bp.packed().clone().unwrap_host();

    let decoded = match_each_unsigned_integer_ptype!(unsigned_ptype, |T| {
        const LANES: usize = T::LANES;

        // `(index, value)` exception pairs sorted by index so the chunk loop can sweep them with a
        // single cursor.
        let mut patches: Vec<(usize, T)> = match &patch_values {
            Some(values) => patch_indices
                .iter()
                .copied()
                .zip(values.as_slice::<T>().iter().copied())
                .collect(),
            None => Vec::new(),
        };
        patches.sort_unstable_by_key(|(index, _)| *index);

        let packed = reinterpret_packed::<T>(&packed);
        let buffer = fused_decompress_primitive::<T, LANES>(
            bases.as_slice(),
            packed,
            bit_width,
            num_chunks,
            &patches,
        );
        let buffer = buffer.slice(start..end);

        PrimitiveArray::new(buffer, Validity::NonNullable)
    });

    Ok(Some(decoded.reinterpret_cast(original_ptype)))
}

/// Reinterpret a byte buffer of bit-packed values as a slice of the unsigned native type `T`.
fn reinterpret_packed<T: NativePType>(packed: &ByteBuffer) -> &[T] {
    debug_assert_eq!(
        packed.as_ptr().align_offset(align_of::<T>()),
        0,
        "packed buffer must be aligned to {}",
        std::any::type_name::<T>()
    );
    let len = packed.len() / size_of::<T>();
    // SAFETY: The packed buffer is aligned to `T` (checked above in debug builds) and outlives the
    // returned slice, which borrows from it.
    unsafe { slice::from_raw_parts(packed.as_ptr().cast::<T>(), len) }
}

/// Performs the low-level delta decompression on primitive values.
///
/// All chunks must be full 1024-element chunks (deltas length must be a multiple of 1024).
pub(crate) fn decompress_primitive<T, const LANES: usize>(bases: &[T], deltas: &[T]) -> Buffer<T>
where
    T: NativePType + Delta + Transpose,
{
    let (chunks, remainder) = deltas.as_chunks::<1024>();
    debug_assert!(
        remainder.is_empty(),
        "deltas must be padded to a multiple of 1024"
    );
    // Use >= because cross-type casts (e.g. u32→u64) may produce more bases than the
    // target LANES requires. Only the first chunks.len() * LANES bases are used.
    assert!(bases.len() >= chunks.len() * LANES);

    // Allocate a result array.
    let mut output = BufferMut::with_capacity(deltas.len());
    let (output_chunks, _) = output.spare_capacity_mut().as_chunks_mut::<1024>();

    // Loop over all the chunks
    let mut transposed: [T; 1024] = [T::default(); 1024];
    for ((i, chunk), output_chunk) in chunks.iter().enumerate().zip_eq(output_chunks.iter_mut()) {
        Delta::undelta::<LANES>(
            chunk,
            unsafe { &*(bases[i * LANES..(i + 1) * LANES].as_ptr().cast()) },
            &mut transposed,
        );

        Transpose::untranspose(&transposed, unsafe {
            mem::transmute::<&mut [MaybeUninit<T>; 1024], &mut [T; 1024]>(output_chunk)
        });
    }

    unsafe { output.set_len(deltas.len()) };

    output.freeze()
}

/// Fused delta decompression: unpack each bit-packed chunk into a cache-resident scratch buffer
/// and feed it straight into `undelta` + `untranspose`, without a full-length intermediate for the
/// unpacked deltas.
///
/// `packed` holds `num_chunks` chunks of `1024 * bit_width / T::T` packed values each. `patches`
/// holds `(index, value)` exception pairs sorted by ascending index, applied to the unpacked deltas
/// before the cumulative sum.
pub(crate) fn fused_decompress_primitive<T, const LANES: usize>(
    bases: &[T],
    packed: &[T],
    bit_width: usize,
    num_chunks: usize,
    patches: &[(usize, T)],
) -> Buffer<T>
where
    T: NativePType + Delta + Transpose + BitPacking,
{
    let elems_per_chunk = FL_CHUNK_SIZE * bit_width / (8 * size_of::<T>());
    debug_assert_eq!(packed.len(), num_chunks * elems_per_chunk);
    assert!(bases.len() >= num_chunks * LANES);

    let mut output = BufferMut::with_capacity(num_chunks * FL_CHUNK_SIZE);
    let (output_chunks, _) = output.spare_capacity_mut().as_chunks_mut::<1024>();

    let mut patch_cursor = 0;
    let mut unpacked: [T; 1024] = [T::default(); 1024];
    let mut transposed: [T; 1024] = [T::default(); 1024];
    for (i, output_chunk) in (0..num_chunks).zip(output_chunks.iter_mut()) {
        let packed_chunk = &packed[i * elems_per_chunk..(i + 1) * elems_per_chunk];

        // SAFETY: `packed_chunk` is exactly `elems_per_chunk = 1024 * bit_width / T::T` elements
        // and `unpacked` is exactly 1024 elements, as required by `unchecked_unpack`.
        unsafe {
            BitPacking::unchecked_unpack(bit_width, packed_chunk, &mut unpacked);
        }

        // Apply any exceptions falling in this chunk before the cumulative sum.
        let chunk_end = (i + 1) * FL_CHUNK_SIZE;
        while let Some(&(index, value)) = patches.get(patch_cursor)
            && index < chunk_end
        {
            unpacked[index - i * FL_CHUNK_SIZE] = value;
            patch_cursor += 1;
        }

        Delta::undelta::<LANES>(
            &unpacked,
            unsafe { &*(bases[i * LANES..(i + 1) * LANES].as_ptr().cast()) },
            &mut transposed,
        );

        Transpose::untranspose(&transposed, unsafe {
            mem::transmute::<&mut [MaybeUninit<T>; 1024], &mut [T; 1024]>(output_chunk)
        });
    }

    unsafe { output.set_len(num_chunks * FL_CHUNK_SIZE) };

    output.freeze()
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::try_fused_delta_decompress;
    use crate::Delta;
    use crate::DeltaArray;
    use crate::bitpack_compress::bitpack_to_best_bit_width;
    use crate::delta_compress;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    /// Build a `Delta` array whose deltas child is a bare `BitPacked` array (the shape that
    /// triggers the fused decode path).
    fn bitpacked_delta(
        values: &PrimitiveArray,
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> VortexResult<DeltaArray> {
        let (bases, deltas) = delta_compress(values, ctx)?;
        let packed = bitpack_to_best_bit_width(&deltas, ctx)?;
        Delta::try_new(bases.into_array(), packed.into_array(), 0, values.len())
    }

    #[rstest]
    #[case::u32_one_chunk((0u32..1024).map(|i| i * 7 + i % 5).collect())]
    #[case::u32_many_chunks((0u32..8192).map(|i| i * 3 + i % 11).collect())]
    // Logical length not a multiple of the chunk size exercises the trailing slice.
    #[case::u32_jagged((0u32..5000).map(|i| i * 9 + i % 4).collect())]
    #[case::u64_many_chunks((0u64..4096).map(|i| i * 100 + i % 7).collect())]
    #[case::u8_full_chunk((0u8..=255).chain(0u8..=255).chain(0u8..=255).chain(0u8..=255).collect())]
    // Signed but monotonically increasing: deltas are non-negative so they bit-pack directly.
    #[case::i32_monotone((0i32..4096).map(|i| 1_000_000 + i * 5 + i % 3).collect())]
    #[case::i64_monotone((0i64..4096).map(|i| 1_700_000_000_000 + i * 13).collect())]
    fn fused_roundtrip(#[case] values: PrimitiveArray) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let delta = bitpacked_delta(&values, &mut ctx)?;

        // The fused fast path must actually be taken for this shape.
        assert!(
            try_fused_delta_decompress(&delta, &mut ctx)?.is_some(),
            "expected the fused path to be taken for bit-packed deltas"
        );

        let decoded = delta
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, values);
        Ok(())
    }

    #[test]
    fn fused_skips_unpacked_deltas() -> VortexResult<()> {
        // When the deltas child is a plain (non-bit-packed) primitive array, the fused path must
        // decline so the general path runs.
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0u32..4096).map(|i| i * 7));
        let delta = Delta::try_from_primitive_array(&values, &mut ctx)?;
        assert!(try_fused_delta_decompress(&delta, &mut ctx)?.is_none());
        // The general path must still produce the correct answer.
        let decoded = delta
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, values);
        Ok(())
    }

    #[test]
    fn fused_skips_nullable() -> VortexResult<()> {
        // Nullable arrays carry transposed validity and must use the general path.
        let mut ctx = SESSION.create_execution_ctx();
        let values =
            PrimitiveArray::from_option_iter((0u32..4096).map(|i| (i % 3 != 0).then_some(i * 2)));
        let (bases, deltas) = delta_compress(&values, &mut ctx)?;
        let packed = bitpack_to_best_bit_width(&deltas, &mut ctx)?;
        let delta = Delta::try_new(bases.into_array(), packed.into_array(), 0, values.len())?;
        assert!(try_fused_delta_decompress(&delta, &mut ctx)?.is_none());
        let decoded = delta
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, values);
        Ok(())
    }
}
