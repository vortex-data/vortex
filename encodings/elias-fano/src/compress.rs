// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Building an Elias-Fano array from a sorted primitive array, and taking it apart again.
//!
//! See [`crate::ef`] for the layout both directions read and write.

use std::mem::MaybeUninit;

use lending_iterator::prelude::LendingIterator;
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::match_each_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::FL_CHUNK_SIZE;
use vortex_fastlanes::bitpack_compress::bitpack_encode_unchecked;

use crate::EliasFano;
use crate::EliasFanoArray;
use crate::EliasFanoData;
use crate::array::EliasFanoArraySlotsExt;
use crate::array::scalar_from_bits;
use crate::ef;
use crate::lower::materialise;
use crate::lower::readable_in_place;
use crate::malformed;

/// Encode a sorted, non-nullable integer array with Elias-Fano.
///
/// The input must be monotonically non-decreasing; duplicates are fine and cost one set bit each.
/// Nulls are rejected: the layout has nowhere to put one. `ctx` is taken for symmetry with the
/// other integer encoders and goes unused.
// Values widen into the 64-bit element domain here, a no-op in the `u64` arm the lint sees.
#[expect(clippy::unnecessary_cast)]
pub fn elias_fano_encode(
    array: ArrayView<'_, Primitive>,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<EliasFanoArray> {
    let dtype = array.dtype().clone();
    vortex_ensure!(
        dtype.is_int(),
        "Elias-Fano requires an integer dtype, got {dtype}"
    );
    vortex_ensure!(
        !dtype.is_nullable(),
        "Elias-Fano requires a non-nullable dtype, got {dtype}"
    );

    let n = array.len();
    if n == 0 {
        return empty(&dtype);
    }

    // Work in sign-extended 64-bit patterns throughout; see `EliasFanoData::reference_bits`.
    let (reference_bits, max_bits) = match_each_integer_ptype!(array.ptype(), |P| {
        let values = array.as_slice::<P>();
        (values[0] as u64, values[n - 1] as u64)
    });

    let span = max_bits.wrapping_sub(reference_bits);
    let encoded = match_each_integer_ptype!(array.ptype(), |P| {
        let values = array.as_slice::<P>();
        ensure_non_decreasing(values)?;
        ef::encode(
            values
                .iter()
                .map(|&value| (value as u64).wrapping_sub(reference_bits)),
            span,
        )
    })
    .map_err(crate::unrepresentable)?;

    // Every buffer adopts its `Vec`'s allocation rather than copying; see `Buffer: From<Vec<T>>`.
    let lower = pack_lower(Buffer::from(encoded.lower), encoded.lower_width, n)?;

    let data = EliasFanoData::try_new(
        ByteBuffer::from(encoded.upper),
        Buffer::from(encoded.samples).into_byte_buffer(),
        scalar_from_bits(&dtype, reference_bits)?,
        scalar_from_bits(&dtype, max_bits)?,
        encoded.lower_width,
        encoded.upper_len,
        0,
    )?;
    EliasFano::try_new(data, lower, n)
}

/// Reject a sequence that decreases anywhere.
///
/// Checked in the **value** domain, before the reference is subtracted: an element is a modular
/// difference, so unsorted input can still yield non-decreasing elements after wrapping, and
/// [`ef::encode`] would build a layout no reader could fault.
fn ensure_non_decreasing<P: NativePType + PartialOrd>(values: &[P]) -> VortexResult<()> {
    if values.is_sorted() {
        return Ok(());
    }
    let index = values
        .windows(2)
        .position(|pair| pair[1] < pair[0])
        .map_or(0, |first| first + 1);
    vortex_bail!(
        "Elias-Fano requires a non-decreasing sequence, but the value at index {index} is below \
         its predecessor"
    )
}

pub(crate) fn elias_fano_decompress(
    array: &EliasFanoArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let len = array.len();
    let ptype = array.dtype().as_ptype();
    if len == 0 {
        return Ok(match_each_integer_ptype!(ptype, |P| {
            PrimitiveArray::empty::<P>(NonNullable)
        }));
    }

    let first_rank = array.first_rank();
    let bits = ef::Bits::new(
        array.upper_buffer().as_slice(),
        0,
        usize::try_from(array.upper_len())?,
    );
    let (_, samples1) = array.sample_bytes()?;

    // Trim the upper array to the window holding exactly our elements' set bits, so the walk below
    // needs no per-element bound check and no early exit. Two sampled selects buy that.
    let start = ef::position_of_rank(bits, samples1, first_rank).map_err(malformed)?;
    let end =
        ef::position_of_rank(bits, samples1, first_rank + len as u64 - 1).map_err(malformed)? + 1;
    let words = ef::window_words(bits, start, end);

    Ok(match_each_integer_ptype!(ptype, |P| {
        PrimitiveArray::new(
            decode::<P>(array, &words, start, ctx)?,
            Validity::NonNullable,
        )
    }))
}

/// Decode `len` elements into the column's own width, feeding [`ef::Decoder`] the low bits one
/// FastLanes block at a time.
///
/// Decides only the shape of the low-bits child: whether its packed bytes can be unpacked a block
/// at a time, or have to be materialised first.
fn decode<P: NativePType>(
    array: &EliasFanoArray,
    words: &[u64],
    start: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<P>>
where
    u64: AsPrimitive<P>,
{
    let len = array.len();
    let first_rank = array.first_rank();
    let reference_bits = array.reference_bits();
    let lower_width = array.lower_width();

    let mut decoder =
        ef::Decoder::new(words, start, first_rank, len, lower_width).map_err(malformed)?;

    let mut values = BufferMut::<P>::zeroed(len);

    if lower_width == 0 {
        // Nothing is stored, so do not execute the slot just to read `len` zeros.
        segment(&mut decoder, &mut values, reference_bits, None);
    } else {
        let first = usize::try_from(first_rank)?;
        let window_lower = array.lower().slice(first..first + len)?;

        if let Some(packed) = readable_in_place(&window_lower) {
            let mut scratch = [const { MaybeUninit::<u64>::uninit() }; FL_CHUNK_SIZE];
            let mut chunks = packed.unpacked_chunks::<u64>(&mut scratch)?;
            if let Some(initial) = chunks.initial() {
                segment(&mut decoder, &mut values, reference_bits, Some(initial));
            }
            if decoder.remaining() > 0 {
                let mut full = chunks.full_chunks();
                while let Some(chunk) = full.next() {
                    segment(&mut decoder, &mut values, reference_bits, Some(chunk));
                }
            }
            if decoder.remaining() > 0
                && let Some(trailer) = chunks.trailer()
            {
                segment(&mut decoder, &mut values, reference_bits, Some(trailer));
            }
        } else {
            // The slot is patched, device-resident, or some other encoding after a rewrite.
            let dense = materialise(window_lower, ctx)?;
            segment(
                &mut decoder,
                &mut values,
                reference_bits,
                Some(dense.as_slice()),
            );
        }
    }

    decoder.finish().map_err(malformed)?;
    Ok(values.freeze())
}

/// One run of low parts, folded into the column's own width.
fn segment<P: NativePType>(
    decoder: &mut ef::Decoder<'_>,
    values: &mut BufferMut<P>,
    reference_bits: u64,
    lows: Option<&[u64]>,
) where
    u64: AsPrimitive<P>,
{
    let out = values.as_mut_slice();
    decoder.segment(lows, |index, element| {
        // Truncating the pattern to the column's width is exactly the two's complement result,
        // signed or unsigned, because the reference was added in the same modular arithmetic.
        out[index] = reference_bits.wrapping_add(element).as_();
    });
}

fn pack_lower(lower: Buffer<u64>, lower_width: u8, n: usize) -> VortexResult<ArrayRef> {
    if lower_width == 0 {
        // Nothing to store, and a constant array costs nothing on disk.
        return Ok(ConstantArray::new(0u64, n).into_array());
    }
    let lower = PrimitiveArray::new(lower, Validity::NonNullable);
    // SAFETY: every value was masked to `lower_width` bits as it was pushed, so all pack losslessly
    // and none needs a patch. The checked path would scan for a minimum and build a bit-width
    // histogram to rediscover what the encoder already guaranteed.
    Ok(unsafe { bitpack_encode_unchecked(lower, lower_width) }?.into_array())
}

/// The degenerate zero-element array, representable rather than rejected so an empty chunk needs no
/// special handling upstream.
///
/// The bounds go unused, there being nothing to offset, and the two-bit upper array holds just the
/// sentinel and its guard.
fn empty(dtype: &DType) -> VortexResult<EliasFanoArray> {
    let upper = BitBufferMut::new_unset(2).freeze();
    let (_, _, upper_bytes) = upper.into_inner();
    let data = EliasFanoData::try_new(
        upper_bytes,
        Buffer::<u64>::empty().into_byte_buffer(),
        scalar_from_bits(dtype, 0)?,
        scalar_from_bits(dtype, 0)?,
        0,
        2,
        0,
    )?;
    EliasFano::try_new(
        data,
        PrimitiveArray::empty::<u64>(NonNullable).into_array(),
        0,
    )
}
