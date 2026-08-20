// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Building an Elias-Fano array from a sorted primitive array, and taking it apart again.
//!
//! See [`crate::params`] for the layout both directions read and write.

use std::iter;

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
use vortex_buffer::Alignment;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BitIndexIterator;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::bitpack_compress::bitpack_encode_unchecked;

use crate::EliasFano;
use crate::EliasFanoArray;
use crate::EliasFanoData;
use crate::array::EliasFanoArraySlotsExt;
use crate::array::scalar_from_bits;
use crate::cursor::position_of_rank;
use crate::params;

/// Encode a sorted, non-nullable integer array with Elias-Fano.
///
/// The input must be monotonically non-decreasing; duplicates are fine and cost one set bit each.
/// Nulls are rejected, because a null has no position in an ordering and the layout has nowhere to
/// put one. `ctx` is taken for symmetry with the other integer encoders; encoding does not use it.
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
    let lower_width = params::lower_width(span, n);
    let upper_len = params::upper_len(span, n, lower_width)?;
    let lower_mask = params::lower_mask(lower_width);

    let mut upper = UpperBuilder::new(usize::try_from(upper_len)?);
    let mut lower = BufferMut::<u64>::with_capacity(n);

    match_each_integer_ptype!(array.ptype(), |P| {
        let values = array.as_slice::<P>();
        // Check monotonicity in the value domain: an element is a modular difference, so on
        // unsorted input the differences can still come out non-decreasing after wrapping.
        let mut previous = values[0];
        for (index, &value) in values.iter().enumerate() {
            vortex_ensure!(
                value >= previous,
                "Elias-Fano requires a non-decreasing sequence, but the value at index {index} \
                 is below its predecessor"
            );
            previous = value;

            let element = (value as u64).wrapping_sub(reference_bits);
            let rank = index as u64;
            upper.push(rank, (element >> lower_width) + rank + 1);
            lower.push(element & lower_mask);
        }
    });

    let (upper_bytes, sample_bytes) = upper.finish(n as u64, upper_len)?;

    let lower = pack_lower(lower.freeze(), lower_width, n)?;

    let data = EliasFanoData::try_new(
        upper_bytes,
        sample_bytes,
        scalar_from_bits(&dtype, reference_bits)?,
        scalar_from_bits(&dtype, max_bits)?,
        lower_width,
        upper_len,
        0,
    )?;
    EliasFano::try_new(data, lower, n)
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
    let upper = array.upper_bits()?;
    let upper_len = usize::try_from(array.upper_len())?;
    let (_, samples1) = array.sample_bytes()?;

    // Trim the upper array to the window holding exactly our elements' set bits, so the walk below
    // needs no per-element bound check and no early exit. Two sampled selects buy that.
    let start = position_of_rank(&upper, samples1, upper_len, first_rank)?;
    let end = position_of_rank(&upper, samples1, upper_len, first_rank + len as u64 - 1)? + 1;
    let window = upper.slice(start..end);

    let lower_width = array.lower_width();
    let fold = Fold {
        start,
        first_rank,
        reference_bits: array.reference_bits(),
        lower_width,
        lower_mask: params::lower_mask(lower_width),
    };

    Ok(match_each_integer_ptype!(ptype, |P| {
        PrimitiveArray::new(
            fold.decode::<P>(&window, array.lower(), len, ctx)?,
            Validity::NonNullable,
        )
    }))
}

/// Reassembles elements from the two halves of the layout, in the column's own width.
struct Fold {
    /// Bit position the upper window starts at, which its set-bit indices are relative to.
    start: usize,
    first_rank: u64,
    reference_bits: u64,
    lower_width: u8,
    lower_mask: u64,
}

impl Fold {
    /// Decode `len` elements, taking high parts from `window`'s set bits and low parts from
    /// `lower`, which is read one FastLanes block at a time.
    fn decode<P: NativePType>(
        &self,
        window: &BitBuffer,
        lower: &ArrayRef,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Buffer<P>>
    where
        u64: AsPrimitive<P>,
    {
        let mut values = BufferMut::<P>::with_capacity(len);
        let mut ones = window.set_indices();

        if self.lower_width == 0 {
            // Nothing is stored, so do not execute the slot just to read `len` zeros.
            self.segment(&mut values, &mut ones, iter::repeat_n(0, len))?;
            return self.finish(values, ones, len);
        }

        // Window before reading: the child spans the whole encoded sequence, and a decode only ever
        // wants the ranks inside its own slice.
        let first = usize::try_from(self.first_rank)?;
        let window_lower = lower.slice(first..first + len)?;

        // The alignment test is the cursor's, for the same reason: the unpack reinterprets the
        // packed bytes as `&[u64]` unchecked, so an under-aligned buffer must take the fallback.
        if let Some(packed) = window_lower.as_opt::<BitPacked>()
            && packed.patches().is_none()
            && packed
                .packed()
                .as_host_opt()
                .is_some_and(|buffer| buffer.is_aligned(Alignment::of::<u64>()))
        {
            let mut chunks = packed.unpacked_chunks::<u64>()?;
            if let Some(initial) = chunks.initial() {
                self.segment(&mut values, &mut ones, initial.iter().copied())?;
            }
            // A single-block child is covered by `initial` alone, and the later phases would hand
            // that same block back.
            if values.len() < len {
                let mut full = chunks.full_chunks();
                while let Some(chunk) = full.next() {
                    self.segment(&mut values, &mut ones, chunk.iter().copied())?;
                }
            }
            if values.len() < len
                && let Some(trailer) = chunks.trailer()
            {
                self.segment(&mut values, &mut ones, trailer.iter().copied())?;
            }
        } else {
            // The slot is patched, device-resident, or some other encoding after a rewrite.
            let dense = window_lower.execute::<PrimitiveArray>(ctx)?;
            self.segment(&mut values, &mut ones, dense.as_slice::<u64>().iter().copied())?;
        }

        self.finish(values, ones, len)
    }

    /// Fold one run of consecutive low parts onto the end of `values`.
    fn segment<P: NativePType, L: Iterator<Item = u64>>(
        &self,
        values: &mut BufferMut<P>,
        ones: &mut BitIndexIterator<'_>,
        lows: L,
    ) -> VortexResult<()>
    where
        u64: AsPrimitive<P>,
    {
        for low in lows {
            let rank = self.first_rank + values.len() as u64;
            let position = ones.next().ok_or_else(|| {
                vortex_err!("Elias-Fano upper array holds no element of rank {rank}")
            })?;
            let position = self.start + position;
            // The inverse of the encoder's `position = (element >> lower_width) + rank + 1`.
            let high = (position as u64).checked_sub(rank + 1).ok_or_else(|| {
                vortex_err!(
                    "Elias-Fano upper array is malformed: the element of rank {rank} sits at bit \
                     {position}, at or below its own rank"
                )
            })?;
            // The low bits are masked for the same reason as in the cursor's `lower_at`: only a
            // bit-packed child's width is checkable at construction, so a patched or rewritten slot
            // could otherwise carry bits above `lower_width` into the high part.
            let bits = self
                .reference_bits
                .wrapping_add((high << self.lower_width) | (low & self.lower_mask));
            // Truncating the pattern to the column's width is exactly the two's complement result,
            // signed or unsigned, because the reference was added in the same modular arithmetic.
            values.push(bits.as_());
        }
        Ok(())
    }

    /// The window holds exactly `len` set bits, and the child exactly `len` low parts, for any
    /// array this crate builds — but `validate` does not check the upper buffer's contents, so a
    /// corrupt file can hold a different number of either.
    fn finish<P: NativePType>(
        &self,
        values: BufferMut<P>,
        mut ones: BitIndexIterator<'_>,
        len: usize,
    ) -> VortexResult<Buffer<P>> {
        vortex_ensure!(
            values.len() == len && ones.next().is_none(),
            "Elias-Fano upper array is malformed: expected exactly {len} set bits above their own \
             ranks, found {}",
            values.len()
        );
        Ok(values.freeze())
    }
}

/// Builds the upper array and both sample tables together, in one pass over the elements.
///
/// The tables have to be built here: a zero-sample is the position of a sampled *unset* bit, and
/// the unset runs are only known in order between two consecutive elements as they are written.
struct UpperBuilder {
    bits: BitBufferMut,
    samples0: BufferMut<u64>,
    samples1: BufferMut<u64>,
    /// The next unset-bit rank owed a sample. Sample 0 is never stored, for either table: the
    /// sentinel puts the 0th unset bit at position 0 and the 0th set bit is the array's first, both
    /// of which a reader can assume.
    next_zero_sample: u64,
}

impl UpperBuilder {
    fn new(upper_len: usize) -> Self {
        Self {
            bits: BitBufferMut::new_unset(upper_len),
            samples0: BufferMut::empty(),
            samples1: BufferMut::empty(),
            next_zero_sample: 1 << params::LOG_SAMPLING0,
        }
    }

    /// Record the element of rank `rank` as a set bit at `position`.
    ///
    /// Must be called with strictly increasing `rank` and `position`.
    fn push(&mut self, rank: u64, position: u64) {
        // Every unset bit below `position` but above the previous element's has exactly `rank` set
        // bits before it, so its own rank is `its position - rank`. Emit a sample for each sampled
        // rank that lands in that gap.
        while self.next_zero_sample + rank < position {
            self.samples0.push(self.next_zero_sample + rank);
            self.next_zero_sample += 1 << params::LOG_SAMPLING0;
        }

        debug_assert!(position < self.bits.len() as u64, "position out of bounds");
        self.bits.set(position as usize);

        if rank > 0 && rank.is_multiple_of(1 << params::LOG_SAMPLING1) {
            self.samples1.push(position);
        }
    }

    /// Close the array, returning the upper bytes and both sample tables packed into one buffer.
    ///
    /// The seam is not returned: a reader recomputes it from the universe (see
    /// [`params::num_samples0`]), and `validate_parts` fails the array if the two disagree.
    fn finish(mut self, n: u64, upper_len: u64) -> VortexResult<(ByteBuffer, ByteBuffer)> {
        // Sample the trailing unset bits past the last element, which `push` never reached. These
        // are the bucket boundaries above the maximum element's high part, plus the guard zero.
        while self.next_zero_sample + n < upper_len {
            self.samples0.push(self.next_zero_sample + n);
            self.next_zero_sample += 1 << params::LOG_SAMPLING0;
        }

        // Both tables share one buffer, zeros first. A reader gets a single zero-copy mapping, and
        // a query that reseats and then walks touches both within a few cache lines of each other.
        self.samples0.extend_from_slice(self.samples1.as_slice());

        let (_, _, upper_bytes) = self.bits.freeze().into_inner();
        Ok((upper_bytes, self.samples0.freeze().into_byte_buffer()))
    }
}

fn pack_lower(lower: Buffer<u64>, lower_width: u8, n: usize) -> VortexResult<ArrayRef> {
    if lower_width == 0 {
        // There is nothing to store. A constant array says so explicitly, and costs nothing on
        // disk.
        return Ok(ConstantArray::new(0u64, n).into_array());
    }
    let lower = PrimitiveArray::new(lower, Validity::NonNullable);
    // SAFETY: every value was masked to `lower_width` bits as it was pushed, so all pack losslessly
    // and none needs a patch. The checked path would scan for a minimum and build a bit-width
    // histogram to rediscover what the encoder already guaranteed.
    Ok(unsafe { bitpack_encode_unchecked(lower, lower_width) }?.into_array())
}


/// The degenerate zero-element array.
///
/// This case is representable rather than rejected, so an empty chunk needs no special handling
/// upstream. The bounds go unused, because there is nothing to offset, and the two-bit upper array
/// holds just the sentinel and its guard.
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
