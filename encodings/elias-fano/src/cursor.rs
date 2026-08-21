// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Random access and predecessor search over an encoded sequence, all in `O(1)`: `access` reads the
//! value at an index, `rank` counts the values strictly below a probe, and `next_geq` finds the
//! first value at or above one.
//!
//! The cursor is stateful on purpose. Probes usually arrive in ascending order — merge joins,
//! intersections, scans of list offsets — so it remembers where it stopped and walks forward from
//! there when the next probe is close, touching neither sample table. See [`crate::params`] for the
//! layout these operations read.

use fastlanes::BitPacking;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_buffer::Alignment;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::FL_CHUNK_SIZE;
use vortex_fastlanes::bitpack_decompress::unpack_single_primitive;

use crate::EliasFano;
use crate::array::EliasFanoSlotsView;
use crate::array::read_sample;
use crate::array::scalar_bits;
use crate::array::scalar_from_bits;
use crate::params::LINEAR_SCAN_THRESHOLD;
use crate::params::LOG_SAMPLING0;
use crate::params::LOG_SAMPLING1;
use crate::params::lower_mask;

/// A whole FastLanes block is bulk-unpacked once *more* than this many reads have landed inside it.
///
/// Unpacking all 1024 values costs roughly nine single-value unpacks, so that is the crossover.
/// Mirrors the crate-private `UNPACK_CHUNK_THRESHOLD` in `vortex-fastlanes`.
const BULK_UNPACK_THRESHOLD: usize = 8;

/// The bit position of the set bit belonging to absolute rank `rank`, as a sampled `select1`.
///
/// `samples1` bounds the search window at `1 << LOG_SAMPLING1` ones, about one cache line. Sample 0
/// is not stored: rank 0 is always the first set bit. Free-standing because the bulk decode needs
/// the window boundaries without a cursor's low-bits reader.
pub(crate) fn position_of_rank(
    upper: &BitBuffer,
    samples1: &[u8],
    upper_len: usize,
    rank: u64,
) -> VortexResult<usize> {
    let sample = (rank >> LOG_SAMPLING1) as usize;
    let start = if sample == 0 {
        0
    } else {
        usize::try_from(read_sample(samples1, sample - 1))?
    };
    let nth = usize::try_from(rank - ((sample as u64) << LOG_SAMPLING1))?;
    upper
        .select_range(start, upper_len, nth)
        .map(|offset| start + offset)
        .ok_or_else(|| vortex_err!("Elias-Fano upper array holds no element of rank {rank}"))
}

/// Where a probe value falls relative to the encoded universe.
enum Bound {
    /// The probe sits below the reference, so it is below every element and its rank is 0.
    Below,
    /// The probe sits inside `reference..=max`, and this is its offset from the reference.
    Inside(u64),
    /// The probe sits above the encoded maximum, so it is above every element and its rank is the
    /// array's length.
    Above,
}

/// A FastLanes-packed low-bits child that can be read in place, with the offsets a read needs.
///
/// [`packed_in_place`] is the only constructor, so the conditions that make an in-place read sound
/// are stated once and both readers — the cursor and [`access_at`] — inherit them.
#[derive(Clone, Copy)]
struct PackedLower<'a> {
    packed: &'a [u64],
    bit_width: usize,
    /// The child's own sub-block offset, which `unpack_single_primitive` does not apply itself
    /// (unlike `unpack_single`). Forgetting it is a silent wrong answer, not a panic.
    child_offset: usize,
}

impl PackedLower<'_> {
    /// Where in the child the element of absolute rank `rank` sits.
    #[inline]
    fn index_of(&self, rank: u64) -> usize {
        rank as usize + self.child_offset
    }

    /// One value unpacked on its own, without a scratch block.
    #[inline]
    fn unpack_one(&self, index: usize) -> u64 {
        // SAFETY: `packed` is `BitPackedData`'s own buffer, whose length the child's validation
        // already tied to `bit_width` and a whole number of blocks, and `index` is within the
        // child's length because `validate_parts` ties `first_rank + len` to it.
        unsafe { unpack_single_primitive::<u64>(self.packed, self.bit_width, index) }
    }
}

/// The low-bits child as a packed slice, if it can be read in place: FastLanes-packed, unpatched,
/// host-resident and `u64`-aligned. `None` means the low bits have to be materialised instead.
fn packed_in_place(lower: &ArrayRef) -> Option<PackedLower<'_>> {
    // `as_opt`, never `as_`: with the experimental patched-array plugin enabled the slot comes back
    // from a file as `Patched(BitPacked)`, and a rewrite may replace it outright.
    let packed = lower.as_opt::<BitPacked>()?;
    if packed.patches().is_some()
        || !packed
            .packed()
            .as_host_opt()
            .is_some_and(|buffer| buffer.is_aligned(Alignment::of::<u64>()))
    {
        return None;
    }
    Some(PackedLower {
        // `.data()` rather than the `Deref`, which would borrow the view rather than the array
        // behind it and so not live long enough.
        packed: packed.data().packed_slice::<u64>(),
        bit_width: packed.bit_width() as usize,
        child_offset: packed.offset() as usize,
    })
}

/// The value at logical `index`, read without building a cursor.
///
/// A cursor earns its setup back over a stream of probes — the seat, the memoised answer and the
/// bulk-unpack scratch all amortise — and every batched path builds one and reuses it. A point
/// lookup through `OperationsVTable::scalar_at` amortises none of it, so it does just the two reads
/// an access needs: one sampled `select1` for the high part, and one low-bits read.
///
/// The two readers must agree, and share only the layout; `tests::check_access` runs them against
/// each other over every shape the suite covers.
pub(crate) fn access_at(
    array: ArrayView<'_, EliasFano>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let len = array.len();
    if index >= len {
        vortex_bail!(OutOfBounds: index, 0usize, len);
    }
    let data = array.data();
    let lower_width = data.lower_width();
    let rank = data.first_rank() + index as u64;

    let (_, samples1) = data.sample_bytes()?;
    let upper = data.upper_bits()?;
    let upper_len = usize::try_from(data.upper_len())?;
    let position = position_of_rank(&upper, samples1, upper_len, rank)?;
    // The inverse of the encoder's `position = (element >> lower_width) + rank + 1`. Checked for
    // the same reason as in `EliasFanoCursor::reseat`: the upper buffer's contents are never
    // validated, so a corrupt one has to raise rather than underflow.
    let high = (position as u64).checked_sub(rank + 1).ok_or_else(|| {
        vortex_err!(
            "Elias-Fano upper array is malformed: the element of rank {rank} sits at bit \
             {position}, at or below its own rank"
        )
    })?;

    // The slots view borrows the array behind the `ArrayView`; the `lower()` accessor would borrow
    // the (`Copy`, stack-local) view itself.
    let lower = EliasFanoSlotsView::from_slots(array.slots()).lower;
    let element = (high << lower_width) | lower_at_rank(lower, lower_width, rank, ctx)?;
    scalar_from_bits(array.dtype(), data.reference_bits().wrapping_add(element))
}

/// The low bits of the element at absolute rank `rank`, for a reader that will only ask once.
///
/// Masked for the same reason as [`EliasFanoCursor::lower_at`]: only a bit-packed child's width is
/// checkable at construction, so a patched or rewritten slot can carry bits above `lower_width`,
/// and those would bleed into the high part.
///
/// The fallback windows to the single rank rather than to the whole slice as
/// [`LowerBits::try_new`] does, so one probe against a child that cannot be read in place
/// materialises one value rather than `len` of them.
fn lower_at_rank(
    lower: &ArrayRef,
    lower_width: u8,
    rank: u64,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    if lower_width == 0 {
        return Ok(0);
    }
    let bits = match packed_in_place(lower) {
        Some(packed) => packed.unpack_one(packed.index_of(rank)),
        None => {
            let index = usize::try_from(rank)?;
            lower
                .slice(index..index + 1)?
                .execute::<PrimitiveArray>(ctx)?
                .into_buffer::<u64>()[0]
        }
    };
    Ok(bits & lower_mask(lower_width))
}

/// How the low bits of each element can be read.
enum LowerBits<'a> {
    /// The width is zero, so there are no low bits to read and nothing is stored.
    Zero,
    /// The normal case, where the low bits are read straight out of the FastLanes-packed child in
    /// place.
    Packed(PackedLower<'a>),
    /// The fallback for a child that cannot be read in place — patches, device memory,
    /// under-aligned, or a slot some rewrite replaced. The low bits are materialised once, up
    /// front.
    Dense {
        /// Low bits for ranks `base..base + values.len()` only, not the whole child, so a slice
        /// near the end of a long sequence does not materialise everything before it.
        values: Buffer<u64>,
        /// The absolute rank `values[0]` holds, i.e. the array's `first_rank`.
        base: u64,
    },
}

/// Where the cursor currently sits, which is one element together with everything already known
/// about it.
struct Seat {
    /// Bit position of this element's set bit in the upper array.
    position: usize,
    /// Absolute rank within the encoded sequence, so `first_rank` is already included.
    rank: u64,
    /// The element, i.e. the value minus the reference.
    element: u64,
}

/// A stateful reader over an [`EliasFanoArray`](crate::EliasFanoArray).
pub struct EliasFanoCursor<'a> {
    upper: BitBuffer,
    /// Position of every `1 << LOG_SAMPLING0`-th unset bit, as raw little-endian `u64`s.
    samples0: &'a [u8],
    /// Position of every `1 << LOG_SAMPLING1`-th set bit, as raw little-endian `u64`s.
    samples1: &'a [u8],
    lower: LowerBits<'a>,
    /// Destination for a bulk-unpacked FastLanes block, together with which block it currently
    /// holds. It is allocated on the first bulk unpack, so a single point lookup never pays for it.
    scratch: Option<Box<[u64; FL_CHUNK_SIZE]>>,
    scratch_chunk: Option<usize>,
    /// The block the last few reads landed in, and how many landed there. Together they drive the
    /// switch over to [`BULK_UNPACK_THRESHOLD`].
    hot_chunk: Option<usize>,
    hot_reads: usize,
    dtype: &'a DType,
    ptype: PType,
    reference_bits: u64,
    span: u64,
    lower_width: u8,
    first_rank: u64,
    len: usize,
    upper_len: usize,
    seat: Option<Seat>,
    /// The last probe [`Self::next_geq_element`] answered, and the answer it gave. A merge join
    /// probes the same value from both sides, so the immediate repeat is common.
    ///
    /// Memoising the answer is not the same as reusing the seat: with duplicates a rank must count
    /// from the first occurrence, and the found element is not generally the probe.
    last_answer: Option<(u64, (usize, Option<u64>))>,
}

impl<'a> EliasFanoCursor<'a> {
    /// Open a cursor over `array`. `ctx` is only used if the low-bits child has to be materialised.
    pub fn try_new(
        array: ArrayView<'a, EliasFano>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<EliasFanoCursor<'a>> {
        let data = array.data();
        let (samples0, samples1) = data.sample_bytes()?;
        // The slots view borrows the array behind the `ArrayView`, which outlives the cursor; the
        // `lower()` accessor would borrow the (`Copy`, stack-local) view itself.
        let lower = EliasFanoSlotsView::from_slots(array.slots()).lower;
        let lower = LowerBits::try_new(
            lower,
            data.lower_width(),
            data.first_rank(),
            array.len(),
            ctx,
        )?;

        Ok(Self {
            upper: data.upper_bits()?,
            samples0,
            samples1,
            lower,
            scratch: None,
            scratch_chunk: None,
            hot_chunk: None,
            hot_reads: 0,
            dtype: array.array().dtype(),
            ptype: array.array().dtype().as_ptype(),
            reference_bits: data.reference_bits(),
            span: data.span(),
            lower_width: data.lower_width(),
            first_rank: data.first_rank(),
            len: array.len(),
            upper_len: usize::try_from(data.upper_len())?,
            seat: None,
            last_answer: None,
        })
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The value at logical index `index`.
    pub fn access(&mut self, index: usize) -> VortexResult<Scalar> {
        let element = self.access_element(index)?;
        scalar_from_bits(self.dtype, self.reference_bits.wrapping_add(element))
    }

    /// The element (value minus reference) at logical index `index`, in one sampled `select1`.
    pub(crate) fn access_element(&mut self, index: usize) -> VortexResult<u64> {
        if index >= self.len {
            vortex_bail!(OutOfBounds: index, 0usize, self.len);
        }
        let rank = self.first_rank + index as u64;

        // Sequential access is the common case, and stepping to the next set bit is cheaper than
        // a select from the sample table, so reuse the seat when it lines up.
        match &self.seat {
            Some(seat) if seat.rank == rank => {}
            Some(seat) if seat.rank + 1 == rank => self.advance()?,
            _ => self.seek(rank)?,
        }
        Ok(self.seated().element)
    }

    /// The number of elements strictly less than `value`.
    ///
    /// This is `search_sorted`'s left bound, and needs no rank directory: see the
    /// `rank1(select0(h)) == select0(h) - h` identity in the crate-private `params` module.
    pub fn rank(&mut self, value: &Scalar) -> VortexResult<usize> {
        Ok(self.next_geq(value)?.0)
    }

    /// The number of elements at or below `value`, i.e. `search_sorted`'s right bound. With
    /// [`Self::rank`] this brackets the run of elements equal to `value`.
    ///
    /// The count is the rank of `value`'s successor, taken in the element domain because the value
    /// domain would overflow at the top of the ptype.
    pub fn rank_inclusive(&mut self, value: &Scalar) -> VortexResult<usize> {
        match self.locate(value)? {
            Bound::Below => Ok(0),
            Bound::Above => Ok(self.len),
            Bound::Inside(element) => match element.checked_add(1) {
                Some(successor) => Ok(self.next_geq_element(successor)?.0),
                None => Ok(self.len),
            },
        }
    }

    /// The first value at or above `value`, together with the number of elements strictly below it.
    ///
    /// Returns `(len, None)` when every element is below `value`.
    pub fn next_geq(&mut self, value: &Scalar) -> VortexResult<(usize, Option<Scalar>)> {
        let (rank, element) = match self.locate(value)? {
            Bound::Below => (
                0,
                (!self.is_empty())
                    .then(|| self.access_element(0))
                    .transpose()?,
            ),
            Bound::Above => (self.len, None),
            Bound::Inside(element) => self.next_geq_element(element)?,
        };
        let value = element
            .map(|element| scalar_from_bits(self.dtype, self.reference_bits.wrapping_add(element)))
            .transpose()?;
        Ok((rank, value))
    }

    /// [`Self::next_geq`] in the element domain. `element` must be no greater than the span;
    /// callers holding a value want [`Self::next_geq`], which classifies it first.
    pub(crate) fn next_geq_element(&mut self, element: u64) -> VortexResult<(usize, Option<u64>)> {
        if self.is_empty() {
            return Ok((0, None));
        }
        if element > self.span {
            return Ok((self.len, None));
        }
        if let Some((probe, answer)) = self.last_answer
            && probe == element
        {
            return Ok(answer);
        }

        let end_rank = self.first_rank + self.len as u64;

        // Stepping forward is only sound when the probe is strictly above the seated element: on
        // equality the seat may sit past earlier duplicates, and a rank counts from the first.
        // The walk is bounded in buckets, measurable up front, and in elements, which is what it
        // pays — a bucket of duplicates is arbitrarily many elements deep.
        if let Some(seat) = &self.seat
            && element > seat.element
            && (element >> self.lower_width) - (seat.element >> self.lower_width)
                < LINEAR_SCAN_THRESHOLD
            && let Some(answer) = self.walk_to(element, end_rank, LINEAR_SCAN_THRESHOLD)?
        {
            return Ok(self.memoise(element, answer));
        }

        let answer = self.search_bucket(element, end_rank)?;
        Ok(self.memoise(element, answer))
    }

    /// Locate the first element `>= element` from a standing start.
    ///
    /// One sampled `select0` finds where the probe's bucket begins; everything before it is below
    /// the probe. A bucket normally holds about one element, so a short walk usually ends it.
    /// Otherwise the bucket is a run of near-duplicates, and a second `select0` brackets it for
    /// bisection — inside a bucket every element shares a high part, so the low bits alone order it
    /// and a bisection step needs no `select`.
    fn search_bucket(&mut self, element: u64, end_rank: u64) -> VortexResult<(usize, Option<u64>)> {
        let high = element >> self.lower_width;
        let start = self.rank_of_bucket(high)?.max(self.first_rank);
        if start >= end_rank {
            return Ok((self.len, None));
        }
        self.seek(start)?;

        if let Some(answer) = self.walk_to(element, end_rank, LINEAR_SCAN_THRESHOLD)? {
            return Ok(answer);
        }

        // The rank the walk stopped *on* was never compared, so the bisection includes it. It may
        // also be the first of a later bucket, which the clamp turns into an empty range and leaves
        // as the answer — correct, since a greater high part is already above the probe.
        let low = element & lower_mask(self.lower_width);
        let mut lo = self.seated().rank;
        let mut hi = self.rank_of_bucket(high + 1)?.clamp(lo, end_rank);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.lower_at(mid) < low {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // `lo` is the first element at or above the probe: either one inside the bucket, or the
        // bucket's successor, which a greater high part already puts above the probe.
        if lo >= end_rank {
            return Ok((self.len, None));
        }
        self.seek(lo)?;
        Ok((self.relative_rank(lo), Some(self.seated().element)))
    }

    /// Step forward at most `budget` times looking for the first element `>= element`, leaving the
    /// cursor where it stopped. `None` means the budget ran out with everything still below the
    /// probe, so stepping is no longer the way to find it.
    fn walk_to(
        &mut self,
        element: u64,
        end_rank: u64,
        budget: u64,
    ) -> VortexResult<Option<(usize, Option<u64>)>> {
        for _ in 0..budget {
            let seat = self.seated();
            let (found, rank) = (seat.element, seat.rank);
            if found >= element {
                return Ok(Some((self.relative_rank(rank), Some(found))));
            }
            if rank + 1 >= end_rank {
                return Ok(Some((self.len, None)));
            }
            self.advance()?;
        }
        Ok(None)
    }

    fn memoise(&mut self, probe: u64, answer: (usize, Option<u64>)) -> (usize, Option<u64>) {
        self.last_answer = Some((probe, answer));
        answer
    }

    fn position_of(&self, rank: u64) -> VortexResult<usize> {
        position_of_rank(&self.upper, self.samples1, self.upper_len, rank)
    }

    /// The absolute rank of the first element whose high part is at least `high`: a sampled
    /// `select0` followed by the rank identity, windowed the same way as [`position_of_rank`].
    fn rank_of_bucket(&self, high: u64) -> VortexResult<u64> {
        let sample = (high >> LOG_SAMPLING0) as usize;
        let start = if sample == 0 {
            0
        } else {
            usize::try_from(read_sample(self.samples0, sample - 1))?
        };
        let nth = usize::try_from(high - ((sample as u64) << LOG_SAMPLING0))?;
        let position = self
            .upper
            .select_zero_range(start, self.upper_len, nth)
            .map(|offset| (start + offset) as u64)
            .ok_or_else(|| {
                vortex_err!("Elias-Fano upper array holds no bucket boundary for high part {high}")
            })?;
        // rank1(select0(high)) == select0(high) - high. Checked because the upper buffer's contents
        // are never validated: a corrupt one must raise rather than underflow.
        position.checked_sub(high).ok_or_else(|| {
            vortex_err!(
                "Elias-Fano upper array is malformed: bucket boundary for high part {high} sits at \
                 bit {position}, below its own rank"
            )
        })
    }

    fn seek(&mut self, rank: u64) -> VortexResult<()> {
        let position = self.position_of(rank)?;
        self.reseat(position, rank)
    }

    fn advance(&mut self) -> VortexResult<()> {
        let seat = self.seated();
        let (from, rank) = (seat.position + 1, seat.rank + 1);
        let offset = self
            .upper
            .select_range(from, self.upper_len, 0)
            .ok_or_else(|| vortex_err!("Elias-Fano upper array holds no element of rank {rank}"))?;
        self.reseat(from + offset, rank)
    }

    fn reseat(&mut self, position: usize, rank: u64) -> VortexResult<()> {
        // The inverse of the encoder's `position = (element >> lower_width) + rank + 1`. Checked
        // for the same reason as in `rank_of_bucket`.
        let high = (position as u64).checked_sub(rank + 1).ok_or_else(|| {
            vortex_err!(
                "Elias-Fano upper array is malformed: the element of rank {rank} sits at bit \
                 {position}, at or below its own rank"
            )
        })?;
        let element = (high << self.lower_width) | self.lower_at(rank);
        self.seat = Some(Seat {
            position,
            rank,
            element,
        });
        Ok(())
    }

    fn seated(&self) -> &Seat {
        self.seat
            .as_ref()
            .vortex_expect("the cursor is seated before it is read")
    }

    #[inline]
    fn relative_rank(&self, rank: u64) -> usize {
        rank.checked_sub(self.first_rank)
            .and_then(|relative| usize::try_from(relative).ok())
            .vortex_expect("rank is within this array")
    }

    /// The low bits of the element at absolute rank `rank`.
    ///
    /// Masked, not trusted. Only a bit-packed child's width is checkable at construction — see
    /// `validate_parts` — so a patched or rewritten slot can arrive as a plain `u64` array carrying
    /// bits above `lower_width`. Those would bleed into the high part in [`Self::reseat`] and
    /// misorder the bisection in [`Self::search_bucket`], both silently.
    fn lower_at(&mut self, rank: u64) -> u64 {
        self.lower_at_unmasked(rank) & lower_mask(self.lower_width)
    }

    fn lower_at_unmasked(&mut self, rank: u64) -> u64 {
        // Copy the descriptor out before touching the scratch buffer: the packed slice borrows the
        // array, not `self`, so this keeps the borrow checker out of the way.
        let packed = match &self.lower {
            LowerBits::Zero => return 0,
            LowerBits::Dense { values, base } => return values[(rank - base) as usize],
            LowerBits::Packed(packed) => *packed,
        };

        let index = packed.index_of(rank);
        let chunk = index / FL_CHUNK_SIZE;
        let within_chunk = index % FL_CHUNK_SIZE;

        if self.scratch_chunk == Some(chunk) {
            return self.scratch_slice()[within_chunk];
        }

        if self.hot_chunk == Some(chunk) {
            self.hot_reads += 1;
        } else {
            self.hot_chunk = Some(chunk);
            self.hot_reads = 1;
        }

        let elems_per_chunk = 128 * packed.bit_width / size_of::<u64>();
        if self.hot_reads > BULK_UNPACK_THRESHOLD {
            let block = &packed.packed[chunk * elems_per_chunk..][..elems_per_chunk];
            let scratch = self
                .scratch
                .get_or_insert_with(|| Box::new([0u64; FL_CHUNK_SIZE]));
            // SAFETY: `block` is exactly `elems_per_chunk` packed values, and `scratch` is exactly
            // one FastLanes block of 1024 values, which is what `unchecked_unpack` requires.
            unsafe {
                BitPacking::unchecked_unpack(packed.bit_width, block, scratch.as_mut_slice())
            };
            self.scratch_chunk = Some(chunk);
            return self.scratch_slice()[within_chunk];
        }

        packed.unpack_one(index)
    }

    fn scratch_slice(&self) -> &[u64; FL_CHUNK_SIZE] {
        self.scratch
            .as_deref()
            .vortex_expect("the scratch block is allocated before it is read")
    }

    fn locate(&self, value: &Scalar) -> VortexResult<Bound> {
        if value.dtype() != self.dtype {
            vortex_bail!(
                "Elias-Fano probe dtype {} does not match array dtype {}",
                value.dtype(),
                self.dtype
            );
        }
        if value.is_null() {
            vortex_bail!("Elias-Fano cannot be probed with a null value");
        }
        let bits = scalar_bits(value);
        let element = bits.wrapping_sub(self.reference_bits);
        if element <= self.span {
            return Ok(Bound::Inside(element));
        }
        // Outside the universe. The subtraction above wraps for values below the reference and
        // overshoots for values above the max, so tell the two apart in the ptype's own ordering.
        let ptype = self.ptype;
        let reference_bits = self.reference_bits;
        let below = match_each_integer_ptype!(ptype, |P| { (bits as P) < (reference_bits as P) });
        Ok(if below { Bound::Below } else { Bound::Above })
    }
}

impl<'a> LowerBits<'a> {
    /// Choose how to read the low bits for absolute ranks `first_rank..first_rank + len`.
    fn try_new(
        lower: &'a ArrayRef,
        lower_width: u8,
        first_rank: u64,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<LowerBits<'a>> {
        if lower_width == 0 {
            return Ok(LowerBits::Zero);
        }
        if let Some(packed) = packed_in_place(lower) {
            return Ok(LowerBits::Packed(packed));
        }
        // Window before executing: the child spans the whole encoded sequence, and a cursor only
        // ever asks for ranks inside its own slice.
        let first = usize::try_from(first_rank)?;
        Ok(LowerBits::Dense {
            values: lower
                .slice(first..first + len)?
                .execute::<PrimitiveArray>(ctx)?
                .into_buffer::<u64>(),
            base: first_rank,
        })
    }
}
