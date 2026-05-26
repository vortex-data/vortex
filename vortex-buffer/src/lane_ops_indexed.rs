// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Indexed-source variant of `lane_ops`.
//!
//! Replaces `&[T]` with an `IndexedSource` trait: each lane read is
//! `unsafe fn get_unchecked(i) -> Item`, independent across iterations. For `&[T]`
//! this inlines to the same indexed load as the slice kernel; for `LaneZip(&[A], &[B])`
//! it gives two independent indexed reads per lane — both shapes the auto-vectorizer
//! handles.
//!
//! See `vortex-buffer/HISTORY.md` for the iterator-API investigation that motivated
//! this design.
//!
//! The output is always a caller-provided `&mut` slice — these kernels never allocate.
//! Both kernels handle a mask with a non-byte-aligned offset and with a logical `len`
//! shorter than the underlying byte buffer, via [`BitBuffer::chunks`].

use std::mem::MaybeUninit;

use crate::BitBuffer;

/// A length-known source supporting unchecked indexed reads.
///
/// Implemented for `&[T]` (with `T: Copy`) and for [`LaneZip`] over two `IndexedSource`s.
/// The kernels in this module require this trait instead of `Iterator` so that lane
/// reads carry no inter-iteration data dependency — the autovectorizer treats each
/// lane independently.
pub trait IndexedSource {
    /// The per-lane item type. Must be `Copy` so the kernels can pass it through
    /// the closure by value without extra moves.
    type Item: Copy;
    /// Logical lane count.
    fn len(&self) -> usize;
    /// Returns true when there are no lanes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Read the lane at `i` without bounds checking.
    ///
    /// # Safety
    ///
    /// `i` must be strictly less than `self.len()`.
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item;
}

impl<T: Copy> IndexedSource for &[T] {
    type Item = T;
    #[inline]
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> T {
        // SAFETY: caller guarantees i < self.len().
        unsafe { *<[T]>::get_unchecked(self, i) }
    }
}

/// Pair of two [`IndexedSource`]s of equal length. Yields `(A::Item, B::Item)` per lane.
///
/// Use this to drive a binary kernel from two columns. Length equality is enforced
/// at construction.
pub struct LaneZip<A, B>(pub A, pub B);

impl<A: IndexedSource, B: IndexedSource> LaneZip<A, B> {
    /// Build a `LaneZip` from two equal-length sources.
    ///
    /// # Panics
    ///
    /// Panics if the two operands have different lengths.
    pub fn new(a: A, b: B) -> Self {
        assert_eq!(
            a.len(),
            b.len(),
            "LaneZip operands must have the same length"
        );
        Self(a, b)
    }
}

impl<A: IndexedSource, B: IndexedSource> IndexedSource for LaneZip<A, B> {
    type Item = (A::Item, B::Item);
    #[inline]
    fn len(&self) -> usize {
        debug_assert_eq!(self.0.len(), self.1.len());
        self.0.len()
    }
    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> (A::Item, B::Item) {
        // SAFETY: caller guarantees i < self.len(); `new` enforces matching lengths.
        unsafe { (self.0.get_unchecked(i), self.1.get_unchecked(i)) }
    }
}

/// Apply `f(value, valid)` lane-by-lane, writing `out[i] = f(values[i], mask[i])`.
///
/// All three inputs must have the same length. The output type `R` may differ from the
/// input type `T` — this kernel is the building block for both same-type transforms
/// (fill_null) and cross-type ones (cast). The caller is responsible for marking `out`
/// initialized (e.g. by calling `BufferMut::set_len` after this returns).
///
/// # Panics
///
/// Panics if `values.len() != mask.len()` or `out.len() != values.len()`.
#[inline]
#[expect(clippy::many_single_char_names, reason = "tight numeric lane kernel")]
pub fn map_with_mask<S, R, F>(values: S, mask: &BitBuffer, out: &mut [MaybeUninit<R>], mut f: F)
where
    S: IndexedSource,
    F: FnMut(S::Item, bool) -> R,
{
    let len = values.len();
    assert_eq!(len, mask.len(), "values and mask must have the same length");
    assert_eq!(out.len(), len, "out must have the same length as values");

    let chunks = mask.chunks();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
        let base = chunk_idx * 64;
        // Inner loop is fixed-size 64 with independent per-lane reads — no iterator
        // state, no cross-iteration dependency, so the auto-vectorizer can fuse
        // 64 indexed loads into vector loads.
        for bit_idx in 0..64 {
            let i = base + bit_idx;
            let bit = (src_chunk >> bit_idx) & 1 == 1;
            // SAFETY: i < chunks_count * 64 <= len.
            let v = unsafe { values.get_unchecked(i) };
            unsafe { out.get_unchecked_mut(i).write(f(v, bit)) };
        }
    }

    if remainder != 0 {
        let src_chunk = chunks.remainder_bits();
        let base = chunks_count * 64;
        for bit_idx in 0..remainder {
            let i = base + bit_idx;
            let bit = (src_chunk >> bit_idx) & 1 == 1;
            // SAFETY: i < len.
            let v = unsafe { values.get_unchecked(i) };
            unsafe { out.get_unchecked_mut(i).write(f(v, bit)) };
        }
    }
}

/// Fallible variant of [`map_with_mask`]. `f` returns `Option<R>`; `None` indicates a
/// per-lane failure (e.g. range overflow on a narrowing cast).
///
/// The kernel does not short-circuit on the first failure inside a chunk: it processes
/// whole 64-lane chunks with `is_none()` flags OR-reduced into a single accumulator,
/// then checks after each chunk. On failure, a cold scalar attribution pass replays the
/// closure over that chunk to identify the first failing lane. The hot loop stays
/// autovectorizable — the per-lane cost is one OR on top of the cast.
///
/// On failure returns `Err(failing_lane_index)`. Lanes whose `f` returned `None` write
/// `R::default()` into `out`, but the contents of `out` must not be relied upon when
/// this function returns `Err`.
///
/// # Panics
///
/// Panics if `values.len() != mask.len()` or `out.len() != values.len()`.
#[inline]
#[expect(clippy::many_single_char_names, reason = "tight numeric lane kernel")]
pub fn try_map_with_mask<S, R, F>(
    values: S,
    mask: &BitBuffer,
    out: &mut [MaybeUninit<R>],
    mut f: F,
) -> Result<(), usize>
where
    S: IndexedSource,
    R: Copy + Default,
    F: FnMut(S::Item, bool) -> Option<R>,
{
    let len = values.len();
    assert_eq!(len, mask.len(), "values and mask must have the same length");
    assert_eq!(out.len(), len, "out must have the same length as values");

    let chunks = mask.chunks();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
        let base = chunk_idx * 64;
        // Per-chunk accumulator — does not escape the SIMD inner loop.
        let mut fail_acc: u64 = 0;
        for bit_idx in 0..64 {
            let i = base + bit_idx;
            let bit = (src_chunk >> bit_idx) & 1 == 1;
            // SAFETY: i < chunks_count * 64 <= len.
            let v = unsafe { values.get_unchecked(i) };
            let opt = f(v, bit);
            fail_acc |= opt.is_none() as u64;
            let r = opt.unwrap_or_default();
            // SAFETY: i < len.
            unsafe { out.get_unchecked_mut(i).write(r) };
        }
        if fail_acc != 0 {
            return Err(attribute_failure(&values, src_chunk, base, 64, &mut f));
        }
    }

    if remainder != 0 {
        let src_chunk = chunks.remainder_bits();
        let base = chunks_count * 64;
        let mut fail_acc: u64 = 0;
        for bit_idx in 0..remainder {
            let i = base + bit_idx;
            let bit = (src_chunk >> bit_idx) & 1 == 1;
            // SAFETY: i < len.
            let v = unsafe { values.get_unchecked(i) };
            let opt = f(v, bit);
            fail_acc |= opt.is_none() as u64;
            let r = opt.unwrap_or_default();
            // SAFETY: i < len.
            unsafe { out.get_unchecked_mut(i).write(r) };
        }
        if fail_acc != 0 {
            return Err(attribute_failure(
                &values, src_chunk, base, remainder, &mut f,
            ));
        }
    }

    Ok(())
}

/// Cold path: identify the first lane in a chunk where `f` returned `None`.
///
/// Called only after the hot loop has detected that at least one lane failed.
/// Walks the chunk scalar-style; not autovectorized, but that's fine — it only
/// runs once per error and the error path is supposed to be exceptional.
#[cold]
#[inline(never)]
#[expect(clippy::many_single_char_names, reason = "tight numeric lane kernel")]
fn attribute_failure<S, R, F>(
    values: &S,
    src_chunk: u64,
    base: usize,
    chunk_len: usize,
    f: &mut F,
) -> usize
where
    S: IndexedSource,
    F: FnMut(S::Item, bool) -> Option<R>,
{
    for bit_idx in 0..chunk_len {
        let i = base + bit_idx;
        let bit = (src_chunk >> bit_idx) & 1 == 1;
        // SAFETY: caller guarantees base + chunk_len <= values.len().
        let v = unsafe { values.get_unchecked(i) };
        if f(v, bit).is_none() {
            return i;
        }
    }
    // Unreachable: hot loop's OR-reduction said at least one lane in [base, base+chunk_len) failed.
    unreachable!("attribute_failure called without a failing lane")
}

/// Infallible, mask-free map: writes `out[i] = f(values[i])` for every lane.
///
/// The total (non-fallible) sibling of [`try_map`] and the building block for wrapping
/// arithmetic and other transforms that cannot fail. The body has no branch and no
/// validity read, so it autovectorizes to the tightest possible loop.
///
/// # Panics
///
/// Panics if `out.len() != values.len()`.
#[inline]
pub fn map<S, R, F>(values: S, out: &mut [MaybeUninit<R>], mut f: F)
where
    S: IndexedSource,
    F: FnMut(S::Item) -> R,
{
    let len = values.len();
    assert_eq!(out.len(), len, "out must have the same length as values");
    for i in 0..len {
        // SAFETY: i < len == values.len() == out.len().
        let v = unsafe { values.get_unchecked(i) };
        unsafe { out.get_unchecked_mut(i).write(f(v)) };
    }
}

/// Fallible map with **no validity argument** — the throughput-oriented sibling of
/// [`try_map_with_mask`].
///
/// `f` sees only the value, never a `valid` flag, so the hot loop never touches a
/// validity bitmap and the per-lane body reduces to exactly the work arrow's checked
/// arithmetic loop does: read, compute, overflow-check, store. Callers that need
/// null-awareness should combine validity in a **separate** bitmap pass (e.g. a
/// word-parallel `BitBuffer & BitBuffer`) rather than gating each lane here — null
/// lanes still get `f` applied to their (semantically irrelevant) values, which keeps
/// this loop branch-free and autovectorizable.
///
/// Failure handling matches [`try_map_with_mask`]: per-chunk `is_none()` flags are
/// OR-reduced, and on failure a cold scalar pass attributes the first failing lane.
///
/// On failure returns `Err(failing_lane_index)`; the contents of `out` must not be
/// relied upon when this returns `Err`.
///
/// # Panics
///
/// Panics if `out.len() != values.len()`.
#[inline]
#[expect(clippy::many_single_char_names, reason = "tight numeric lane kernel")]
pub fn try_map<S, R, F>(values: S, out: &mut [MaybeUninit<R>], mut f: F) -> Result<(), usize>
where
    S: IndexedSource,
    R: Copy + Default,
    F: FnMut(S::Item) -> Option<R>,
{
    let len = values.len();
    assert_eq!(out.len(), len, "out must have the same length as values");

    let chunks_count = len / 64;
    let remainder = len % 64;

    for chunk_idx in 0..chunks_count {
        let base = chunk_idx * 64;
        let mut fail_acc: u64 = 0;
        for bit_idx in 0..64 {
            let i = base + bit_idx;
            // SAFETY: i < chunks_count * 64 <= len.
            let v = unsafe { values.get_unchecked(i) };
            let opt = f(v);
            fail_acc |= opt.is_none() as u64;
            // SAFETY: i < len.
            unsafe { out.get_unchecked_mut(i).write(opt.unwrap_or_default()) };
        }
        if fail_acc != 0 {
            return Err(attribute_failure_unmasked(&values, base, 64, &mut f));
        }
    }

    if remainder != 0 {
        let base = chunks_count * 64;
        let mut fail_acc: u64 = 0;
        for bit_idx in 0..remainder {
            let i = base + bit_idx;
            // SAFETY: i < len.
            let v = unsafe { values.get_unchecked(i) };
            let opt = f(v);
            fail_acc |= opt.is_none() as u64;
            // SAFETY: i < len.
            unsafe { out.get_unchecked_mut(i).write(opt.unwrap_or_default()) };
        }
        if fail_acc != 0 {
            return Err(attribute_failure_unmasked(&values, base, remainder, &mut f));
        }
    }

    Ok(())
}

/// Cold attribution path for [`try_map`]; mirror of [`attribute_failure`] without a mask.
#[cold]
#[inline(never)]
#[expect(clippy::many_single_char_names, reason = "tight numeric lane kernel")]
fn attribute_failure_unmasked<S, R, F>(
    values: &S,
    base: usize,
    chunk_len: usize,
    f: &mut F,
) -> usize
where
    S: IndexedSource,
    F: FnMut(S::Item) -> Option<R>,
{
    for bit_idx in 0..chunk_len {
        let i = base + bit_idx;
        // SAFETY: caller guarantees base + chunk_len <= values.len().
        let v = unsafe { values.get_unchecked(i) };
        if f(v).is_none() {
            return i;
        }
    }
    unreachable!("attribute_failure_unmasked called without a failing lane")
}

/// Null-lenient fallible map matching `arrow_arith::numeric` checked-arithmetic semantics.
///
/// `f` is applied to **every** lane, so the arithmetic hot loop never reads `validity` and
/// stays autovectorizable. The hot loop only OR-reduces a single overflow flag (no positioned
/// bitmap, no loop-carried bit-insert), so it vectorizes exactly like [`try_map`]. Only when
/// an overflow is actually observed does a cold pass re-walk the data, gate the overflow by
/// `validity`, and decide whether any *valid* lane failed. This mirrors arrow, where null
/// lanes are skipped, so an overflow whose value lives at a null lane is ignored.
///
/// `out[i]` receives `f(values[i]).unwrap_or_default()`. The value written at null lanes is
/// irrelevant (it is masked by `validity`); on `Err` the contents of `out` must not be used.
///
/// Returns `Err(lane)` for the lowest **valid** lane where `f` returned `None`. An overflow at
/// a null lane is not an error.
///
/// # Panics
///
/// Panics if `values.len() != validity.len()` or `out.len() != values.len()`.
#[inline]
#[expect(clippy::many_single_char_names, reason = "tight numeric lane kernel")]
pub fn try_map_nullable<S, R, F>(
    values: S,
    validity: &BitBuffer,
    out: &mut [MaybeUninit<R>],
    mut f: F,
) -> Result<(), usize>
where
    S: IndexedSource,
    R: Copy + Default,
    F: FnMut(S::Item) -> Option<R>,
{
    let len = values.len();
    assert_eq!(
        len,
        validity.len(),
        "values and validity must have the same length"
    );
    assert_eq!(out.len(), len, "out must have the same length as values");

    let chunks_count = len / 64;
    let remainder = len % 64;

    // Hot path: write every lane and OR-reduce a plain overflow flag. No validity, no
    // positioned bitmap — identical shape to `try_map`, so this autovectorizes.
    let mut any_overflow: u64 = 0;
    for chunk_idx in 0..chunks_count {
        let base = chunk_idx * 64;
        for bit_idx in 0..64 {
            let i = base + bit_idx;
            // SAFETY: i < chunks_count * 64 <= len.
            let v = unsafe { values.get_unchecked(i) };
            let opt = f(v);
            any_overflow |= opt.is_none() as u64;
            // SAFETY: i < len.
            unsafe { out.get_unchecked_mut(i).write(opt.unwrap_or_default()) };
        }
    }
    if remainder != 0 {
        let base = chunks_count * 64;
        for bit_idx in 0..remainder {
            let i = base + bit_idx;
            // SAFETY: i < len.
            let v = unsafe { values.get_unchecked(i) };
            let opt = f(v);
            any_overflow |= opt.is_none() as u64;
            // SAFETY: i < len.
            unsafe { out.get_unchecked_mut(i).write(opt.unwrap_or_default()) };
        }
    }

    if any_overflow == 0 {
        return Ok(());
    }
    // Cold: at least one lane overflowed. Re-walk and find the lowest *valid* failing lane;
    // if every overflow sits at a null lane, it is not an error (arrow semantics).
    attribute_nullable_failure(&values, validity, &mut f)
}

/// Cold path for [`try_map_nullable`]: with at least one overflow somewhere, gate the per-lane
/// overflow flags by `validity` word-by-word and return the lowest valid failing lane, or `Ok`
/// if all overflows landed on null lanes.
#[cold]
#[inline(never)]
#[expect(clippy::many_single_char_names, reason = "tight numeric lane kernel")]
fn attribute_nullable_failure<S, R, F>(
    values: &S,
    validity: &BitBuffer,
    f: &mut F,
) -> Result<(), usize>
where
    S: IndexedSource,
    F: FnMut(S::Item) -> Option<R>,
{
    let len = values.len();
    let chunks = validity.chunks();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for (chunk_idx, valid_word) in chunks.iter().enumerate() {
        let base = chunk_idx * 64;
        let mut overflow_word: u64 = 0;
        for bit_idx in 0..64 {
            let i = base + bit_idx;
            // SAFETY: i < chunks_count * 64 <= len.
            let v = unsafe { values.get_unchecked(i) };
            overflow_word |= (f(v).is_none() as u64) << bit_idx;
        }
        let bad = overflow_word & valid_word;
        if bad != 0 {
            return Err(base + bad.trailing_zeros() as usize);
        }
    }

    if remainder != 0 {
        let valid_word = chunks.remainder_bits();
        let base = chunks_count * 64;
        let mut overflow_word: u64 = 0;
        for bit_idx in 0..remainder {
            let i = base + bit_idx;
            // SAFETY: i < len.
            let v = unsafe { values.get_unchecked(i) };
            overflow_word |= (f(v).is_none() as u64) << bit_idx;
        }
        let bad = overflow_word & valid_word;
        if bad != 0 {
            return Err(base + bad.trailing_zeros() as usize);
        }
    }

    Ok(())
}

/// Apply `f(value) -> bool` lane-by-lane, packing into `out` as `u64` words.
///
/// This is the validity-free sibling of [`map_with_mask_to_bits`]. Use it when the
/// predicate is a pure function of the value (e.g. compare-to-constant on a primitive
/// buffer) and combine the validity bitmap in a separate pass — splitting the work
/// this way lets the value-compare loop autovectorize cleanly.
///
/// `out.len()` must equal `values.len().div_ceil(64)`. Trailing bits in the final word
/// beyond `len % 64` are written as `0`.
///
/// # Panics
///
/// Panics if `out.len() != values.len().div_ceil(64)`.
#[inline]
pub fn map_to_bits<S, F>(values: S, out: &mut [u64], mut f: F)
where
    S: IndexedSource,
    F: FnMut(S::Item) -> bool,
{
    let len = values.len();
    assert_eq!(
        out.len(),
        len.div_ceil(64),
        "out must have len.div_ceil(64) words",
    );

    let chunks_count = len / 64;
    let remainder = len % 64;

    for chunk_idx in 0..chunks_count {
        let base = chunk_idx * 64;
        let mut packed = 0u64;
        for bit_idx in 0..64 {
            // SAFETY: base + bit_idx < chunks_count * 64 <= len.
            let v = unsafe { values.get_unchecked(base + bit_idx) };
            packed |= (f(v) as u64) << bit_idx;
        }
        // SAFETY: chunk_idx < chunks_count <= out.len().
        unsafe { *out.get_unchecked_mut(chunk_idx) = packed };
    }

    if remainder != 0 {
        let base = chunks_count * 64;
        let mut packed = 0u64;
        for bit_idx in 0..remainder {
            // SAFETY: base + bit_idx < len.
            let v = unsafe { values.get_unchecked(base + bit_idx) };
            packed |= (f(v) as u64) << bit_idx;
        }
        // SAFETY: chunks_count < out.len() because remainder != 0.
        unsafe { *out.get_unchecked_mut(chunks_count) = packed };
    }
}

/// Apply `f(value, valid) -> bool` lane-by-lane, packing into `out` as `u64` words.
///
/// `out.len()` must equal `values.len().div_ceil(64)`. Trailing bits in the final word
/// beyond `len % 64` are written as `0`.
///
/// # Panics
///
/// Panics if `values.len() != mask.len()` or `out.len() != values.len().div_ceil(64)`.
#[inline]
#[expect(clippy::many_single_char_names, reason = "tight numeric lane kernel")]
pub fn map_with_mask_to_bits<S, F>(values: S, mask: &BitBuffer, out: &mut [u64], mut f: F)
where
    S: IndexedSource,
    F: FnMut(S::Item, bool) -> bool,
{
    let len = values.len();
    assert_eq!(len, mask.len(), "values and mask must have the same length");
    assert_eq!(
        out.len(),
        len.div_ceil(64),
        "out must have len.div_ceil(64) words",
    );

    let chunks = mask.chunks();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
        let base = chunk_idx * 64;
        let mut packed = 0u64;
        for bit_idx in 0..64 {
            let i = base + bit_idx;
            let bit = (src_chunk >> bit_idx) & 1 == 1;
            // SAFETY: i < chunks_count * 64 <= len.
            let v = unsafe { values.get_unchecked(i) };
            packed |= (f(v, bit) as u64) << bit_idx;
        }
        // SAFETY: chunk_idx < chunks_count <= out.len().
        unsafe { *out.get_unchecked_mut(chunk_idx) = packed };
    }

    if remainder != 0 {
        let src_chunk = chunks.remainder_bits();
        let base = chunks_count * 64;
        let mut packed = 0u64;
        for bit_idx in 0..remainder {
            let i = base + bit_idx;
            let bit = (src_chunk >> bit_idx) & 1 == 1;
            // SAFETY: i < len.
            let v = unsafe { values.get_unchecked(i) };
            packed |= (f(v, bit) as u64) << bit_idx;
        }
        // SAFETY: chunks_count < out.len() because remainder != 0.
        unsafe { *out.get_unchecked_mut(chunks_count) = packed };
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    reason = "tests use unwrap, short lane names, and narrowing casts for brevity"
)]
mod tests {
    use super::*;
    use crate::BitBufferMut;

    fn write_t<T: Copy>(out: Vec<MaybeUninit<T>>) -> Vec<T> {
        // SAFETY: tests always fully initialize the buffer.
        unsafe { std::mem::transmute::<Vec<MaybeUninit<T>>, Vec<T>>(out) }
    }

    #[test]
    fn map_with_mask_aligned() {
        let values: Vec<i32> = (0..10).collect();
        let mask = {
            let mut m = BitBufferMut::with_capacity(10);
            for i in 0..10 {
                m.append(i % 2 == 0);
            }
            m.freeze()
        };
        let mut out = vec![MaybeUninit::<i32>::uninit(); 10];
        map_with_mask(values.as_slice(), &mask, &mut out, |v, valid| {
            if valid { v } else { -1 }
        });
        assert_eq!(write_t(out), vec![0, -1, 2, -1, 4, -1, 6, -1, 8, -1]);
    }

    #[test]
    fn map_with_mask_partial_chunk() {
        // 130 lanes — two full u64 words + a 2-bit remainder.
        let values: Vec<i32> = (0..130).collect();
        let mask = BitBuffer::new_set(130);
        let mut out = vec![MaybeUninit::<i32>::uninit(); 130];
        map_with_mask(values.as_slice(), &mask, &mut out, |v, valid| {
            if valid { v + 1 } else { 0 }
        });
        let got = write_t(out);
        assert_eq!(got.len(), 130);
        assert_eq!(got[0], 1);
        assert_eq!(got[63], 64);
        assert_eq!(got[64], 65);
        assert_eq!(got[129], 130);
    }

    #[test]
    fn map_with_mask_offset_mask() {
        // Build a 128-bit all-true mask, then slice off the first 5 bits to force offset=5.
        let big = BitBuffer::new_set(128);
        let sliced = big.slice(5..70); // logical len = 65, offset = 5
        assert_eq!(sliced.len(), 65);
        assert_eq!(sliced.offset(), 5);

        let values: Vec<u32> = (0..65).collect();
        let mut out = vec![MaybeUninit::<u32>::uninit(); 65];
        map_with_mask(values.as_slice(), &sliced, &mut out, |v, valid| {
            if valid { v } else { u32::MAX }
        });
        let got = write_t(out);
        assert_eq!(got, (0..65).collect::<Vec<u32>>());
    }

    #[test]
    fn map_with_mask_offset_past_word() {
        // Slicing past a full word still works. `BitBuffer::slice` normalizes the
        // logical offset to `offset % 8` and bumps the underlying byte pointer,
        // so `offset()` won't equal 70 here — what we exercise is that the kernel
        // walks the chunked u64 view (which BitChunks handles internally).
        let big = BitBuffer::new_set(256);
        let sliced = big.slice(70..200);
        assert_eq!(sliced.len(), 130);

        let values: Vec<i16> = (0..130).map(|i| i as i16).collect();
        let mut out = vec![MaybeUninit::<i16>::uninit(); 130];
        map_with_mask(values.as_slice(), &sliced, &mut out, |v, valid| {
            if valid { v } else { -1 }
        });
        let got = write_t(out);
        assert_eq!(got, (0..130).map(|i| i as i16).collect::<Vec<_>>());
    }

    #[test]
    fn map_with_mask_empty() {
        let values: Vec<i32> = vec![];
        let mask = BitBuffer::new_unset(0);
        let mut out: Vec<MaybeUninit<i32>> = vec![];
        map_with_mask(values.as_slice(), &mask, &mut out, |v, _| v);
    }

    #[test]
    fn map_with_mask_null_to_zero_branchless() {
        // The trick from primitive/compute/cast.rs:147 — multiply by valid as T.
        let values: Vec<i64> = (1..=100).collect();
        let mask = {
            let mut m = BitBufferMut::with_capacity(100);
            for i in 0..100 {
                m.append(i % 3 != 0);
            }
            m.freeze()
        };
        let mut out = vec![MaybeUninit::<i64>::uninit(); 100];
        map_with_mask(values.as_slice(), &mask, &mut out, |v, valid| {
            v * (valid as i64)
        });
        let got = write_t(out);
        for (i, &x) in got.iter().enumerate() {
            if i % 3 == 0 {
                assert_eq!(x, 0);
            } else {
                assert_eq!(x, (i + 1) as i64);
            }
        }
    }

    #[test]
    fn map_with_mask_to_bits_aligned() {
        let values: Vec<i32> = (0..128).collect();
        let mask = BitBuffer::new_set(128);
        let mut out = vec![0u64; 2];
        map_with_mask_to_bits(values.as_slice(), &mask, &mut out, |v, valid| {
            valid && v % 2 == 0
        });
        // Even numbers in [0, 128) set, odd unset.
        for word_idx in 0..2 {
            let word = out[word_idx];
            for bit in 0..64 {
                let i = word_idx * 64 + bit;
                let expected = i % 2 == 0;
                assert_eq!((word >> bit) & 1 == 1, expected, "lane {i}");
            }
        }
    }

    #[test]
    fn map_with_mask_to_bits_partial_chunk() {
        // 130 lanes — three u64 words, last word has only 2 valid bits.
        let values: Vec<i32> = (0..130).collect();
        let mask = BitBuffer::new_set(130);
        let mut out = vec![0u64; 130usize.div_ceil(64)];
        assert_eq!(out.len(), 3);
        map_with_mask_to_bits(values.as_slice(), &mask, &mut out, |v, valid| {
            valid && v >= 64
        });
        // Bits 64..128 set in word 1; bits 128..130 set in word 2.
        assert_eq!(out[0], 0);
        assert_eq!(out[1], u64::MAX);
        assert_eq!(out[2], 0b11);
    }

    #[test]
    fn map_with_mask_to_bits_offset() {
        let big = BitBuffer::new_set(256);
        let sliced = big.slice(13..143); // offset=13, len=130
        assert_eq!(sliced.len(), 130);
        let values: Vec<u8> = (0..130).map(|i| (i % 4) as u8).collect();
        let mut out = vec![0u64; 130usize.div_ceil(64)];
        map_with_mask_to_bits(values.as_slice(), &sliced, &mut out, |v, valid| {
            valid && v == 0
        });
        for i in 0..130 {
            let word = out[i / 64];
            let bit = (word >> (i % 64)) & 1 == 1;
            assert_eq!(bit, i % 4 == 0, "lane {i}");
        }
    }

    #[test]
    fn try_map_with_mask_all_ok() {
        let values: Vec<u64> = (0..200).collect();
        let mask = BitBuffer::new_set(200);
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        let res = try_map_with_mask(values.as_slice(), &mask, &mut out, |v, valid| {
            let scaled = v * valid as u64;
            (scaled <= u32::MAX as u64).then_some(scaled as u32)
        });
        assert!(res.is_ok());
        let got = write_t(out);
        assert_eq!(got, (0..200u32).collect::<Vec<_>>());
    }

    #[test]
    fn try_map_with_mask_overflow_fails() {
        // Put an overflowing value at lane 137 — the kernel must report Err(137).
        let mut values: Vec<u64> = (0..200).collect();
        values[137] = (u32::MAX as u64) + 1;
        let mask = BitBuffer::new_set(200);
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        let res = try_map_with_mask(values.as_slice(), &mask, &mut out, |v, valid| {
            let scaled = v * valid as u64;
            (scaled <= u32::MAX as u64).then_some(scaled as u32)
        });
        assert_eq!(res, Err(137));
    }

    #[test]
    fn try_map_with_mask_overflow_reports_first_failing_lane() {
        // Multiple failing lanes — must report the lowest index.
        let mut values: Vec<u64> = (0..200).collect();
        values[50] = u64::MAX;
        values[51] = u64::MAX;
        values[137] = u64::MAX;
        let mask = BitBuffer::new_set(200);
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        let res = try_map_with_mask(values.as_slice(), &mask, &mut out, |v, valid| {
            let scaled = v * valid as u64;
            (scaled <= u32::MAX as u64).then_some(scaled as u32)
        });
        assert_eq!(res, Err(50));
    }

    #[test]
    fn try_map_with_mask_null_lane_bypasses_check() {
        // Null lanes are neutralized by `valid as u64` before the range check, so an
        // out-of-range value at a null lane must NOT trigger failure.
        let mut values: Vec<u64> = (0..200).collect();
        values[5] = u64::MAX;
        let mask = {
            let mut m = BitBufferMut::with_capacity(200);
            for i in 0..200 {
                m.append(i != 5);
            }
            m.freeze()
        };
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        let res = try_map_with_mask(values.as_slice(), &mask, &mut out, |v, valid| {
            let scaled = v * valid as u64;
            (scaled <= u32::MAX as u64).then_some(scaled as u32)
        });
        assert!(res.is_ok());
        let got = write_t(out);
        assert_eq!(got[5], 0); // null-lane wrote default
        assert_eq!(got[6], 6);
    }

    #[test]
    fn try_map_with_mask_branchful_matches_branchless() {
        let mut values: Vec<u64> = (0..130).map(|i| i as u64 * 7).collect();
        values[2] = u64::MAX;
        values[65] = u32::MAX as u64;
        let mask = {
            let mut m = BitBufferMut::with_capacity(130);
            for i in 0..130 {
                m.append(!matches!(i, 2 | 17 | 99));
            }
            m.freeze()
        };

        let mut branchless = vec![MaybeUninit::<u32>::uninit(); 130];
        let mut branchful = vec![MaybeUninit::<u32>::uninit(); 130];
        try_map_with_mask(values.as_slice(), &mask, &mut branchless, |v, valid| {
            let scaled = v * valid as u64;
            (scaled <= u32::MAX as u64).then_some(scaled as u32)
        })
        .unwrap();
        try_map_with_mask(values.as_slice(), &mask, &mut branchful, |v, valid| {
            if valid {
                u32::try_from(v).ok()
            } else {
                Some(0)
            }
        })
        .unwrap();

        assert_eq!(write_t(branchful), write_t(branchless));
    }

    #[test]
    fn try_map_with_mask_partial_chunk() {
        let values: Vec<u64> = (0..130).collect();
        let mask = BitBuffer::new_set(130);
        let mut out = vec![MaybeUninit::<u32>::uninit(); 130];
        let res = try_map_with_mask(values.as_slice(), &mask, &mut out, |v, valid| {
            let scaled = v * valid as u64;
            (scaled <= u32::MAX as u64).then_some(scaled as u32)
        });
        assert!(res.is_ok());
        let got = write_t(out);
        assert_eq!(got.len(), 130);
        assert_eq!(got[129], 129);
    }

    #[test]
    fn try_map_with_mask_sliced_mask_unaligned_offset() {
        // The mask's first byte is not word-aligned: slice off 13 bits, so the
        // underlying BitChunks iterator must shift across byte boundaries on every
        // 64-bit chunk it yields.
        let big = BitBuffer::new_set(256);
        let mask = big.slice(13..143); // logical len = 130, bit offset = 13 % 8 = 5
        assert_eq!(mask.len(), 130);

        let values: Vec<u64> = (0..130).collect();
        let mut out = vec![MaybeUninit::<u32>::uninit(); 130];
        let res = try_map_with_mask(values.as_slice(), &mask, &mut out, |v, valid| {
            let scaled = v * valid as u64;
            (scaled <= u32::MAX as u64).then_some(scaled as u32)
        });
        assert!(res.is_ok());
        let got = write_t(out);
        assert_eq!(got, (0..130u32).collect::<Vec<_>>());
    }

    #[test]
    fn try_map_with_mask_sliced_mask_with_overflow() {
        // Sliced mask + overflowing value — the cold attribution path must report
        // the correct lane index in the sliced (post-offset) coordinate space.
        let big = BitBuffer::new_set(256);
        let mask = big.slice(13..143);
        assert_eq!(mask.len(), 130);

        let mut values: Vec<u64> = (0..130).collect();
        values[77] = u64::MAX;
        let mut out = vec![MaybeUninit::<u32>::uninit(); 130];
        let res = try_map_with_mask(values.as_slice(), &mask, &mut out, |v, valid| {
            let scaled = v * valid as u64;
            (scaled <= u32::MAX as u64).then_some(scaled as u32)
        });
        assert_eq!(res, Err(77));
    }

    #[test]
    fn try_map_with_mask_sliced_mask_null_lanes() {
        // Mix sliced offset with a non-trivial validity pattern. Null lanes must
        // not contribute to fail_acc, even when their underlying value would overflow.
        let mut m = BitBufferMut::with_capacity(256);
        for i in 0..256 {
            m.append(i % 3 != 0);
        }
        let big = m.freeze();
        let mask = big.slice(13..143);
        assert_eq!(mask.len(), 130);

        // After the 13-lane slice, original index `13 + j` becomes lane `j`.
        // Lane `j` is valid iff `(13 + j) % 3 != 0`.
        let mut values: Vec<u64> = (0..130).collect();
        // Pick a lane that is INVALID in the sliced coords: 13+2 = 15, 15 % 3 == 0 → invalid.
        // Stuff in an overflowing value; it must be neutralized by `* valid as u64`.
        values[2] = u64::MAX;
        let mut out = vec![MaybeUninit::<u32>::uninit(); 130];
        let res = try_map_with_mask(values.as_slice(), &mask, &mut out, |v, valid| {
            let scaled = v * valid as u64;
            (scaled <= u32::MAX as u64).then_some(scaled as u32)
        });
        assert!(res.is_ok(), "null lane should bypass the range check");
    }

    #[test]
    fn try_map_with_mask_overflow_in_remainder() {
        // Overflow in the trailing partial chunk (not aligned to 64).
        let mut values: Vec<u64> = (0..130).collect();
        values[129] = (u32::MAX as u64) + 1;
        let mask = BitBuffer::new_set(130);
        let mut out = vec![MaybeUninit::<u32>::uninit(); 130];
        let res = try_map_with_mask(values.as_slice(), &mask, &mut out, |v, valid| {
            let scaled = v * valid as u64;
            (scaled <= u32::MAX as u64).then_some(scaled as u32)
        });
        assert_eq!(res, Err(129));
    }

    #[test]
    fn map_writes_every_lane() {
        let values: Vec<u32> = (0..200).collect();
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        map(values.as_slice(), &mut out, |v| v.wrapping_mul(3));
        let got = write_t(out);
        assert_eq!(
            got,
            (0..200u32).map(|v| v.wrapping_mul(3)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn map_wrapping_add_matches_scalar() {
        let lhs: Vec<u32> = vec![u32::MAX, 5, 100, 0];
        let rhs: Vec<u32> = vec![2, 7, 1, 0];
        let mut out = vec![MaybeUninit::<u32>::uninit(); 4];
        map(
            LaneZip::new(lhs.as_slice(), rhs.as_slice()),
            &mut out,
            |(a, b)| a.wrapping_add(b),
        );
        let got = write_t(out);
        assert_eq!(got, vec![1, 12, 101, 0]); // MAX + 2 wraps to 1
    }

    #[test]
    fn map_empty() {
        let values: Vec<u32> = vec![];
        let mut out: Vec<MaybeUninit<u32>> = vec![];
        map(values.as_slice(), &mut out, |v| v + 1);
    }

    #[test]
    fn try_map_nullable_all_ok() {
        let values: Vec<u64> = (0..200).collect();
        let validity = BitBuffer::new_set(200);
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        let res = try_map_nullable(values.as_slice(), &validity, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert!(res.is_ok());
        let got = write_t(out);
        assert_eq!(got, (0..200u32).collect::<Vec<_>>());
    }

    #[test]
    fn try_map_nullable_ignores_overflow_at_null_lane() {
        // This is the arrow-parity property: an overflow whose value lives at a NULL
        // lane must NOT fault — exactly like arrow's `try_binary` skipping null lanes.
        let mut values: Vec<u64> = (0..130).collect();
        values[77] = (u32::MAX as u64) + 1; // would overflow u32
        let validity = {
            let mut m = BitBufferMut::with_capacity(130);
            for i in 0..130 {
                m.append(i != 77); // lane 77 is null
            }
            m.freeze()
        };
        let mut out = vec![MaybeUninit::<u32>::uninit(); 130];
        let res = try_map_nullable(values.as_slice(), &validity, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert!(
            res.is_ok(),
            "overflow at a null lane must be ignored (arrow semantics)"
        );
    }

    #[test]
    fn try_map_nullable_faults_at_valid_lane() {
        // Overflow at a VALID lane must fault, reporting that lane.
        let mut values: Vec<u64> = (0..130).collect();
        values[77] = (u32::MAX as u64) + 1;
        let validity = BitBuffer::new_set(130); // all valid
        let mut out = vec![MaybeUninit::<u32>::uninit(); 130];
        let res = try_map_nullable(values.as_slice(), &validity, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert_eq!(res, Err(77));
    }

    #[test]
    fn try_map_nullable_reports_lowest_valid_failing_lane() {
        // Mix of null and valid overflows; only valid ones count, lowest index wins.
        let mut values: Vec<u64> = (0..200).collect();
        values[40] = u64::MAX; // null -> ignored
        values[90] = u64::MAX; // valid -> should be reported
        values[150] = u64::MAX; // valid
        let validity = {
            let mut m = BitBufferMut::with_capacity(200);
            for i in 0..200 {
                m.append(i != 40); // only lane 40 is null
            }
            m.freeze()
        };
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        let res = try_map_nullable(values.as_slice(), &validity, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert_eq!(res, Err(90));
    }

    #[test]
    fn try_map_nullable_overflow_in_remainder() {
        // Valid-lane overflow in the trailing partial chunk.
        let mut values: Vec<u64> = (0..130).collect();
        values[129] = (u32::MAX as u64) + 1;
        let validity = BitBuffer::new_set(130);
        let mut out = vec![MaybeUninit::<u32>::uninit(); 130];
        let res = try_map_nullable(values.as_slice(), &validity, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert_eq!(res, Err(129));
    }

    #[test]
    fn try_map_nullable_sliced_validity_offset() {
        // Unaligned validity offset must still AND correctly per chunk.
        let big = BitBuffer::new_set(256);
        let validity = big.slice(13..143); // offset 5, len 130
        assert_eq!(validity.len(), 130);
        let values: Vec<u64> = (0..130).collect();
        let mut out = vec![MaybeUninit::<u32>::uninit(); 130];
        let res = try_map_nullable(values.as_slice(), &validity, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert!(res.is_ok());
        let got = write_t(out);
        assert_eq!(got, (0..130u32).collect::<Vec<_>>());
    }

    #[test]
    fn try_map_nullable_matches_arrow_semantics_property() {
        // Cross-check the failure condition against a reference: fault iff some VALID
        // lane overflows. Build a pattern with overflows scattered across null/valid.
        let n = 300usize;
        let validity = {
            let mut m = BitBufferMut::with_capacity(n);
            for i in 0..n {
                m.append(i % 5 != 0); // 1-in-5 null
            }
            m.freeze()
        };
        let mut values: Vec<u64> = (0..n as u64).collect();
        // Overflow only at null lanes (multiples of 5) -> must be Ok.
        for i in (0..n).step_by(5) {
            values[i] = u64::MAX;
        }
        let mut out = vec![MaybeUninit::<u32>::uninit(); n];
        let res = try_map_nullable(values.as_slice(), &validity, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert!(res.is_ok(), "all overflows are at null lanes");

        // Now flip one valid lane to overflow -> must fault at that lane.
        let mut values2 = values.clone();
        values2[123] = u64::MAX; // 123 % 5 == 3 -> valid
        let mut out2 = vec![MaybeUninit::<u32>::uninit(); n];
        let res2 = try_map_nullable(values2.as_slice(), &validity, &mut out2, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert_eq!(res2, Err(123));
    }

    #[test]
    fn map_to_bits_aligned() {
        let values: Vec<i32> = (0..128).collect();
        let mut out = vec![0u64; 2];
        map_to_bits(values.as_slice(), &mut out, |v| v % 2 == 0);
        for word_idx in 0..2 {
            for bit in 0..64 {
                let i = word_idx * 64 + bit;
                let expected = i % 2 == 0;
                assert_eq!((out[word_idx] >> bit) & 1 == 1, expected, "lane {i}");
            }
        }
    }

    #[test]
    fn map_to_bits_partial_chunk() {
        let values: Vec<i32> = (0..130).collect();
        let mut out = vec![0u64; 130usize.div_ceil(64)];
        assert_eq!(out.len(), 3);
        map_to_bits(values.as_slice(), &mut out, |v| v >= 64);
        assert_eq!(out[0], 0);
        assert_eq!(out[1], u64::MAX);
        assert_eq!(out[2], 0b11);
    }

    #[test]
    fn map_to_bits_empty() {
        let values: Vec<i32> = vec![];
        let mut out: Vec<u64> = vec![];
        map_to_bits(values.as_slice(), &mut out, |v| v > 0);
    }

    #[test]
    fn map_to_bits_matches_fused_with_all_valid_mask() {
        // map_to_bits + AND with an all-true mask must equal map_with_mask_to_bits.
        let values: Vec<i64> = (0..200).map(|i| i % 7).collect();
        let mask = BitBuffer::new_set(200);

        let mut a = vec![0u64; 200usize.div_ceil(64)];
        map_with_mask_to_bits(values.as_slice(), &mask, &mut a, |v, valid| valid && v == 3);

        let mut b = vec![0u64; 200usize.div_ceil(64)];
        map_to_bits(values.as_slice(), &mut b, |v| v == 3);

        assert_eq!(a, b);
    }

    #[test]
    fn map_with_mask_to_bits_validity_kills_lane() {
        // Even if predicate is true, null lanes should produce false.
        let values: Vec<i32> = vec![1; 70];
        let mask = {
            let mut m = BitBufferMut::with_capacity(70);
            for i in 0..70 {
                m.append(i >= 32); // first 32 lanes are null
            }
            m.freeze()
        };
        let mut out = vec![0u64; 70usize.div_ceil(64)];
        map_with_mask_to_bits(values.as_slice(), &mask, &mut out, |v, valid| {
            valid && v == 1
        });
        for i in 0..70 {
            let bit = (out[i / 64] >> (i % 64)) & 1 == 1;
            assert_eq!(bit, i >= 32, "lane {i}");
        }
    }
}
