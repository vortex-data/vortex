// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Elementwise lane kernels over indexed sources.
//!
//! Replaces `&[T]` with an [`IndexedSource`] trait: each lane read is
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

#![allow(clippy::many_single_char_names)]

use std::mem::MaybeUninit;

use crate::BitBuffer;

macro_rules! for_full_lanes {
    ($base:expr, | $bit_idx:ident, $i:ident | $body:block) => {
        for $bit_idx in 0..64 {
            let $i = $base + $bit_idx;
            $body
        }
    };
}

macro_rules! for_remainder_lanes {
    ($base:expr, $remainder:expr, | $bit_idx:ident, $i:ident | $body:block) => {
        for $bit_idx in 0..$remainder {
            let $i = $base + $bit_idx;
            $body
        }
    };
}

macro_rules! for_full_mask_lanes {
    ($src_chunk:expr, $base:expr, | $bit_idx:ident, $i:ident, $valid:ident | $body:block) => {
        for $bit_idx in 0..64 {
            let $i = $base + $bit_idx;
            let $valid = ($src_chunk >> $bit_idx) & 1 == 1;
            $body
        }
    };
}

macro_rules! for_remainder_mask_lanes {
    (
        $src_chunk:expr,
        $base:expr,
        $remainder:expr, |
        $bit_idx:ident,
        $i:ident,
        $valid:ident |
        $body:block
    ) => {
        for $bit_idx in 0..$remainder {
            let $i = $base + $bit_idx;
            let $valid = ($src_chunk >> $bit_idx) & 1 == 1;
            $body
        }
    };
}

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

impl<T: Copy> IndexedSource for &mut [T] {
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

/// An [`IndexedSource`] that also supports unchecked indexed writes — the binding
/// for in-place kernels.
///
/// Implemented for `&mut [T]`; not implemented for [`LaneZip`] (you can't write a
/// `(A, B)` pair back to two separate sources via a single index).
pub trait IndexedSink: IndexedSource {
    /// Write `value` into lane `i` without bounds checking.
    ///
    /// # Safety
    ///
    /// `i` must be strictly less than `self.len()`.
    unsafe fn set_unchecked(&mut self, i: usize, value: Self::Item);
}

impl<T: Copy> IndexedSink for &mut [T] {
    #[inline]
    unsafe fn set_unchecked(&mut self, i: usize, value: T) {
        // SAFETY: caller guarantees i < self.len().
        unsafe { *<[T]>::get_unchecked_mut(self, i) = value };
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
        for_full_mask_lanes!(src_chunk, base, |bit_idx, i, bit| {
            // SAFETY: i < chunks_count * 64 <= len.
            let v = unsafe { values.get_unchecked(i) };
            unsafe { out.get_unchecked_mut(i).write(f(v, bit)) };
        });
    }

    if remainder != 0 {
        let src_chunk = chunks.remainder_bits();
        let base = chunks_count * 64;
        for_remainder_mask_lanes!(src_chunk, base, remainder, |bit_idx, i, bit| {
            // SAFETY: i < len.
            let v = unsafe { values.get_unchecked(i) };
            unsafe { out.get_unchecked_mut(i).write(f(v, bit)) };
        });
    }
}

/// Fallible variant of [`map_with_mask`]. `f` returns `Option<R>`; `None` indicates a
/// per-lane failure (e.g. range overflow on a narrowing cast).
///
/// **Null-lane failures are filtered automatically.** If a null lane's stored value
/// causes `f(v, false)` to return `None`, the kernel does *not* propagate that as
/// `Err`. The per-lane `is_none()` flags are bit-packed into a `u64` at the lane's
/// position, then ANDed with the chunk's validity bitmap — null-lane bits vanish.
/// The closure may also explicitly suppress null-lane failures by branching on
/// `valid` itself; both behaviors compose.
///
/// ## Hot loop
///
/// `fail_bits |= (opt.is_none() as u64) << bit_idx`. After unrolling, `bit_idx` is a
/// compile-time constant per-iteration, so the shift folds. The closure receives
/// `(value, valid)`; LLVM DCEs the per-lane `(src_chunk >> bit_idx) & 1` extract
/// when the closure ignores `valid`, leaving a value-only SIMD body.
///
/// ## Attribution
///
/// `valid_failures = fail_bits & src_chunk` — non-zero only when at least one
/// valid lane failed. `trailing_zeros()` gives the first failing valid lane.
/// **No cold replay**: failure detection and lane attribution happen entirely in
/// the hot loop. Worst-case bounded per chunk regardless of how many null lanes
/// returned `None`.
///
/// On failure returns `Err(failing_lane_index)`. Lanes whose `f` returned `None`
/// write `R::default()` into `out`, but the contents of `out` must not be relied
/// upon when this function returns `Err`.
///
/// # Panics
///
/// Panics if `values.len() != mask.len()` or `out.len() != values.len()`.
#[inline]
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
        // Bit-pack per-lane fails into a u64 at lane-position. `bit_idx` is a
        // compile-time constant after unrolling, so the shift folds. The
        // `src_chunk` here is the validity bitmap for this chunk; the closure
        // still gets `bit` per lane — LLVM DCEs the per-lane mask extract if
        // the closure ignores it.
        let mut fail_bits: u64 = 0;
        for_full_mask_lanes!(src_chunk, base, |bit_idx, i, bit| {
            // SAFETY: i < chunks_count * 64 <= len.
            let v = unsafe { values.get_unchecked(i) };
            let opt = f(v, bit);
            fail_bits |= (opt.is_none() as u64) << bit_idx;
            let r = opt.unwrap_or_default();
            // SAFETY: i < len.
            unsafe { out.get_unchecked_mut(i).write(r) };
        });
        // Drop null-lane failures: only failures at lanes the mask marks as
        // valid count. Direct attribution via trailing_zeros — no cold replay.
        let valid_failures = fail_bits & src_chunk;
        if valid_failures != 0 {
            return Err(base + valid_failures.trailing_zeros() as usize);
        }
    }

    if remainder != 0 {
        let src_chunk = chunks.remainder_bits();
        let base = chunks_count * 64;
        let mut fail_bits: u64 = 0;
        for_remainder_mask_lanes!(src_chunk, base, remainder, |bit_idx, i, bit| {
            // SAFETY: i < len.
            let v = unsafe { values.get_unchecked(i) };
            let opt = f(v, bit);
            fail_bits |= (opt.is_none() as u64) << bit_idx;
            let r = opt.unwrap_or_default();
            // SAFETY: i < len.
            unsafe { out.get_unchecked_mut(i).write(r) };
        });
        let valid_failures = fail_bits & src_chunk;
        if valid_failures != 0 {
            return Err(base + valid_failures.trailing_zeros() as usize);
        }
    }

    Ok(())
}

/// Apply `f(value)` lane-by-lane with **no validity awareness at all** — every
/// closure invocation is treated as "happened", regardless of whether the lane
/// is null. Use this only when the input is known non-nullable.
///
/// For nullable inputs where the closure is infallible (no overflow / no error
/// branch), prefer [`map_with_mask`]; for nullable inputs with a fallible
/// closure, prefer [`try_map_with_mask`] — both correctly suppress
/// null-lane logic. This kernel exists for the narrow "no validity exists"
/// case (non-nullable column, internal pipelines, etc.).
///
/// # Panics
///
/// Panics if `out.len() != values.len()`.
#[inline]
pub fn map_no_validity<S, R, F>(values: S, out: &mut [MaybeUninit<R>], mut f: F)
where
    S: IndexedSource,
    F: FnMut(S::Item) -> R,
{
    let len = values.len();
    assert_eq!(out.len(), len, "out must have the same length as values");

    let chunks_count = len / 64;
    let remainder = len % 64;

    for chunk_idx in 0..chunks_count {
        let base = chunk_idx * 64;
        for_full_lanes!(base, |bit_idx, i| {
            // SAFETY: i < chunks_count * 64 <= len.
            let v = unsafe { values.get_unchecked(i) };
            unsafe { out.get_unchecked_mut(i).write(f(v)) };
        });
    }

    if remainder != 0 {
        let base = chunks_count * 64;
        for_remainder_lanes!(base, remainder, |bit_idx, i| {
            // SAFETY: i < len.
            let v = unsafe { values.get_unchecked(i) };
            unsafe { out.get_unchecked_mut(i).write(f(v)) };
        });
    }
}

/// Fallible map with **no validity awareness at all** — every `None` returned
/// by the closure is treated as a failure, even at null lanes.
///
/// # Use this only for non-nullable inputs.
///
/// For nullable inputs with a fallible closure, use
/// [`try_map_with_mask`] — it has the same value-only closure shape
/// (and the same perf win) but **correctly suppresses null-lane failures**
/// via per-chunk `fail_bits & mask_chunk`.
///
/// Using this kernel on a nullable input where a null lane's stored value
/// would cause `f` to return `None` will produce a spurious `Err`. This is a
/// correctness footgun on purpose — the name and this doc are how the API
/// signals "you must know your input has no nulls."
///
/// On failure returns `Err(failing_lane_index)`.
///
/// # Panics
///
/// Panics if `out.len() != values.len()`.
#[inline]
pub fn try_map_no_validity<S, R, F>(
    values: S,
    out: &mut [MaybeUninit<R>],
    mut f: F,
) -> Result<(), usize>
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
        for_full_lanes!(base, |bit_idx, i| {
            // SAFETY: i < chunks_count * 64 <= len.
            let v = unsafe { values.get_unchecked(i) };
            let opt = f(v);
            fail_acc |= opt.is_none() as u64;
            let r = opt.unwrap_or_default();
            // SAFETY: i < len.
            unsafe { out.get_unchecked_mut(i).write(r) };
        });
        if fail_acc != 0 {
            return Err(attribute_failure_no_mask(&values, base, 64, &mut f));
        }
    }

    if remainder != 0 {
        let base = chunks_count * 64;
        let mut fail_acc: u64 = 0;
        for_remainder_lanes!(base, remainder, |bit_idx, i| {
            // SAFETY: i < len.
            let v = unsafe { values.get_unchecked(i) };
            let opt = f(v);
            fail_acc |= opt.is_none() as u64;
            let r = opt.unwrap_or_default();
            // SAFETY: i < len.
            unsafe { out.get_unchecked_mut(i).write(r) };
        });
        if fail_acc != 0 {
            return Err(attribute_failure_no_mask(&values, base, remainder, &mut f));
        }
    }

    Ok(())
}

/// Shared cold scan: walks a chunk, returns the first lane index where
/// `lane_fails(bit_idx, value)` returns `true`. Used by
/// [`attribute_failure_no_mask`].
///
/// Caller guarantees `base + chunk_len <= values.len()`.
#[cold]
#[inline(never)]
fn cold_scan<S>(
    values: &S,
    base: usize,
    chunk_len: usize,
    mut lane_fails: impl FnMut(usize /* bit_idx */, S::Item) -> bool,
) -> usize
where
    S: IndexedSource,
{
    for bit_idx in 0..chunk_len {
        let i = base + bit_idx;
        // SAFETY: caller guarantees i < values.len().
        let v = unsafe { values.get_unchecked(i) };
        if lane_fails(bit_idx, v) {
            return i;
        }
    }
    unreachable!("cold_scan called without a failing lane")
}

/// Cold attribution for the no-mask variant. Replays `f` over the chunk to find
/// the first lane that returns `None`.
#[inline]
fn attribute_failure_no_mask<S, R, F>(values: &S, base: usize, chunk_len: usize, f: &mut F) -> usize
where
    S: IndexedSource,
    F: FnMut(S::Item) -> Option<R>,
{
    cold_scan(values, base, chunk_len, |_bit_idx, v| f(v).is_none())
}

/// In-place variant of [`map_with_mask`]. Each lane is replaced with
/// `f(values[i], mask[i])`. The source `S` must be writable (an [`IndexedSink`]).
///
/// # Panics
///
/// Panics if `values.len() != mask.len()`.
#[inline]
pub fn map_with_mask_in_place<S, F>(mut values: S, mask: &BitBuffer, mut f: F)
where
    S: IndexedSink,
    F: FnMut(S::Item, bool) -> S::Item,
{
    let len = values.len();
    assert_eq!(len, mask.len(), "values and mask must have the same length");

    let chunks = mask.chunks();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
        let base = chunk_idx * 64;
        for_full_mask_lanes!(src_chunk, base, |bit_idx, i, bit| {
            // SAFETY: i < chunks_count * 64 <= len.
            let v = unsafe { values.get_unchecked(i) };
            let r = f(v, bit);
            // SAFETY: i < len.
            unsafe { values.set_unchecked(i, r) };
        });
    }

    if remainder != 0 {
        let src_chunk = chunks.remainder_bits();
        let base = chunks_count * 64;
        for_remainder_mask_lanes!(src_chunk, base, remainder, |bit_idx, i, bit| {
            // SAFETY: i < len.
            let v = unsafe { values.get_unchecked(i) };
            let r = f(v, bit);
            // SAFETY: i < len.
            unsafe { values.set_unchecked(i, r) };
        });
    }
}

/// In-place variant of [`try_map_with_mask`]. Each lane of `values` is replaced
/// with `f(values[i], mask[i])`, or `S::Item::default()` if `f` returned `None`.
/// On failure returns `Err(first_failing_lane)`; lanes before that point have been
/// written, and lanes within the failing chunk hold their unwrapped-or-default
/// result. The buffer state on `Err` is intentionally unspecified.
///
/// ## Error attribution
///
/// Per-lane `is_none()` flags are folded into `first_fail` via a branchless
/// `min` of `(if is_none { i as u32 } else { u32::MAX })`. After the 64-lane
/// loop, `first_fail` holds the smallest failing index in the chunk (or `MAX`
/// if no failure). Vectorizes to NEON `bsl.16b` + `umin.4s` on AArch64. The
/// cold replay scheme used by [`try_map_with_mask`] isn't viable here because
/// the original input values have already been overwritten by the time we
/// would attribute the failure.
///
/// ## Why in-place is slower at cache-resident sizes
///
/// At sizes that fit in L1/L2 the in-place kernel is ~1.5× slower than the
/// out-of-place kernel despite having half the memory traffic, because input
/// and output share memory and the compiler must be conservative reordering
/// loads/stores across iterations. At sizes that exceed L2 the in-place kernel
/// wins back the gap by avoiding the second buffer's DRAM read+write traffic.
///
/// # Panics
///
/// Panics if `values.len() != mask.len()`.
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub fn try_map_with_mask_in_place<S, F>(
    mut values: S,
    mask: &BitBuffer,
    mut f: F,
) -> Result<(), usize>
where
    S: IndexedSink,
    S::Item: Default,
    F: FnMut(S::Item, bool) -> Option<S::Item>,
{
    let len = values.len();
    assert_eq!(len, mask.len(), "values and mask must have the same length");

    let chunks = mask.chunks();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
        // `count = 64` is a literal; `#[inline(always)]` on the helper inlines its body
        // into this loop and the compiler propagates 64 into the inner `0..count` bound,
        // unrolling exactly as `for_full_mask_lanes!` would.
        if let Some(failing) = try_inplace_chunk(&mut values, src_chunk, chunk_idx * 64, 64, &mut f)
        {
            return Err(failing as usize);
        }
    }

    if remainder != 0 {
        // Runtime `count = remainder` — same shape as the prior remainder loop.
        if let Some(failing) = try_inplace_chunk(
            &mut values,
            chunks.remainder_bits(),
            chunks_count * 64,
            remainder,
            &mut f,
        ) {
            return Err(failing as usize);
        }
    }

    Ok(())
}

/// Per-chunk worker for [`try_map_with_mask_in_place`]. Body written once; the kernel
/// calls this twice (with `count = 64` for full chunks, `count = remainder` for the
/// tail). `#[inline(always)]` so the const-64 unroll for the full-chunk callers is
/// preserved.
///
/// Returns `Some(first_failing_lane_index_as_u32)` if any lane in `[base, base+count)`
/// failed (cast width-truncated since `i < 2^32` in any realistic batch), else `None`.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
fn try_inplace_chunk<S, F>(
    values: &mut S,
    src_chunk: u64,
    base: usize,
    count: usize,
    f: &mut F,
) -> Option<u32>
where
    S: IndexedSink,
    S::Item: Default,
    F: FnMut(S::Item, bool) -> Option<S::Item>,
{
    let mut first_fail: u32 = u32::MAX;
    for bit_idx in 0..count {
        let i = base + bit_idx;
        let bit = (src_chunk >> bit_idx) & 1 == 1;
        // SAFETY: caller guarantees `base + count <= values.len()`.
        let v = unsafe { values.get_unchecked(i) };
        let opt = f(v, bit);
        let candidate = if opt.is_none() { i as u32 } else { u32::MAX };
        first_fail = first_fail.min(candidate);
        let r = opt.unwrap_or_default();
        // SAFETY: same as above.
        unsafe { values.set_unchecked(i, r) };
    }
    (first_fail != u32::MAX).then_some(first_fail)
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
        for_full_lanes!(base, |bit_idx, i| {
            // SAFETY: base + bit_idx < chunks_count * 64 <= len.
            let v = unsafe { values.get_unchecked(i) };
            packed |= (f(v) as u64) << bit_idx;
        });
        // SAFETY: chunk_idx < chunks_count <= out.len().
        unsafe { *out.get_unchecked_mut(chunk_idx) = packed };
    }

    if remainder != 0 {
        let base = chunks_count * 64;
        let mut packed = 0u64;
        for_remainder_lanes!(base, remainder, |bit_idx, i| {
            // SAFETY: base + bit_idx < len.
            let v = unsafe { values.get_unchecked(i) };
            packed |= (f(v) as u64) << bit_idx;
        });
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
        for_full_mask_lanes!(src_chunk, base, |bit_idx, i, bit| {
            // SAFETY: i < chunks_count * 64 <= len.
            let v = unsafe { values.get_unchecked(i) };
            packed |= (f(v, bit) as u64) << bit_idx;
        });
        // SAFETY: chunk_idx < chunks_count <= out.len().
        unsafe { *out.get_unchecked_mut(chunk_idx) = packed };
    }

    if remainder != 0 {
        let src_chunk = chunks.remainder_bits();
        let base = chunks_count * 64;
        let mut packed = 0u64;
        for_remainder_mask_lanes!(src_chunk, base, remainder, |bit_idx, i, bit| {
            // SAFETY: i < len.
            let v = unsafe { values.get_unchecked(i) };
            packed |= (f(v, bit) as u64) << bit_idx;
        });
        // SAFETY: chunks_count < out.len() because remainder != 0.
        unsafe { *out.get_unchecked_mut(chunks_count) = packed };
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::BitBufferMut;

    fn write_t<T: Copy>(out: Vec<MaybeUninit<T>>) -> Vec<T> {
        // SAFETY: tests always fully initialize the buffer.
        unsafe { std::mem::transmute(out) }
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
    fn try_map_with_mask_value_only_closure_filters_null_overflow() {
        // `|v, _|` closure that ignores validity. A null lane with an overflowing
        // value MUST NOT cause Err — the kernel's cold-path mask filter rescues us.
        let mut values: Vec<u64> = (0..200).collect();
        values[5] = u64::MAX; // null lane with overflowing value
        values[42] = u64::MAX; // null lane with overflowing value
        let mask = {
            let mut m = BitBufferMut::with_capacity(200);
            for i in 0..200 {
                m.append(i != 5 && i != 42);
            }
            m.freeze()
        };
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        let res = try_map_with_mask(values.as_slice(), &mask, &mut out, |v, _valid| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert!(
            res.is_ok(),
            "null-lane overflow should be filtered by the cold path"
        );
    }

    #[test]
    fn try_map_with_mask_value_only_closure_reports_first_valid_failure() {
        // Valid lane overflow must propagate — and the reported index must be
        // the lowest VALID failing lane, even if earlier null lanes also "failed"
        // their unconditional cast.
        let mut values: Vec<u64> = (0..200).collect();
        values[5] = u64::MAX; // null lane — filtered out
        values[42] = u64::MAX; // null lane — filtered out
        values[77] = u64::MAX; // VALID lane — should be reported
        values[100] = u64::MAX; // VALID lane — higher index, ignored
        let mask = {
            let mut m = BitBufferMut::with_capacity(200);
            for i in 0..200 {
                m.append(i != 5 && i != 42);
            }
            m.freeze()
        };
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        let res = try_map_with_mask(values.as_slice(), &mask, &mut out, |v, _valid| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert_eq!(res, Err(77));
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
    fn map_with_mask_in_place_basic() {
        let mut values: Vec<u32> = (0..130).collect();
        let mask = {
            let mut m = BitBufferMut::with_capacity(130);
            for i in 0..130 {
                m.append(i % 2 == 0);
            }
            m.freeze()
        };
        map_with_mask_in_place(values.as_mut_slice(), &mask, |v, valid| {
            v.wrapping_mul(valid as u32)
        });
        let expected: Vec<u32> = (0..130u32)
            .map(|v| if v % 2 == 0 { v } else { 0 })
            .collect();
        assert_eq!(values, expected);
    }

    #[test]
    fn try_map_with_mask_in_place_all_ok() {
        let mut values: Vec<u32> = (0..200).collect();
        let mask = BitBuffer::new_set(200);
        let res = try_map_with_mask_in_place(values.as_mut_slice(), &mask, |v, valid| {
            let scaled = v.wrapping_mul(valid as u32);
            scaled.checked_mul(2)
        });
        assert!(res.is_ok());
        let expected: Vec<u32> = (0..200u32).map(|v| v * 2).collect();
        assert_eq!(values, expected);
    }

    #[test]
    fn try_map_with_mask_in_place_first_failing_chunk_wins() {
        let mut values: Vec<u32> = (0..200).collect();
        values[83] = u32::MAX;
        values[150] = u32::MAX;
        let mask = BitBuffer::new_set(200);
        let res =
            try_map_with_mask_in_place(values.as_mut_slice(), &mask, |v, _valid| v.checked_mul(2));
        assert_eq!(res, Err(83));
    }

    #[test]
    fn try_map_with_mask_in_place_within_chunk_reports_lowest() {
        let mut values: Vec<u32> = (0..200).collect();
        values[80] = u32::MAX;
        values[100] = u32::MAX;
        let mask = BitBuffer::new_set(200);
        let res =
            try_map_with_mask_in_place(values.as_mut_slice(), &mask, |v, _valid| v.checked_mul(2));
        assert_eq!(res, Err(80));
    }

    #[test]
    fn try_map_with_mask_in_place_single_failure_lane_exact() {
        let mut values: Vec<u32> = (0..200).collect();
        values[42] = u32::MAX;
        let mask = BitBuffer::new_set(200);
        let res =
            try_map_with_mask_in_place(values.as_mut_slice(), &mask, |v, _valid| v.checked_mul(2));
        assert_eq!(res, Err(42));
    }

    #[test]
    fn try_map_with_mask_in_place_null_bypass() {
        let mut values: Vec<u32> = (0..200).collect();
        values[5] = u32::MAX;
        let mask = {
            let mut m = BitBufferMut::with_capacity(200);
            for i in 0..200 {
                m.append(i != 5);
            }
            m.freeze()
        };
        let res = try_map_with_mask_in_place(values.as_mut_slice(), &mask, |v, valid| {
            v.wrapping_mul(valid as u32).checked_mul(2)
        });
        assert!(res.is_ok());
        assert_eq!(values[5], 0);
        assert_eq!(values[6], 12);
    }

    #[test]
    fn try_map_with_mask_in_place_remainder_overflow() {
        let mut values: Vec<u32> = (0..130).collect();
        values[129] = u32::MAX;
        let mask = BitBuffer::new_set(130);
        let res =
            try_map_with_mask_in_place(values.as_mut_slice(), &mask, |v, _valid| v.checked_mul(2));
        assert_eq!(res, Err(129));
    }

    #[test]
    fn try_map_with_mask_in_place_sliced_mask() {
        let big = BitBuffer::new_set(256);
        let mask = big.slice(13..143);
        assert_eq!(mask.len(), 130);

        let mut values: Vec<u32> = (0..130).collect();
        values[77] = u32::MAX;
        let res =
            try_map_with_mask_in_place(values.as_mut_slice(), &mask, |v, _valid| v.checked_mul(2));
        assert_eq!(res, Err(77));
    }

    #[test]
    fn try_map_with_mask_in_place_partial_chunk_success() {
        let mut values: Vec<u32> = (0..130).collect();
        let mask = BitBuffer::new_set(130);
        let res = try_map_with_mask_in_place(values.as_mut_slice(), &mask, |v, _valid| Some(v + 1));
        assert!(res.is_ok());
        assert_eq!(values[0], 1);
        assert_eq!(values[63], 64);
        assert_eq!(values[64], 65);
        assert_eq!(values[129], 130);
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
