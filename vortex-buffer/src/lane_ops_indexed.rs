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

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::mem::align_of;
use std::mem::size_of;

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
/// `Write` is the type written by `set_unchecked` and may differ from
/// `IndexedSource::Item` (the read type). For the canonical `&mut [T]` impl
/// both are `T`. The decoupling is what makes [`ReinterpretSink`] possible —
/// a wrapper that reads `F` and writes `T` over the same backing memory when
/// the two have identical size and alignment.
///
/// Implemented for `&mut [T]`; not implemented for [`LaneZip`] (you can't write a
/// `(A, B)` pair back to two separate sources via a single index).
pub trait IndexedSink: IndexedSource {
    /// The per-lane write type. Equal to `<Self as IndexedSource>::Item` for
    /// `&mut [T]`; different for [`ReinterpretSink`].
    type Write: Copy;

    /// Write `value` into lane `i` without bounds checking.
    ///
    /// # Safety
    ///
    /// `i` must be strictly less than `self.len()`.
    unsafe fn set_unchecked(&mut self, i: usize, value: Self::Write);
}

impl<T: Copy> IndexedSink for &mut [T] {
    type Write = T;
    #[inline]
    unsafe fn set_unchecked(&mut self, i: usize, value: T) {
        // SAFETY: caller guarantees i < self.len().
        unsafe { *<[T]>::get_unchecked_mut(self, i) = value };
    }
}

/// A sink that reads `F`-values and writes `T`-values over the same backing
/// slice of `F`, reinterpreting each `T` as `F`-bits on write.
///
/// Requires `size_of::<F>() == size_of::<T>()` and `align_of::<F>() == align_of::<T>()`.
/// Both hold for any pair of `NativePType` primitives with equal byte width
/// (e.g. `u32` ↔ `f32`, `u64` ↔ `i64`, `f64` ↔ `u64`).
///
/// Use this when an in-place kernel needs to convert lanes between two
/// types of identical width without allocating a second buffer. After the
/// kernel completes every slot holds a valid `T`-bit pattern; the caller
/// can recover a typed view via `BufferMut::transmute::<T>()`.
pub struct ReinterpretSink<'a, F, T> {
    slice: &'a mut [F],
    _phantom: PhantomData<T>,
}

impl<'a, F, T> ReinterpretSink<'a, F, T> {
    /// Construct a `ReinterpretSink` from `&mut [F]`.
    ///
    /// # Panics
    ///
    /// Panics if `size_of::<F>() != size_of::<T>()` or
    /// `align_of::<F>() != align_of::<T>()`.
    pub fn new(slice: &'a mut [F]) -> Self {
        assert_eq!(
            size_of::<F>(),
            size_of::<T>(),
            "ReinterpretSink requires F and T to have the same size",
        );
        assert_eq!(
            align_of::<F>(),
            align_of::<T>(),
            "ReinterpretSink requires F and T to have the same alignment",
        );
        Self {
            slice,
            _phantom: PhantomData,
        }
    }
}

impl<F: Copy, T: Copy> IndexedSource for ReinterpretSink<'_, F, T> {
    type Item = F;
    #[inline]
    fn len(&self) -> usize {
        self.slice.len()
    }
    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> F {
        // SAFETY: caller guarantees i < self.slice.len(). Pointer arithmetic
        // avoids method-resolution ambiguity between `<[F]>::get_unchecked` and
        // `IndexedSource::get_unchecked`.
        unsafe { *self.slice.as_ptr().add(i) }
    }
}

impl<F: Copy, T: Copy> IndexedSink for ReinterpretSink<'_, F, T> {
    type Write = T;
    #[inline]
    unsafe fn set_unchecked(&mut self, i: usize, value: T) {
        // SAFETY: caller guarantees i < self.slice.len(); `new` enforces
        // size_of::<F>() == size_of::<T>() and align_of::<F>() == align_of::<T>(),
        // so the F-slot can hold a `T` without overflow or misalignment.
        unsafe {
            let ptr = self.slice.as_mut_ptr().add(i) as *mut T;
            ptr.write(value);
        }
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
    /// Per-chunk worker. Called twice (literal `64` for full chunks, `remainder`
    /// for the tail). `#[inline(always)]` preserves the const-64 unroll at the
    /// full-chunk call site via constant propagation through inlining.
    #[inline(always)]
    fn chunk<S, R, F>(
        values: &S,
        out: &mut [MaybeUninit<R>],
        f: &mut F,
        src_chunk: u64,
        base: usize,
        count: usize,
    ) where
        S: IndexedSource,
        F: FnMut(S::Item, bool) -> R,
    {
        for bit_idx in 0..count {
            let i = base + bit_idx;
            let bit = (src_chunk >> bit_idx) & 1 == 1;
            // SAFETY: caller guarantees base + count <= len.
            let v = unsafe { values.get_unchecked(i) };
            unsafe { out.get_unchecked_mut(i).write(f(v, bit)) };
        }
    }

    let len = values.len();
    assert_eq!(len, mask.len(), "values and mask must have the same length");
    assert_eq!(out.len(), len, "out must have the same length as values");

    let chunks = mask.chunks();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
        chunk(&values, out, &mut f, src_chunk, chunk_idx * 64, 64);
    }
    if remainder != 0 {
        chunk(
            &values,
            out,
            &mut f,
            chunks.remainder_bits(),
            chunks_count * 64,
            remainder,
        );
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
    #[inline(always)]
    fn chunk<S, R, F>(
        values: &S,
        out: &mut [MaybeUninit<R>],
        f: &mut F,
        src_chunk: u64,
        base: usize,
        count: usize,
    ) -> Option<usize>
    where
        S: IndexedSource,
        R: Copy + Default,
        F: FnMut(S::Item, bool) -> Option<R>,
    {
        let mut fail_bits: u64 = 0;
        for bit_idx in 0..count {
            let i = base + bit_idx;
            let bit = (src_chunk >> bit_idx) & 1 == 1;
            // SAFETY: caller guarantees base + count <= len.
            let v = unsafe { values.get_unchecked(i) };
            let opt = f(v, bit);
            fail_bits |= (opt.is_none() as u64) << bit_idx;
            let r = opt.unwrap_or_default();
            unsafe { out.get_unchecked_mut(i).write(r) };
        }
        let valid_failures = fail_bits & src_chunk;
        (valid_failures != 0).then_some(base + valid_failures.trailing_zeros() as usize)
    }

    let len = values.len();
    assert_eq!(len, mask.len(), "values and mask must have the same length");
    assert_eq!(out.len(), len, "out must have the same length as values");

    let chunks = mask.chunks();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
        if let Some(idx) = chunk(&values, out, &mut f, src_chunk, chunk_idx * 64, 64) {
            return Err(idx);
        }
    }
    if remainder != 0
        && let Some(idx) = chunk(
            &values,
            out,
            &mut f,
            chunks.remainder_bits(),
            chunks_count * 64,
            remainder,
        )
    {
        return Err(idx);
    }
    Ok(())
}

/// Apply `f(value)` lane-by-lane with **no validity awareness at all** — every
/// closure invocation is treated as "happened", regardless of whether the lane
/// is null. Use this only when the input is known non-nullable.
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
    #[inline(always)]
    fn chunk<S, R, F>(values: &S, out: &mut [MaybeUninit<R>], f: &mut F, base: usize, count: usize)
    where
        S: IndexedSource,
        F: FnMut(S::Item) -> R,
    {
        for bit_idx in 0..count {
            let i = base + bit_idx;
            // SAFETY: caller guarantees base + count <= len.
            let v = unsafe { values.get_unchecked(i) };
            unsafe { out.get_unchecked_mut(i).write(f(v)) };
        }
    }

    let len = values.len();
    assert_eq!(out.len(), len, "out must have the same length as values");

    let chunks_count = len / 64;
    let remainder = len % 64;

    for chunk_idx in 0..chunks_count {
        chunk(&values, out, &mut f, chunk_idx * 64, 64);
    }
    if remainder != 0 {
        chunk(&values, out, &mut f, chunks_count * 64, remainder);
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
    /// Returns `true` if any lane in `[base, base+count)` failed (OR-reduced);
    /// the cold attribution path is called at the kernel level so it can be
    /// inlined separately for full vs remainder.
    #[inline(always)]
    fn chunk<S, R, F>(
        values: &S,
        out: &mut [MaybeUninit<R>],
        f: &mut F,
        base: usize,
        count: usize,
    ) -> bool
    where
        S: IndexedSource,
        R: Copy + Default,
        F: FnMut(S::Item) -> Option<R>,
    {
        let mut fail_acc: u64 = 0;
        for bit_idx in 0..count {
            let i = base + bit_idx;
            // SAFETY: caller guarantees base + count <= len.
            let v = unsafe { values.get_unchecked(i) };
            let opt = f(v);
            fail_acc |= opt.is_none() as u64;
            let r = opt.unwrap_or_default();
            unsafe { out.get_unchecked_mut(i).write(r) };
        }
        fail_acc != 0
    }

    let len = values.len();
    assert_eq!(out.len(), len, "out must have the same length as values");

    let chunks_count = len / 64;
    let remainder = len % 64;

    for chunk_idx in 0..chunks_count {
        let base = chunk_idx * 64;
        if chunk(&values, out, &mut f, base, 64) {
            return Err(attribute_failure_no_mask(&values, base, 64, &mut f));
        }
    }
    if remainder != 0 {
        let base = chunks_count * 64;
        if chunk(&values, out, &mut f, base, remainder) {
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

/// In-place variant of [`map_no_validity`]. Each lane is replaced with `f(values[i])`.
/// The source `S` must be writable (an [`IndexedSink`]).
///
/// The closure reads `S::Item` and returns `S::Write`. For the common case
/// `S = &mut [T]` both are `T`; for [`ReinterpretSink`] the read and write
/// types can differ (e.g. read `f32`, write `u32`) over the same backing memory
/// when sizes and alignments match.
///
/// As with [`map_no_validity`], use this only when the input is known
/// non-nullable.
#[inline]
pub fn map_no_validity_in_place<S, F>(mut values: S, mut f: F)
where
    S: IndexedSink,
    F: FnMut(S::Item) -> S::Write,
{
    #[inline(always)]
    fn chunk<S, F>(values: &mut S, f: &mut F, base: usize, count: usize)
    where
        S: IndexedSink,
        F: FnMut(S::Item) -> S::Write,
    {
        for bit_idx in 0..count {
            let i = base + bit_idx;
            // SAFETY: caller guarantees base + count <= len.
            let v = unsafe { values.get_unchecked(i) };
            let r = f(v);
            // SAFETY: caller guarantees base + count <= len.
            unsafe { values.set_unchecked(i, r) };
        }
    }

    let len = values.len();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for chunk_idx in 0..chunks_count {
        chunk(&mut values, &mut f, chunk_idx * 64, 64);
    }
    if remainder != 0 {
        chunk(&mut values, &mut f, chunks_count * 64, remainder);
    }
}

/// In-place variant of [`try_map_no_validity`]. Each lane is replaced with
/// `f(values[i])`, or `S::Write::default()` when `f` returns `None`. On failure
/// returns `Err(first_failing_lane)`; the buffer state on `Err` is unspecified.
///
/// As with [`try_map_no_validity`], use this only when the input is known
/// non-nullable — a `None` from `f` is treated as a failure regardless of any
/// upstream validity bitmap.
///
/// ## Error attribution
///
/// Per-lane `is_none()` flags are folded into `first_fail` via the same
/// branchless `min` scheme as [`try_map_with_mask_in_place`]. Cold replay
/// isn't viable here because the original input values have already been
/// overwritten by the time we'd attribute the failure.
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub fn try_map_no_validity_in_place<S, F>(mut values: S, mut f: F) -> Result<(), usize>
where
    S: IndexedSink,
    S::Write: Default,
    F: FnMut(S::Item) -> Option<S::Write>,
{
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn chunk<S, F>(values: &mut S, base: usize, count: usize, f: &mut F) -> Option<u32>
    where
        S: IndexedSink,
        S::Write: Default,
        F: FnMut(S::Item) -> Option<S::Write>,
    {
        let mut first_fail: u32 = u32::MAX;
        for bit_idx in 0..count {
            let i = base + bit_idx;
            // SAFETY: caller guarantees base + count <= len.
            let v = unsafe { values.get_unchecked(i) };
            let opt = f(v);
            let candidate = if opt.is_none() { i as u32 } else { u32::MAX };
            first_fail = first_fail.min(candidate);
            let r = opt.unwrap_or_default();
            // SAFETY: caller guarantees base + count <= len.
            unsafe { values.set_unchecked(i, r) };
        }
        (first_fail != u32::MAX).then_some(first_fail)
    }

    let len = values.len();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for chunk_idx in 0..chunks_count {
        if let Some(failing) = chunk(&mut values, chunk_idx * 64, 64, &mut f) {
            return Err(failing as usize);
        }
    }
    if remainder != 0
        && let Some(failing) = chunk(&mut values, chunks_count * 64, remainder, &mut f)
    {
        return Err(failing as usize);
    }
    Ok(())
}

/// In-place variant of [`map_with_mask`]. Each lane is replaced with
/// `f(values[i], mask[i])`. The source `S` must be writable (an [`IndexedSink`]).
///
/// The closure reads `S::Item` and returns `S::Write`. For the common case
/// `S = &mut [T]` both are `T`; for [`ReinterpretSink`] the read and write
/// types can differ (e.g. read `f32`, write `u32`) over the same backing
/// memory when sizes and alignments match.
///
/// # Panics
///
/// Panics if `values.len() != mask.len()`.
#[inline]
pub fn map_with_mask_in_place<S, F>(mut values: S, mask: &BitBuffer, mut f: F)
where
    S: IndexedSink,
    F: FnMut(S::Item, bool) -> S::Write,
{
    #[inline(always)]
    fn chunk<S, F>(values: &mut S, f: &mut F, src_chunk: u64, base: usize, count: usize)
    where
        S: IndexedSink,
        F: FnMut(S::Item, bool) -> S::Write,
    {
        for bit_idx in 0..count {
            let i = base + bit_idx;
            let bit = (src_chunk >> bit_idx) & 1 == 1;
            // SAFETY: caller guarantees base + count <= len.
            let v = unsafe { values.get_unchecked(i) };
            let r = f(v, bit);
            unsafe { values.set_unchecked(i, r) };
        }
    }

    let len = values.len();
    assert_eq!(len, mask.len(), "values and mask must have the same length");

    let chunks = mask.chunks();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
        chunk(&mut values, &mut f, src_chunk, chunk_idx * 64, 64);
    }
    if remainder != 0 {
        chunk(
            &mut values,
            &mut f,
            chunks.remainder_bits(),
            chunks_count * 64,
            remainder,
        );
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
    S::Write: Default,
    F: FnMut(S::Item, bool) -> Option<S::Write>,
{
    /// Returns `Some(first_failing_lane_index_as_u32)` if any lane in
    /// `[base, base+count)` failed (cast width-truncated since `i < 2^32` in any
    /// realistic batch), else `None`. `#[inline(always)]` so the literal `64` at the
    /// full-chunk call site enables const-propagation through inlining.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn chunk<S, F>(
        values: &mut S,
        src_chunk: u64,
        base: usize,
        count: usize,
        f: &mut F,
    ) -> Option<u32>
    where
        S: IndexedSink,
        S::Write: Default,
        F: FnMut(S::Item, bool) -> Option<S::Write>,
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
            unsafe { values.set_unchecked(i, r) };
        }
        (first_fail != u32::MAX).then_some(first_fail)
    }

    let len = values.len();
    assert_eq!(len, mask.len(), "values and mask must have the same length");

    let chunks = mask.chunks();
    let chunks_count = len / 64;
    let remainder = len % 64;

    for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
        if let Some(failing) = chunk(&mut values, src_chunk, chunk_idx * 64, 64, &mut f) {
            return Err(failing as usize);
        }
    }
    if remainder != 0
        && let Some(failing) = chunk(
            &mut values,
            chunks.remainder_bits(),
            chunks_count * 64,
            remainder,
            &mut f,
        )
    {
        return Err(failing as usize);
    }
    Ok(())
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
    fn reinterpret_sink_same_width_f32_u32() {
        // Read f32, write u32-bits in place. After transmuting the slice back to u32 we
        // should see exactly the bit patterns the closure produced.
        let mut buf: Vec<f32> = (0..130).map(|i| i as f32).collect();
        let mask = BitBuffer::new_set(130);
        try_map_with_mask_in_place(
            ReinterpretSink::<f32, u32>::new(buf.as_mut_slice()),
            &mask,
            |f, _valid| Some(f.to_bits().wrapping_add(1)),
        )
        .unwrap();
        // SAFETY: same size + alignment for f32 and u32; every slot now holds a u32 written by
        // the closure.
        let as_u32: &[u32] =
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u32, buf.len()) };
        for (i, &got) in as_u32.iter().enumerate() {
            assert_eq!(got, (i as f32).to_bits().wrapping_add(1), "lane {i}");
        }
    }

    #[test]
    fn reinterpret_sink_failure_reports_lane() {
        // Closure fails at a specific lane; the kernel must report that lane index.
        let mut buf: Vec<f32> = (0..200).map(|i| i as f32).collect();
        let mask = BitBuffer::new_set(200);
        let res = try_map_with_mask_in_place(
            ReinterpretSink::<f32, u32>::new(buf.as_mut_slice()),
            &mask,
            |f, _valid| {
                if f as u32 == 137 {
                    None
                } else {
                    Some(f as u32)
                }
            },
        );
        assert_eq!(res, Err(137));
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
}
