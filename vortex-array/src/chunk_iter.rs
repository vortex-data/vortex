// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression.
//!
//! [`ArrayRef::decompress_chunks`] walks an encoding tree's decompressed values in cache-resident
//! chunks of [`DECOMPRESS_CHUNK_LEN`] without materializing the array. Leaf encodings stream
//! blocks straight out of their decompression kernel; wrapper encodings compose by interposing a
//! stack-allocated [`ChunkSink`] adapter and recursing into their child through the erased
//! [`ArrayRef`] entry point, so a whole tree decompresses and transforms one L1-resident block at
//! a time.
//!
//! # The vtable API
//!
//! Encodings implement two methods (see [`VTable`](crate::vtable::VTable)):
//!
//! ```ignore
//! fn supports_decompress_chunks(array: ArrayView<'_, Self>) -> bool;      // default: false
//! fn decompress_chunks(
//!     array: ArrayView<'_, Self>,
//!     ctx: &mut ExecutionCtx,
//!     sink: &mut dyn ChunkSink,
//! ) -> VortexResult<()>;                                                  // default: unsupported
//! ```
//!
//! They are separate because support must be answerable *before* any work happens, and because it
//! **cascades**: a wrapper only advertises support when the children it streams from do, so the
//! whole tree is validated up front and [`ArrayRef::decompress_chunks`] can reject an unsupported
//! tree without emitting a partial stream. There is no silent fallback — callers that want one
//! ask for it by name via [`ArrayRef::decompress_chunks_or_materialize`].
//!
//! # Contract
//!
//! - Chunks arrive in order, are contiguous, and cover `0..array.len()` exactly (debug-checked).
//! - Chunks hold at most [`DECOMPRESS_CHUNK_LEN`] rows (debug-checked), and may hold fewer
//!   (sliced blocks, filtered blocks), so adapters that change the value type can convert into a
//!   fixed stack scratch chunk.
//! - Chunks are producer-owned scratch: a sink may mutate one freely — wrappers rely on this to
//!   transform values in place — and its contents are invalid once `accept` returns.
//! - A sink may offer the memory it would copy a chunk into (`ChunkSink::destination`), so the
//!   producer decodes straight into it and reports it with `accept_written`. Materializing then
//!   costs no copy out of scratch.
//! - Validity is *not* streamed. Positions that are logically null hold unspecified but
//!   initialized values, matching what `execute` produces; read `array.validity()` separately.
//! - Primitive-typed arrays only.
//!
//! # Cost model
//!
//! Dispatch is per chunk, never per value: one virtual call per ~1024 values per encoding level
//! (~0.06 ns/element), with all per-element work monomorphized behind a real typed slice. No heap
//! state is introduced descending the tree — the sink chain is one stack frame per level — and
//! leaf producers reuse the scratch buffer their decompressor already owns.
//!
//! What streaming trades: one full-length intermediate buffer per encoding level becomes one pass
//! over an L1-resident block, at the cost of a final scratch-to-output copy when the consumer
//! wants a buffer. So it wins outright for consumers that never materialize (folding values
//! directly), and for materialization it wins once a tree is deep enough that the eliminated
//! intermediates outweigh that copy. The executor streams trees whose streaming chain reaches
//! three nodes, the measured break-even (`MIN_STREAMING_CHAIN`).

use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use num_traits::AsPrimitive;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::builders::PrimitiveBuilder;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::executor::DonePredicate;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::matcher::Matcher;
use crate::patches::Patches;

/// The target number of elements per streamed chunk.
///
/// This matches the FastLanes block size so leaf decompressors can hand their unpack scratch
/// buffer to the sink without copying. Producers may emit shorter chunks (e.g. a sliced first or
/// last block).
pub const DECOMPRESS_CHUNK_LEN: usize = 1024;

/// A stack scratch chunk for producers to decode into where their sink offers no destination.
///
/// It is zeroed on first use rather than up front: a materializing stream decodes whole chunks
/// straight into its sink's destination and may never touch it, and zeroing 8 KiB per stream is
/// measurable on small arrays.
pub struct ScratchChunk<T> {
    values: [MaybeUninit<T>; DECOMPRESS_CHUNK_LEN],
    zeroed: bool,
}

impl<T: NativePType> ScratchChunk<T> {
    /// A scratch chunk that is not zeroed until first used.
    #[inline]
    pub fn new() -> Self {
        Self {
            values: [const { MaybeUninit::uninit() }; DECOMPRESS_CHUNK_LEN],
            zeroed: false,
        }
    }

    /// The scratch values, zeroed the first time they are asked for.
    #[inline]
    pub fn values(&mut self) -> &mut [T; DECOMPRESS_CHUNK_LEN] {
        if !self.zeroed {
            self.zero();
        }
        // SAFETY: every value was initialized when the chunk was zeroed, and `MaybeUninit<T>`
        // has `T`'s layout.
        unsafe { &mut *self.values.as_mut_ptr().cast::<[T; DECOMPRESS_CHUNK_LEN]>() }
    }

    #[cold]
    fn zero(&mut self) {
        self.values.fill(MaybeUninit::new(T::default()));
        self.zeroed = true;
    }
}

impl<T: NativePType> Default for ScratchChunk<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A type-erased, mutable view over one chunk of decompressed primitive values.
///
/// This is a `(PType, *mut, len)` triple rather than a generic slice so it can cross the
/// `dyn ChunkSink` boundary; sinks recover the typed slice with [`Self::as_slice_mut`]. The
/// erasure cost is paid once per chunk, not per element.
pub struct ChunkMut<'a> {
    ptype: PType,
    data: *mut u8,
    len: usize,
    _marker: PhantomData<&'a mut u8>,
}

impl<'a> ChunkMut<'a> {
    /// Wrap a typed slice of decompressed values.
    pub fn new<T: NativePType>(values: &'a mut [T]) -> Self {
        Self {
            ptype: T::PTYPE,
            data: values.as_mut_ptr().cast(),
            len: values.len(),
            _marker: PhantomData,
        }
    }

    /// The primitive type of the values in this chunk.
    #[inline]
    pub fn ptype(&self) -> PType {
        self.ptype
    }

    /// The number of values in this chunk.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// View the chunk as a typed slice.
    ///
    /// # Panics
    /// Panics if `T::PTYPE` does not match the chunk's ptype.
    #[inline]
    pub fn as_slice<T: NativePType>(&self) -> &[T] {
        assert_eq!(T::PTYPE, self.ptype, "ChunkMut ptype mismatch");
        // SAFETY: constructed from a valid `&mut [T]` with matching ptype; the lifetime is tied
        // to the original borrow via `_marker`.
        unsafe { std::slice::from_raw_parts(self.data.cast(), self.len) }
    }

    /// View the chunk as a mutable typed slice.
    ///
    /// # Panics
    /// Panics if `T::PTYPE` does not match the chunk's ptype.
    #[inline]
    pub fn as_slice_mut<T: NativePType>(&mut self) -> &mut [T] {
        assert_eq!(T::PTYPE, self.ptype, "ChunkMut ptype mismatch");
        // SAFETY: constructed from a valid, exclusively borrowed `&mut [T]` with matching ptype.
        unsafe { std::slice::from_raw_parts_mut(self.data.cast(), self.len) }
    }

    /// View a full chunk as a fixed-size block, or `None` for a shorter one (a sliced first or
    /// last block, a filtered block), so consumers can process the common full chunk with a
    /// constant length, as a kernel written for whole blocks would.
    ///
    /// # Panics
    /// Panics if `T::PTYPE` does not match the chunk's ptype.
    #[inline]
    pub fn as_block<T: NativePType>(&self) -> Option<&[T; DECOMPRESS_CHUNK_LEN]> {
        self.as_slice::<T>().try_into().ok()
    }

    /// Re-tag the chunk as values of `U`, a type of the same width, after transforming its values
    /// in place (e.g. decoding ALP integers into floats).
    ///
    /// The bytes are not converted, which is sound because every primitive type is valid for
    /// any bit pattern.
    ///
    /// # Panics
    /// Panics if `U` is not the same width as the chunk's ptype.
    #[inline]
    pub fn retype<U: NativePType>(self) -> ChunkMut<'a> {
        self.retype_to(U::PTYPE)
    }

    /// Re-tag the chunk as values of `ptype`, as [`Self::retype`] does for a type chosen at
    /// runtime, e.g. to decode a signed type through its unsigned counterpart.
    ///
    /// # Panics
    /// Panics if `ptype` is not the same width as the chunk's ptype.
    #[inline]
    pub fn retype_to(self, ptype: PType) -> ChunkMut<'a> {
        assert_eq!(
            ptype.byte_width(),
            self.ptype.byte_width(),
            "ChunkMut can only be re-tagged to a type of the same width"
        );
        ChunkMut {
            ptype,
            data: self.data,
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Narrow the chunk to the values in `range`, e.g. to forward only the rows a slice covers.
    ///
    /// # Panics
    /// Panics if `range` is not within the chunk.
    #[inline]
    pub fn narrow(self, range: Range<usize>) -> ChunkMut<'a> {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "range {range:?} out of bounds for a chunk of {}",
            self.len
        );
        ChunkMut {
            ptype: self.ptype,
            // SAFETY: `range.start` is within the chunk, so the offset stays in its allocation.
            data: unsafe { self.data.add(range.start * self.ptype.byte_width()) },
            len: range.len(),
            _marker: PhantomData,
        }
    }

    /// Reborrow this chunk with a shorter lifetime, e.g. to forward it to a downstream sink.
    #[inline]
    pub fn reborrow(&mut self) -> ChunkMut<'_> {
        ChunkMut {
            ptype: self.ptype,
            data: self.data,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

/// Consumer side of [`ArrayRef::decompress_chunks`].
///
/// Implementations receive each decompressed chunk exactly once, in array order. `row_range` is
/// the range of logical rows (relative to the array being iterated) that `chunk` covers; its
/// length always equals `chunk.len()`.
pub trait ChunkSink {
    /// Accept the next chunk of decompressed values.
    fn accept(&mut self, chunk: ChunkMut<'_>, row_range: Range<usize>) -> VortexResult<()>;

    /// Offer the memory the sink would copy rows `row_range` into, so the producer can decode them
    /// there directly and spare the copy. A producer that takes it writes every value of it and
    /// then calls [`Self::accept_written`] with the same rows instead of [`Self::accept`].
    ///
    /// The memory may hold uninitialized values until the producer writes them. Sinks without
    /// such memory, e.g. ones that fold values, return `None`, the default. Requesting the same
    /// rows again returns the same memory with what was written to it, so adapters that transform
    /// chunks in place forward their inner sink's destination and re-request it in
    /// [`Self::accept_written`] to transform the written values.
    fn destination(&mut self, row_range: Range<usize>) -> Option<ChunkMut<'_>> {
        _ = row_range;
        None
    }

    /// Accept rows `row_range`, which the producer wrote into the memory from
    /// [`Self::destination`].
    fn accept_written(&mut self, row_range: Range<usize>) -> VortexResult<()> {
        vortex_bail!("accept_written for rows {row_range:?} of a sink that offers no destination")
    }
}

impl<F> ChunkSink for F
where
    F: FnMut(ChunkMut<'_>, Range<usize>) -> VortexResult<()>,
{
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, row_range: Range<usize>) -> VortexResult<()> {
        self(chunk, row_range)
    }
}

impl ArrayRef {
    /// Returns whether this array (recursively, through wrapper encodings) can stream its
    /// decompressed values via [`Self::decompress_chunks`] without materializing anything.
    pub fn supports_decompress_chunks(&self) -> bool {
        self.dtype().is_primitive() && self.dyn_array().supports_decompress_chunks(self)
    }

    /// Returns whether the executor will canonicalize this array by streaming chunks when
    /// executing it to canonical (see [`should_execute_via_chunks`]). Exposed for diagnostics and
    /// tests.
    #[doc(hidden)]
    pub fn should_execute_via_chunks(&self) -> bool {
        should_execute_via_chunks(self, AnyCanonical::matches)
    }

    /// Stream the array's decompressed values through `sink` in cache-resident chunks.
    ///
    /// See the [module docs](self) for the contract and cost model. This errors — without
    /// emitting any chunks — if the encoding tree does not support streaming (check with
    /// [`Self::supports_decompress_chunks`]); it never silently falls back to full
    /// materialization. Use [`Self::decompress_chunks_or_materialize`] to opt into that
    /// fallback explicitly.
    pub fn decompress_chunks(
        &self,
        ctx: &mut ExecutionCtx,
        sink: &mut dyn ChunkSink,
    ) -> VortexResult<()> {
        vortex_ensure!(
            self.supports_decompress_chunks(),
            "decompress_chunks is not supported by this array tree (root encoding {}, dtype {}); \
             use decompress_chunks_or_materialize for an explicit two-pass fallback",
            self.encoding_id(),
            self.dtype()
        );
        self.decompress_child_chunks(ctx, sink)
    }

    /// Stream chunks if the tree supports it, otherwise fall back to executing the array to
    /// canonical and streaming the materialized result — the two-pass behavior, chosen by name.
    pub fn decompress_chunks_or_materialize(
        &self,
        ctx: &mut ExecutionCtx,
        sink: &mut dyn ChunkSink,
    ) -> VortexResult<()> {
        vortex_ensure!(
            self.dtype().is_primitive(),
            "decompress_chunks requires a primitive-typed array, got {}",
            self.dtype()
        );
        if self.supports_decompress_chunks() {
            self.decompress_child_chunks(ctx, sink)
        } else {
            decompress_chunks_via_canonical(self, ctx, sink)
        }
    }

    /// Stream a child of an encoding's [`VTable::decompress_chunks`] without first checking that
    /// it streams, because the encoding's own [`VTable::supports_decompress_chunks`] already
    /// required it. [`Self::decompress_chunks`] would walk the child's tree again at every level.
    ///
    /// Called on a tree that does not stream, it errors when it reaches the unsupported encoding,
    /// possibly after emitting earlier chunks.
    ///
    /// [`VTable::decompress_chunks`]: crate::vtable::VTable::decompress_chunks
    /// [`VTable::supports_decompress_chunks`]: crate::vtable::VTable::supports_decompress_chunks
    pub fn decompress_child_chunks(
        &self,
        ctx: &mut ExecutionCtx,
        sink: &mut dyn ChunkSink,
    ) -> VortexResult<()> {
        #[cfg(debug_assertions)]
        {
            let mut checked = CoverageCheckSink {
                inner: sink,
                next_row: 0,
                ptype: self.dtype().as_ptype(),
            };
            self.dyn_array()
                .decompress_chunks(self, ctx, &mut checked)?;
            debug_assert_eq!(
                checked.next_row,
                self.len(),
                "decompress_chunks did not cover the full array"
            );
            Ok(())
        }
        #[cfg(not(debug_assertions))]
        self.dyn_array().decompress_chunks(self, ctx, sink)
    }
}

/// Global kill switch for the executor's stream-to-canonical shortcut (see
/// [`execute_via_chunks`]). Enabled by default, subject to the depth rule in
/// [`should_execute_via_chunks`]; set `VORTEX_CHUNKED_EXECUTE=0` to disable it entirely, or use
/// [`set_chunked_execute_enabled`] at runtime (benchmarks use this to compare both executor paths
/// in one process).
static CHUNKED_EXECUTE_ENABLED: std::sync::LazyLock<AtomicBool> = std::sync::LazyLock::new(|| {
    AtomicBool::new(!std::env::var("VORTEX_CHUNKED_EXECUTE").is_ok_and(|v| v == "0"))
});

thread_local! {
    /// Set while [`without_chunked_execute`] runs, so only this thread executes level-wise.
    static CHUNKED_EXECUTE_SUPPRESSED: Cell<bool> = const { Cell::new(false) };
}

#[doc(hidden)]
pub fn set_chunked_execute_enabled(enabled: bool) {
    CHUNKED_EXECUTE_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Run `f` with the executor's streaming shortcut disabled on this thread only, e.g. to compute a
/// level-wise reference in a test without affecting tests running on other threads.
#[doc(hidden)]
pub fn without_chunked_execute<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            CHUNKED_EXECUTE_SUPPRESSED.set(self.0);
        }
    }
    let _restore = Restore(CHUNKED_EXECUTE_SUPPRESSED.replace(true));
    f()
}

/// Minimum streaming chain length for the executor to canonicalize via chunk streaming.
///
/// Streaming trades one full-length intermediate buffer per encoding level for one pass over an
/// L1-resident block, but pays a scratch-to-output copy at the end, so it pays off once a tree has
/// an intermediate to eliminate. On nested `FoR` over `BitPacked` (medians of three runs, with
/// mimalloc), streaming runs 0.89x, 1.07x, 1.10x and 1.13x as fast as level-wise execution at
/// chains of 2, 3, 4 and 9 nodes for 64Ki rows, and 1.07x, 1.64x, 1.95x and 2.64x for 4Mi rows,
/// whose intermediates no longer fit in cache. A chain of 2, one encoding over its leaf, usually
/// decodes straight into the destination level-wise, leaving nothing to eliminate.
const MIN_STREAMING_CHAIN: usize = 3;

/// Length of the streaming chain rooted at `array`: the number of consecutive
/// streaming-capable nodes from the root down to and including the deepest leaf producer.
///
/// Only children that a streaming parent actually decodes *through* extend the chain: a child
/// must itself stream, must not already be canonical (canonical children are read directly, with
/// no decode to eliminate), and must preserve row count. The cardinality rule is what keeps
/// selection encodings such as `Filter` out of the executor path: their level-wise kernels decode
/// the child straight into the output buffer and compact in place, so streaming would only add a
/// copy. Streaming them is still available to consumers that never materialize, via
/// [`ArrayRef::decompress_chunks`].
///
/// Returns `None` when a node the chain would decode through satisfies `is_done`: the executor
/// is executing toward that node, e.g. to export a run-end array as is, and streaming to
/// canonical would skip past it.
fn streaming_chain_len(array: &ArrayRef, is_done: DonePredicate) -> Option<usize> {
    if !array.supports_decompress_chunks() {
        return Some(0);
    }
    let mut longest = 0;
    for child in array.children_iter() {
        if child.len() != array.len() || child.is_canonical() {
            continue;
        }
        if is_done(child) {
            return None;
        }
        longest = longest.max(streaming_chain_len(child, is_done)?);
    }
    Some(1 + longest)
}

/// Whether the chain of same-length, non-canonical children below `array` reaches `depth`
/// nodes, counting `array` itself.
///
/// This walk makes no encoding calls, so the executor turns away the common shallow trees before
/// asking any encoding whether it streams. A canonical child ends the chain, but checking that
/// costs more than the rest of the walk, so it is checked only for children deep enough to count.
fn spine_reaches(array: &ArrayRef, depth: usize) -> bool {
    depth <= 1
        || array.children_iter().any(|child| {
            child.len() == array.len() && spine_reaches(child, depth - 1) && !child.is_canonical()
        })
}

/// Returns whether the executor should canonicalize this array by streaming chunks rather than
/// materializing an intermediate per encoding level.
///
/// True when streaming is enabled and the tree's streaming chain reaches
/// [`MIN_STREAMING_CHAIN`] nodes — the depth at which streaming is measured to win.
pub(crate) fn should_execute_via_chunks(array: &ArrayRef, is_done: DonePredicate) -> bool {
    CHUNKED_EXECUTE_ENABLED.load(Ordering::Relaxed)
        && !CHUNKED_EXECUTE_SUPPRESSED.get()
        // Only primitive arrays stream, and the dtype is the cheapest thing to check.
        && array.dtype().is_primitive()
        && spine_reaches(array, MIN_STREAMING_CHAIN)
        && streaming_chain_len(array, is_done).is_some_and(|len| len >= MIN_STREAMING_CHAIN)
}

/// Execute a streaming-capable primitive array tree to a canonical [`PrimitiveArray`] by
/// decompressing chunks straight into the output buffer: each block is decoded and transformed
/// while L1-resident, and the only full-length write is the final copy into the builder.
///
/// The caller must have checked [`ArrayRef::supports_decompress_chunks`].
pub fn execute_via_chunks(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let len = array.len();
    // A non-nullable tree is all valid; asking its encodings would only walk the tree to say so.
    let validity_mask = if array.dtype().is_nullable() {
        array.validity()?.execute_mask(len, ctx)?
    } else {
        Mask::new_true(len)
    };
    match_each_native_ptype!(array.dtype().as_ptype(), |T| {
        let mut builder = PrimitiveBuilder::<T>::with_capacity_in(
            array.dtype().nullability(),
            len,
            ctx.allocator(),
        );
        let mut uninit_range = builder.uninit_range(len);
        // SAFETY: the range is only finished below once every value slot is initialized.
        unsafe {
            uninit_range.append_mask(&validity_mask);
        }
        {
            // SAFETY: `BuilderSink` initializes the slots it accepts, and the stream must cover
            // 0..len contiguously before the range is finished.
            let dst = unsafe { uninit_range.slice_uninit_mut(0, len) };
            let mut sink = BuilderSink::<T> { dst, written: 0 };
            // The caller checked that the tree streams; an encoding that does not errors out.
            array.decompress_child_chunks(ctx, &mut sink)?;
            vortex_ensure!(
                sink.written == len,
                "decompress_chunks of {} streamed {} of {len} rows",
                array.encoding_id(),
                sink.written
            );
        }
        // SAFETY: mask appended for len rows, and the stream initialized all len values.
        unsafe {
            uninit_range.finish();
        }
        Ok(builder.finish_into_primitive())
    })
}

struct BuilderSink<'a, T> {
    dst: &'a mut [MaybeUninit<T>],
    /// Rows accepted so far. Chunks must arrive contiguously, so `dst[..written]` is initialized.
    written: usize,
}

impl<T> BuilderSink<'_, T> {
    /// Record that rows `row_range` follow the rows accepted so far. Checked in release builds
    /// too, because the builder exposes every row the stream claims to have written.
    #[inline]
    fn advance(&mut self, row_range: &Range<usize>) -> VortexResult<()> {
        vortex_ensure!(
            row_range.start == self.written,
            "decompress_chunks streamed rows {row_range:?} after {} rows",
            self.written
        );
        self.written = row_range.end;
        Ok(())
    }
}

impl<T: NativePType> ChunkSink for BuilderSink<'_, T> {
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, row_range: Range<usize>) -> VortexResult<()> {
        self.advance(&row_range)?;
        self.dst[row_range].write_copy_of_slice(chunk.as_slice::<T>());
        Ok(())
    }

    #[inline]
    fn destination(&mut self, row_range: Range<usize>) -> Option<ChunkMut<'_>> {
        let dst = &mut self.dst[row_range];
        // SAFETY: `MaybeUninit<T>` has `T`'s layout, and producers only write a destination
        // before anything reads it, as decoding into a builder's spare capacity does elsewhere.
        let dst =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<T>(), dst.len()) };
        Some(ChunkMut::new(dst))
    }

    #[inline]
    fn accept_written(&mut self, row_range: Range<usize>) -> VortexResult<()> {
        vortex_ensure!(
            row_range.end <= self.dst.len(),
            "decompress_chunks wrote rows {row_range:?} past {} rows",
            self.dst.len()
        );
        self.advance(&row_range)
    }
}

/// Emit rows `row_range`, at most one chunk, of a stream of `ptype` values, decoded by `decode`:
/// straight into the sink's destination when it offers one, else into `scratch`.
///
/// `T` may differ from `ptype` in signedness only, so signed types decode through their unsigned
/// counterparts.
#[inline]
pub fn emit_with<T: NativePType>(
    sink: &mut dyn ChunkSink,
    ptype: PType,
    row_range: Range<usize>,
    scratch: &mut ScratchChunk<T>,
    decode: impl FnOnce(&mut [T]) -> VortexResult<()>,
) -> VortexResult<()> {
    if let Some(destination) = sink.destination(row_range.clone()) {
        with_block_len(destination.retype::<T>().as_slice_mut::<T>(), decode)?;
        return sink.accept_written(row_range);
    }
    let chunk = &mut scratch.values()[..row_range.len()];
    with_block_len(chunk, decode)?;
    sink.accept(ChunkMut::new(chunk).retype_to(ptype), row_range)
}

/// Call `f` with `values`, through a fixed-size block when they fill one, so that `f`, inlined,
/// decodes the common full chunk with a constant length.
#[inline]
fn with_block_len<T, R>(values: &mut [T], f: impl FnOnce(&mut [T]) -> R) -> R {
    match <&mut [T; DECOMPRESS_CHUNK_LEN]>::try_from(&mut *values) {
        Ok(block) => f(block),
        Err(_) => f(values),
    }
}

/// Stream `len` rows through `sink`, each chunk written by `fill` into the sink's destination
/// when it offers one, else into one stack scratch chunk. This is the shape of every leaf producer
/// that generates its values (constants, sequences, runs) or copies them out of a buffer.
pub fn stream_from_fn<T: NativePType>(
    len: usize,
    sink: &mut dyn ChunkSink,
    mut fill: impl FnMut(&mut [T], Range<usize>) -> VortexResult<()>,
) -> VortexResult<()> {
    let mut scratch = ScratchChunk::new();
    let mut start = 0;
    while start < len {
        let end = (start + DECOMPRESS_CHUNK_LEN).min(len);
        emit_with(sink, T::PTYPE, start..end, &mut scratch, |chunk| {
            fill(chunk, start..end)
        })?;
        start = end;
    }
    Ok(())
}

/// Sparse patches resolved once into row-sorted `(row, value)` pairs, applied to streamed chunks
/// with a forward cursor.
///
/// This is the only heap state streaming producers allocate, and it is proportional to the patch
/// count, not the array length.
pub struct ChunkPatches<T> {
    patches: Vec<(usize, T)>,
    cursor: usize,
}

impl<T: NativePType> ChunkPatches<T> {
    /// Wrap `(row, value)` pairs that are already sorted by row.
    pub fn new(patches: Vec<(usize, T)>) -> Self {
        debug_assert!(patches.is_sorted_by_key(|&(row, _)| row));
        Self { patches, cursor: 0 }
    }

    /// Resolve `patches`, mapping each value through `map(row, value)`.
    pub fn try_from_patches(
        patches: &Patches,
        ctx: &mut ExecutionCtx,
        map: impl Fn(usize, T) -> T,
    ) -> VortexResult<Self> {
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        let values = values.as_slice::<T>();
        let offset = patches.offset();
        let patches = match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
            indices
                .as_slice::<P>()
                .iter()
                .zip(values)
                .map(|(&index, &value)| {
                    let row = <P as AsPrimitive<usize>>::as_(index) - offset;
                    (row, map(row, value))
                })
                .collect()
        });
        Ok(Self::new(patches))
    }

    /// Overwrite the patched rows of `chunk`, which starts at row `start`.
    ///
    /// Chunks must be applied in row order, as they are streamed.
    #[inline]
    pub fn apply(&mut self, chunk: &mut [T], start: usize) {
        let end = start + chunk.len();
        while let Some(&(row, value)) = self.patches.get(self.cursor)
            && row < end
        {
            chunk[row - start] = value;
            self.cursor += 1;
        }
    }
}

/// Adapts a decoder of whole [`DECOMPRESS_CHUNK_LEN`]-row blocks, such as FastLanes RLE or delta,
/// into a sink over its child's stream.
///
/// The child's chunks are regrouped into whole blocks aligned to the start of the stream: chunks
/// that already are whole aligned blocks pass straight through, others are buffered. Each block is
/// decoded by `decode(block_index, input, output)` into a scratch block, and the rows of it that
/// fall in the array's `offset..offset + len` window of the stream are forwarded, renumbered from
/// the window's start. Blocks past the window are never decoded.
///
/// Input chunks are read as `I` and output chunks are tagged `output_ptype`, each of which may
/// differ from the stream's type in signedness only, so signed types decode through their
/// unsigned counterparts.
///
/// Call [`Self::finish`] once the child's stream ends, which rejects a stream that stopped part
/// way through a block.
pub struct BlockDecodeSink<'a, I, O, D> {
    offset: usize,
    len: usize,
    output_ptype: PType,
    decode: D,
    pending: ScratchChunk<I>,
    pending_len: usize,
    output: ScratchChunk<O>,
    inner: &'a mut dyn ChunkSink,
}

impl<'a, I, O, D> BlockDecodeSink<'a, I, O, D>
where
    I: NativePType,
    O: NativePType,
    D: FnMut(usize, &[I; DECOMPRESS_CHUNK_LEN], &mut [O; DECOMPRESS_CHUNK_LEN]) -> VortexResult<()>,
{
    /// Forward rows `offset..offset + len` of the decoded stream, tagged `output_ptype`, to
    /// `inner`.
    pub fn new(
        offset: usize,
        len: usize,
        output_ptype: PType,
        decode: D,
        inner: &'a mut dyn ChunkSink,
    ) -> Self {
        Self {
            offset,
            len,
            output_ptype,
            decode,
            pending: ScratchChunk::new(),
            pending_len: 0,
            output: ScratchChunk::new(),
            inner,
        }
    }

    /// Check that the child's stream ended on a block boundary. A trailing partial block cannot
    /// be decoded, so its rows would otherwise be missing from the output.
    pub fn finish(self) -> VortexResult<()> {
        vortex_ensure!(
            self.pending_len == 0,
            "block decoder input ended {} rows into a {DECOMPRESS_CHUNK_LEN}-row block",
            self.pending_len
        );
        Ok(())
    }
}

impl<I, O, D> ChunkSink for BlockDecodeSink<'_, I, O, D>
where
    I: NativePType,
    O: NativePType,
    D: FnMut(usize, &[I; DECOMPRESS_CHUNK_LEN], &mut [O; DECOMPRESS_CHUNK_LEN]) -> VortexResult<()>,
{
    fn accept(&mut self, chunk: ChunkMut<'_>, rows: Range<usize>) -> VortexResult<()> {
        let chunk = chunk.retype_to(I::PTYPE);
        let mut input = chunk.as_slice::<I>();
        if self.pending_len == 0
            && rows.start.is_multiple_of(DECOMPRESS_CHUNK_LEN)
            && let Ok(block) = <&[I; DECOMPRESS_CHUNK_LEN]>::try_from(input)
        {
            return emit_block(
                rows.start / DECOMPRESS_CHUNK_LEN,
                block,
                &mut self.decode,
                &mut self.output,
                self.offset..self.offset + self.len,
                self.output_ptype,
                &mut *self.inner,
            );
        }

        let mut row = rows.start;
        while !input.is_empty() {
            let take = (DECOMPRESS_CHUNK_LEN - self.pending_len).min(input.len());
            let pending = self.pending.values();
            pending[self.pending_len..][..take].copy_from_slice(&input[..take]);
            self.pending_len += take;
            row += take;
            input = &input[take..];
            if self.pending_len == DECOMPRESS_CHUNK_LEN {
                self.pending_len = 0;
                emit_block(
                    row / DECOMPRESS_CHUNK_LEN - 1,
                    pending,
                    &mut self.decode,
                    &mut self.output,
                    self.offset..self.offset + self.len,
                    self.output_ptype,
                    &mut *self.inner,
                )?;
            }
        }
        Ok(())
    }
}

/// Decode block `index` of a stream with `decode` and forward the rows of it within `window`,
/// tagged `output_ptype`: a whole block decodes straight into the sink's destination when it
/// offers one, and a partial one into `output` first.
///
/// [`BlockDecodeSink`] uses this for the blocks it regroups from a child's stream; decoders that
/// read whole blocks of their input themselves, e.g. straight from bit-packed storage, call it
/// directly.
pub fn emit_block<I, O, D>(
    index: usize,
    input: &[I; DECOMPRESS_CHUNK_LEN],
    decode: &mut D,
    output: &mut ScratchChunk<O>,
    window: Range<usize>,
    output_ptype: PType,
    inner: &mut dyn ChunkSink,
) -> VortexResult<()>
where
    O: NativePType,
    D: FnMut(usize, &[I; DECOMPRESS_CHUNK_LEN], &mut [O; DECOMPRESS_CHUNK_LEN]) -> VortexResult<()>,
{
    let block_start = index * DECOMPRESS_CHUNK_LEN;
    let start = block_start.max(window.start);
    let end = (block_start + DECOMPRESS_CHUNK_LEN).min(window.end);
    if start >= end {
        return Ok(());
    }
    let rows = start - window.start..end - window.start;
    // A whole block decodes straight into the sink's destination when it offers one.
    if rows.len() == DECOMPRESS_CHUNK_LEN
        && let Some(destination) = inner.destination(rows.clone())
    {
        let mut destination = destination.retype::<O>();
        let out = <&mut [O; DECOMPRESS_CHUNK_LEN]>::try_from(destination.as_slice_mut::<O>())
            .unwrap_or_else(|_| unreachable!("a destination holds its rows"));
        decode(index, input, out)?;
        return inner.accept_written(rows);
    }
    let output = output.values();
    decode(index, input, output)?;
    let chunk = ChunkMut::new(&mut output[start - block_start..end - block_start]);
    inner.accept(chunk.retype_to(output_ptype), rows)
}

#[cfg(debug_assertions)]
struct CoverageCheckSink<'a> {
    inner: &'a mut dyn ChunkSink,
    next_row: usize,
    ptype: PType,
}

#[cfg(debug_assertions)]
impl ChunkSink for CoverageCheckSink<'_> {
    fn accept(&mut self, chunk: ChunkMut<'_>, row_range: Range<usize>) -> VortexResult<()> {
        debug_assert_eq!(row_range.start, self.next_row, "non-contiguous chunk");
        debug_assert_eq!(
            row_range.len(),
            chunk.len(),
            "chunk/row_range length mismatch"
        );
        debug_assert_eq!(chunk.ptype(), self.ptype, "chunk ptype mismatch");
        debug_assert!(
            chunk.len() <= DECOMPRESS_CHUNK_LEN,
            "chunk exceeds the chunk length"
        );
        self.next_row = row_range.end;
        self.inner.accept(chunk, row_range)
    }

    fn destination(&mut self, row_range: Range<usize>) -> Option<ChunkMut<'_>> {
        self.inner.destination(row_range)
    }

    fn accept_written(&mut self, row_range: Range<usize>) -> VortexResult<()> {
        debug_assert_eq!(row_range.start, self.next_row, "non-contiguous chunk");
        debug_assert!(
            row_range.len() <= DECOMPRESS_CHUNK_LEN,
            "chunk exceeds the chunk length"
        );
        self.next_row = row_range.end;
        self.inner.accept_written(row_range)
    }
}

/// Fallback chunked decompression: execute the array to a canonical [`PrimitiveArray`], then
/// stream copies of its values in [`DECOMPRESS_CHUNK_LEN`]-sized chunks.
///
/// This is the two-pass baseline.
pub fn decompress_chunks_via_canonical(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
    match_each_native_ptype!(primitive.ptype(), |T| {
        stream_slice_chunks::<T>(primitive.as_slice::<T>(), sink)
    })
}

/// Stream copies of `values` through `sink`, one stack scratch chunk at a time, since sinks receive
/// exclusive, mutable chunks.
pub fn stream_slice_chunks<T: NativePType>(
    values: &[T],
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    stream_from_fn(values.len(), sink, |chunk, rows| {
        chunk.copy_from_slice(&values[rows]);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ConstantArray;
    use crate::arrays::Patched;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::patches::Patches;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::operators::Operator;

    fn collect_chunks<T: NativePType>(array: &ArrayRef) -> VortexResult<Vec<T>> {
        let mut ctx = array_session().create_execution_ctx();
        let mut out = Vec::with_capacity(array.len());
        array.decompress_chunks(&mut ctx, &mut |chunk: ChunkMut<'_>,
                                                 _range: Range<usize>|
         -> VortexResult<()> {
            out.extend_from_slice(chunk.as_slice::<T>());
            Ok(())
        })?;
        Ok(out)
    }

    /// The builder exposes every row a stream claims to have written, so it rejects gaps.
    #[test]
    fn builder_sink_rejects_gaps() -> VortexResult<()> {
        let mut dst = [MaybeUninit::<u32>::uninit(); 8];
        let mut sink = BuilderSink {
            dst: &mut dst,
            written: 0,
        };
        sink.accept(ChunkMut::new(&mut [1u32, 2]), 0..2)?;
        assert!(sink.accept(ChunkMut::new(&mut [3u32]), 3..4).is_err());
        assert!(sink.accept_written(5..6).is_err());
        assert!(sink.accept_written(2..9).is_err());
        sink.accept_written(2..8)?;
        assert_eq!(sink.written, 8);
        Ok(())
    }

    /// A child stream that stops part way through a block cannot be decoded.
    #[test]
    fn block_decode_sink_rejects_partial_block() -> VortexResult<()> {
        let decode = |_: usize,
                      input: &[u32; DECOMPRESS_CHUNK_LEN],
                      out: &mut [u32; DECOMPRESS_CHUNK_LEN]| {
            out.copy_from_slice(input);
            Ok(())
        };
        let mut rows = 0;
        let mut count = |chunk: ChunkMut<'_>, _: Range<usize>| -> VortexResult<()> {
            rows += chunk.len();
            Ok(())
        };
        let mut input = [7u32; 1500];
        let mut sink = BlockDecodeSink::new(0, 1500, PType::U32, decode, &mut count);
        sink.accept(ChunkMut::new(&mut input[..1000]), 0..1000)?;
        sink.accept(ChunkMut::new(&mut input[1000..]), 1000..1500)?;
        assert!(sink.finish().is_err());
        assert_eq!(rows, DECOMPRESS_CHUNK_LEN);
        Ok(())
    }

    #[test]
    fn constant_chunks() -> VortexResult<()> {
        // Length deliberately not a multiple of the chunk size.
        let array = ConstantArray::new(7i32, 2500).into_array();
        assert!(array.supports_decompress_chunks());
        let chunked = collect_chunks::<i32>(&array)?;
        assert_eq!(chunked, vec![7i32; 2500]);
        Ok(())
    }

    #[test]
    fn null_constant_chunks_cover_length() -> VortexResult<()> {
        let array = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            100,
        )
        .into_array();
        // Values are unspecified for nulls; only coverage matters (checked in debug builds too).
        let chunked = collect_chunks::<i32>(&array)?;
        assert_eq!(chunked.len(), 100);
        Ok(())
    }

    #[test]
    fn patched_over_constant_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let inner = ConstantArray::new(0u16, 2500).into_array();
        let patches = Patches::new(
            2500,
            0,
            buffer![1u32, 1023, 1024, 2047, 2499].into_array(),
            buffer![11u16, 22, 33, 44, 55].into_array(),
            None,
        )?;
        let array = Patched::from_array_and_patches(inner, &patches, &mut ctx)?.into_array();

        assert!(array.supports_decompress_chunks());
        let chunked = collect_chunks::<u16>(&array)?;
        let expected = array.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(chunked.as_slice(), expected.as_slice::<u16>());
        Ok(())
    }

    #[test]
    fn unsupported_encoding_errors_without_emitting() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // A primitive-typed encoding tree with no streaming support: a lazy expression.
        let array = buffer![1i32, 2, 1, 2]
            .into_array()
            .binary(buffer![9i32, 18, 9, 18].into_array(), Operator::Add)?;
        assert!(!array.supports_decompress_chunks());

        let mut emitted = 0usize;
        let result = array.decompress_chunks(&mut ctx, &mut |chunk: ChunkMut<'_>,
                                                             _range: Range<usize>|
         -> VortexResult<()> {
            emitted += chunk.len();
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(emitted, 0, "no chunks may be emitted on unsupported trees");

        // The explicit fallback still works and covers the array.
        let mut out = Vec::new();
        array.decompress_chunks_or_materialize(&mut ctx, &mut |chunk: ChunkMut<'_>,
                                                                _range: Range<usize>|
         -> VortexResult<()> {
            out.extend_from_slice(chunk.as_slice::<i32>());
            Ok(())
        })?;
        assert_eq!(out, vec![10i32, 20, 10, 20]);
        Ok(())
    }
}
