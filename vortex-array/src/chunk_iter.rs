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
//! - Chunks may be shorter than [`DECOMPRESS_CHUNK_LEN`] (sliced blocks, filtered blocks).
//! - Chunks are producer-owned scratch: a sink may mutate one freely — wrappers rely on this to
//!   transform values in place — and its contents are invalid once `accept` returns.
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
//! intermediates outweigh that copy — see [`MIN_STREAMING_CHAIN`] for the measured threshold the
//! executor uses.

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::builders::PrimitiveBuilder;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;

/// The target number of elements per streamed chunk.
///
/// This matches the FastLanes block size so leaf decompressors can hand their unpack scratch
/// buffer to the sink without copying. Producers may emit shorter chunks (e.g. a sliced first or
/// last block).
pub const DECOMPRESS_CHUNK_LEN: usize = 1024;

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

    /// Returns whether the executor will canonicalize this array by streaming chunks (see
    /// [`should_execute_via_chunks`]). Exposed for diagnostics and tests.
    #[doc(hidden)]
    pub fn should_execute_via_chunks(&self) -> bool {
        should_execute_via_chunks(self)
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
        self.decompress_chunks_unchecked(ctx, sink)
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
            self.decompress_chunks_unchecked(ctx, sink)
        } else {
            decompress_chunks_via_canonical(self, ctx, sink)
        }
    }

    fn decompress_chunks_unchecked(
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
/// [`set_chunked_execute_enabled`] at runtime (benchmarks and tests use this to compare both
/// executor paths in one process).
static CHUNKED_EXECUTE_ENABLED: std::sync::LazyLock<AtomicBool> = std::sync::LazyLock::new(|| {
    AtomicBool::new(!std::env::var("VORTEX_CHUNKED_EXECUTE").is_ok_and(|v| v == "0"))
});

#[doc(hidden)]
pub fn set_chunked_execute_enabled(enabled: bool) {
    CHUNKED_EXECUTE_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Minimum streaming chain length for the executor to canonicalize via chunk streaming.
///
/// Streaming trades one full-length intermediate buffer per encoding level for one pass over an
/// L1-resident block, but pays a scratch-to-output copy at the end. That trade only pays off once
/// a tree is deep enough. Measured on 4Mi-row `FoR`-over-`BitPacked` stacks (three rounds,
/// medians): the marginal cost of an added level is ~0.83ms level-wise versus ~0.53ms streaming,
/// so chains of 5 and 9 nodes run 1.24x and 1.46x faster streaming, while chains of 2-3 are
/// within noise or slightly slower (their level-wise paths decode straight into the destination
/// via `decode_into`, leaving no intermediate to eliminate). Streaming also has far tighter tail
/// latency: p100/median ~1.2x versus 2.5-5x for level-wise, which repeatedly allocates
/// full-length intermediates.
const MIN_STREAMING_CHAIN: usize = 4;

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
fn streaming_chain_len(array: &ArrayRef) -> usize {
    if !array.supports_decompress_chunks() {
        return 0;
    }
    1 + array
        .children_iter()
        .filter(|child| !child.is_canonical() && child.len() == array.len())
        .map(streaming_chain_len)
        .max()
        .unwrap_or(0)
}

/// Returns whether the executor should canonicalize this array by streaming chunks rather than
/// materializing an intermediate per encoding level.
///
/// True when streaming is enabled and the tree's streaming chain reaches
/// [`MIN_STREAMING_CHAIN`] nodes — the depth at which streaming is measured to win.
pub(crate) fn should_execute_via_chunks(array: &ArrayRef) -> bool {
    CHUNKED_EXECUTE_ENABLED.load(Ordering::Relaxed)
        && streaming_chain_len(array) >= MIN_STREAMING_CHAIN
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
    let validity_mask = array.validity()?.execute_mask(len, ctx)?;
    match_each_native_ptype!(array.dtype().as_ptype(), |T| {
        let mut builder = PrimitiveBuilder::<T>::with_capacity(array.dtype().nullability(), len);
        let mut uninit_range = builder.uninit_range(len);
        // SAFETY: every value slot is initialized by the chunk stream below, which covers
        // exactly 0..len (checked in debug builds).
        unsafe {
            uninit_range.append_mask(&validity_mask);
        }
        {
            // SAFETY: the chunk stream covers 0..len contiguously.
            let dst = unsafe { uninit_range.slice_uninit_mut(0, len) };
            let mut sink = BuilderSink::<T> { dst };
            array.decompress_chunks(ctx, &mut sink)?;
        }
        // SAFETY: mask appended for len rows and all len values initialized above.
        unsafe {
            uninit_range.finish();
        }
        Ok(builder.finish_into_primitive())
    })
}

struct BuilderSink<'a, T> {
    dst: &'a mut [MaybeUninit<T>],
}

impl<T: NativePType> ChunkSink for BuilderSink<'_, T> {
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, row_range: Range<usize>) -> VortexResult<()> {
        // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout.
        let src: &[MaybeUninit<T>] = unsafe { std::mem::transmute(chunk.as_slice::<T>()) };
        self.dst[row_range].copy_from_slice(src);
        Ok(())
    }
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
        self.next_row = row_range.end;
        self.inner.accept(chunk, row_range)
    }
}

/// Fallback chunked decompression: execute the array to a canonical [`PrimitiveArray`], then
/// stream copies of its values in [`DECOMPRESS_CHUNK_LEN`]-sized chunks.
///
/// This is the two-pass baseline. It copies each chunk into a single reusable scratch buffer
/// (allocated once) because sinks receive exclusive, mutable chunks.
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

fn stream_slice_chunks<T: NativePType>(values: &[T], sink: &mut dyn ChunkSink) -> VortexResult<()> {
    let mut scratch = vec![T::default(); values.len().min(DECOMPRESS_CHUNK_LEN)];
    for (chunk_idx, chunk) in values.chunks(DECOMPRESS_CHUNK_LEN).enumerate() {
        let start = chunk_idx * DECOMPRESS_CHUNK_LEN;
        let scratch = &mut scratch[..chunk.len()];
        scratch.copy_from_slice(chunk);
        sink.accept(ChunkMut::new(scratch), start..start + chunk.len())?;
    }
    Ok(())
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
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::patches::Patches;
    use crate::scalar::Scalar;

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
        // A primitive-typed encoding tree with no streaming support: Dict over primitives.
        let array = crate::arrays::DictArray::try_new(
            buffer![0u32, 1, 0, 1].into_array(),
            buffer![10i32, 20].into_array(),
        )?
        .into_array();
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
