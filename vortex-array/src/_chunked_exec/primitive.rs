// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chunked decode producers that emit primitive values into a [`Scratch`] buffer.
//!
//! See the module-level docs in [`super`] for the model. This file holds:
//!
//! - The [`PrimitiveChunkProducer`] trait — the producer contract.
//! - The [`PrimitiveChunkKernel`] trait + [`PrimitiveChunkKernelDispatcher`] — dyn-dispatch
//!   registry of fused kernels keyed on the encoding identifier of the outermost array.
//! - Concrete producers: [`SliceProducer`] (canonical fallback),
//!   [`DictPrimitiveProducer`] (chunked gather over a small materialized dict — this is
//!   also the implementation that *naturally fuses* `Dict<RunEnd<P>>` because [`DictKernel`]
//!   materializes its values slot via the regular executor), and [`RunEndPrimitiveProducer`]
//!   (standalone run-end streaming, built via [`build_runend_producer`] by encodings that
//!   know how to fetch the run-end children).
//! - [`build_primitive_producer`] — the entry point that dispatches into a registered
//!   fused kernel or falls back to a canonical-then-stream path.
//!
//! For v1, the fast paths require non-nullable inputs and primitive children of the
//! expected shape. Anything outside that envelope falls back to the canonical-then-stream
//! producer, which is correct but not faster than the array-by-array baseline.

use std::any::Any;
use std::mem::MaybeUninit;
use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use super::CHUNK_LEN;
use super::scratch::Scratch;
use crate::ArrayRef;
use crate::array::ArrayId;
use crate::array::VTable;
use crate::arrays::Dict;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::validity::Validity;

/// A streaming producer of primitive chunks.
///
/// Implementors write up to [`CHUNK_LEN`] elements into the supplied scratch and return
/// the initialized prefix to the driver. Returning `Ok(None)` signals end-of-stream.
///
/// The producer is responsible for owning anything it needs across chunk calls (e.g. a
/// materialized dictionary). The scratch is supplied by the driver and reused across
/// calls.
pub trait PrimitiveChunkProducer<T: NativePType>: Send {
    /// Decode the next chunk of values.
    fn next_chunk<'a>(
        &mut self,
        scratch: &'a mut Scratch<T>,
    ) -> VortexResult<Option<&'a [T]>>;

    /// Number of elements not yet produced.
    fn remaining(&self) -> usize;
}

/// Pluggable kernel that can build a fused chunk producer for an array.
///
/// Kernels are registered against the outermost encoding id; each kernel inspects the
/// children at build time to decide whether it can handle the array shape. Returning
/// `Ok(None)` means "this kernel doesn't apply"; the dispatcher tries the next kernel
/// (or the fallback) instead.
pub trait PrimitiveChunkKernel<T: NativePType>: Send + Sync {
    /// Try to build a fused producer for `array`. Returning `Ok(None)` defers to the next
    /// kernel.
    fn build(
        &self,
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Box<dyn PrimitiveChunkProducer<T>>>>;
}

/// A typed entry in [`PrimitiveChunkKernelDispatcher`].
///
/// Boxed so the registry can hold heterogeneous `T`s; the typed lookup downcasts via
/// `Any` to recover the strongly-typed kernel slice.
struct TypedKernels<T: NativePType> {
    by_encoding: rustc_hash::FxHashMap<ArrayId, Vec<Arc<dyn PrimitiveChunkKernel<T>>>>,
}

impl<T: NativePType> TypedKernels<T> {
    fn new() -> Self {
        Self {
            by_encoding: rustc_hash::FxHashMap::default(),
        }
    }
}

/// Registry of fused chunk kernels.
///
/// In the long run this would hang off [`vortex_session::VortexSession`] and ride the same
/// session-scoped lookup as the existing [`crate::optimizer::kernels::ArrayKernels`]. For
/// the v1 spike the dispatcher is constructed per-call; see [`default_dispatcher`].
pub struct PrimitiveChunkKernelDispatcher {
    // Keyed by TypeId of T to support multiple primitive output types.
    entries: rustc_hash::FxHashMap<std::any::TypeId, Box<dyn Any + Send + Sync>>,
}

impl Default for PrimitiveChunkKernelDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl PrimitiveChunkKernelDispatcher {
    /// Empty registry.
    pub fn new() -> Self {
        Self {
            entries: rustc_hash::FxHashMap::default(),
        }
    }

    /// Register `kernel` for outermost encoding `encoding`.
    pub fn register<T: NativePType>(
        &mut self,
        encoding: ArrayId,
        kernel: Arc<dyn PrimitiveChunkKernel<T>>,
    ) {
        let typed = self
            .entries
            .entry(std::any::TypeId::of::<T>())
            .or_insert_with(|| Box::new(TypedKernels::<T>::new()));
        let typed = typed
            .downcast_mut::<TypedKernels<T>>()
            .expect("TypedKernels<T> matches TypeId<T>");
        typed.by_encoding.entry(encoding).or_default().push(kernel);
    }

    fn kernels_for<T: NativePType>(
        &self,
        encoding: ArrayId,
    ) -> &[Arc<dyn PrimitiveChunkKernel<T>>] {
        let Some(typed) = self.entries.get(&std::any::TypeId::of::<T>()) else {
            return &[];
        };
        let typed = typed
            .downcast_ref::<TypedKernels<T>>()
            .expect("TypedKernels<T> matches TypeId<T>");
        typed
            .by_encoding
            .get(&encoding)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

/// Build the v1 default dispatcher with the in-crate kernels only (Dict).
///
/// To register the `RunEnd` kernel (and the fused `Dict<RunEnd<P>>` path that emerges from
/// it), call `vortex_runend::register_chunk_kernels(&mut dispatcher)` from the consumer
/// crate. Out-of-crate encodings layer their kernels on top in the same way.
pub fn default_dispatcher() -> PrimitiveChunkKernelDispatcher {
    let mut d = PrimitiveChunkKernelDispatcher::new();
    register_defaults(&mut d);
    d
}

/// Register the in-crate v1 kernels onto `dispatcher` for every supported `T`.
pub fn register_defaults(dispatcher: &mut PrimitiveChunkKernelDispatcher) {
    macro_rules! register_all_for {
        ($($T:ty),*) => {
            $(
                dispatcher.register::<$T>(Dict.id(), Arc::new(DictKernel::<$T>::new()));
            )*
        };
    }
    register_all_for!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
}

/// Build a chunk producer for `array`, dispatching to any registered fused kernel and
/// falling back to a canonicalize-then-stream producer if none apply.
///
/// The caller passes a dispatcher (typically obtained from [`default_dispatcher`]). For
/// session integration the caller would fetch the session-scoped registry, but the spike
/// stays out of `VortexSession` for now to keep the diff small.
pub fn build_primitive_producer<T: NativePType>(
    array: ArrayRef,
    dispatcher: &PrimitiveChunkKernelDispatcher,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn PrimitiveChunkProducer<T>>> {
    let encoding = array.encoding_id();
    for kernel in dispatcher.kernels_for::<T>(encoding) {
        if let Some(p) = kernel.build(&array, ctx)? {
            return Ok(p);
        }
    }
    // Fallback: canonicalize, then stream the resulting slice in fixed-size chunks.
    let canonical = array.execute::<PrimitiveArray>(ctx)?;
    Ok(Box::new(SliceProducer::<T>::from_primitive(canonical)?))
}

// ---------------------------------------------------------------------------------------
// Slice producer
// ---------------------------------------------------------------------------------------

/// Walks a primitive buffer and emits fixed-size chunks via `copy_from_slice`.
///
/// Used as the universal fallback; also useful as a baseline in benchmarks.
pub struct SliceProducer<T: NativePType> {
    buffer: Buffer<T>,
    cursor: usize,
}

impl<T: NativePType> SliceProducer<T> {
    /// Construct a slice producer over a primitive buffer.
    pub fn new(buffer: Buffer<T>) -> Self {
        Self { buffer, cursor: 0 }
    }

    /// Construct a slice producer from a [`PrimitiveArray`].
    ///
    /// Errors if the array has a different physical type than `T`.
    pub fn from_primitive(array: PrimitiveArray) -> VortexResult<Self> {
        if T::PTYPE != array.ptype() {
            vortex_error::vortex_bail!(
                "SliceProducer<{}> cannot be built from PrimitiveArray of {}",
                T::PTYPE,
                array.ptype()
            );
        }
        Ok(Self::new(array.into_buffer::<T>()))
    }
}

impl<T: NativePType> PrimitiveChunkProducer<T> for SliceProducer<T> {
    fn next_chunk<'a>(
        &mut self,
        scratch: &'a mut Scratch<T>,
    ) -> VortexResult<Option<&'a [T]>> {
        let len = self.buffer.as_slice().len();
        if self.cursor >= len {
            return Ok(None);
        }
        let take = CHUNK_LEN.min(len - self.cursor);
        let src = &self.buffer.as_slice()[self.cursor..self.cursor + take];
        let dst = &mut scratch.as_uninit_mut()[..take];
        // SAFETY: `MaybeUninit<T>` has the same layout as `T`; we overwrite exactly `take`.
        unsafe {
            let dst_ptr = dst.as_mut_ptr().cast::<T>();
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr, take);
        }
        self.cursor += take;
        // SAFETY: `take` elements were just initialized.
        Ok(Some(unsafe {
            std::slice::from_raw_parts(scratch.as_uninit_mut().as_ptr().cast::<T>(), take)
        }))
    }

    fn remaining(&self) -> usize {
        self.buffer.as_slice().len().saturating_sub(self.cursor)
    }
}

// ---------------------------------------------------------------------------------------
// Dict<Primitive> producer
// ---------------------------------------------------------------------------------------

/// Stream a `Dict<Primitive>` by chunked gather over the codes into a pre-materialized
/// values buffer.
///
/// The values are loaded once into a [`Buffer<T>`] (the "dictionary"); each chunk picks
/// up to [`CHUNK_LEN`] codes and gathers values into the scratch. The dictionary is
/// expected to fit in L1d (otherwise this encoding shouldn't have been chosen).
pub struct DictPrimitiveProducer<T: NativePType, I: NativePType> {
    dict: Buffer<T>,
    codes: Buffer<I>,
    cursor: usize,
}

impl<T: NativePType, I: NativePType> DictPrimitiveProducer<T, I> {
    /// Construct directly from raw dict and code buffers.
    pub fn new(dict: Buffer<T>, codes: Buffer<I>) -> Self {
        Self {
            dict,
            codes,
            cursor: 0,
        }
    }
}

impl<T: NativePType, I: NativePType> PrimitiveChunkProducer<T>
    for DictPrimitiveProducer<T, I>
where
    I: num_traits::AsPrimitive<usize>,
{
    fn next_chunk<'a>(
        &mut self,
        scratch: &'a mut Scratch<T>,
    ) -> VortexResult<Option<&'a [T]>> {
        let total = self.codes.as_slice().len();
        if self.cursor >= total {
            return Ok(None);
        }
        let take = CHUNK_LEN.min(total - self.cursor);
        let codes_chunk = &self.codes.as_slice()[self.cursor..self.cursor + take];
        let dst: &mut [MaybeUninit<T>] = &mut scratch.as_uninit_mut()[..take];
        let dict = self.dict.as_slice();

        let ptr = dst.as_mut_ptr().cast::<T>();
        for (i, code) in codes_chunk.iter().enumerate() {
            let idx: usize = (*code).as_();
            // SAFETY: capacity reserved by Scratch is CHUNK_LEN; i < take ≤ CHUNK_LEN.
            // The `dict[idx]` lookup is bounds-checked.
            unsafe { ptr.add(i).write(dict[idx]) };
        }
        self.cursor += take;
        // SAFETY: `take` elements just written.
        Ok(Some(unsafe { std::slice::from_raw_parts(ptr, take) }))
    }

    fn remaining(&self) -> usize {
        self.codes.as_slice().len().saturating_sub(self.cursor)
    }
}

/// The `Dict<Primitive>` kernel — and, when values are `RunEnd<Primitive>`, the fused
/// `Dict<RunEnd<Primitive>>` kernel. Both compose to a [`DictPrimitiveProducer`] after
/// any necessary up-front materialization of the (bounded) values slot.
pub struct DictKernel<T: NativePType> {
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: NativePType> DictKernel<T> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: NativePType> Default for DictKernel<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NativePType> PrimitiveChunkKernel<T> for DictKernel<T> {
    fn build(
        &self,
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Box<dyn PrimitiveChunkProducer<T>>>> {
        let Some(dict) = array.as_opt::<Dict>() else {
            return Ok(None);
        };
        if !matches!(array.dtype().nullability(), Nullability::NonNullable) {
            return Ok(None);
        }
        let codes = dict.codes();
        let values = dict.values();
        if !matches!(codes.dtype().nullability(), Nullability::NonNullable)
            || !matches!(values.dtype().nullability(), Nullability::NonNullable)
        {
            return Ok(None);
        }
        let DType::Primitive(values_ptype, _) = *values.dtype() else {
            return Ok(None);
        };
        if values_ptype != T::PTYPE {
            return Ok(None);
        }

        // Materialize the values slot to a primitive buffer of length `dict_len`.
        // Dictionary sizes are bounded by definition; this is the "up-front" step that
        // makes the streaming chunked gather possible. For `Dict<RunEnd<P>>` this is
        // also where the RunEnd is unrolled into a flat dict, so the produced kernel
        // is the *fused* implementation.
        let values_canonical = values.clone().execute::<PrimitiveArray>(ctx)?;
        let dict_buf: Buffer<T> = values_canonical.into_buffer::<T>();

        let codes_canonical = codes.clone().execute::<PrimitiveArray>(ctx)?;
        let codes_ptype = codes_canonical.ptype();

        Ok(Some(dispatch_dict_producer::<T>(
            dict_buf,
            codes_canonical,
            codes_ptype,
        )))
    }
}

fn dispatch_dict_producer<T: NativePType>(
    dict: Buffer<T>,
    codes: PrimitiveArray,
    codes_ptype: PType,
) -> Box<dyn PrimitiveChunkProducer<T>> {
    match_each_integer_ptype!(codes_ptype, |I| {
        Box::new(DictPrimitiveProducer::<T, I>::new(
            dict,
            codes.into_buffer::<I>(),
        ))
    })
}

// ---------------------------------------------------------------------------------------
// RunEnd<Primitive> producer
// ---------------------------------------------------------------------------------------

/// Stream a `RunEnd<Primitive>` by walking the run-ends and replicating each run value
/// into chunk-sized strides.
pub struct RunEndPrimitiveProducer<T: NativePType, E: NativePType> {
    values: Buffer<T>,
    ends: Buffer<E>,
    /// Logical position in the decoded array (post-offset).
    cursor: usize,
    /// Total logical length to produce.
    len: usize,
    /// The slicing offset to apply to `cursor` before searching ends.
    offset: usize,
    /// Cached run index for `cursor`, so we avoid re-searching across chunk boundaries.
    run: usize,
}

impl<T: NativePType, E: NativePType> RunEndPrimitiveProducer<T, E>
where
    E: num_traits::AsPrimitive<usize>,
{
    /// Construct from value + end buffers plus a logical (offset, len) window.
    pub fn new(
        values: Buffer<T>,
        ends: Buffer<E>,
        offset: usize,
        len: usize,
    ) -> VortexResult<Self> {
        let run = find_run(ends.as_slice(), offset);
        Ok(Self {
            values,
            ends,
            cursor: 0,
            len,
            offset,
            run,
        })
    }
}

impl<T: NativePType, E: NativePType> PrimitiveChunkProducer<T>
    for RunEndPrimitiveProducer<T, E>
where
    E: num_traits::AsPrimitive<usize>,
{
    fn next_chunk<'a>(
        &mut self,
        scratch: &'a mut Scratch<T>,
    ) -> VortexResult<Option<&'a [T]>> {
        if self.cursor >= self.len {
            return Ok(None);
        }

        let take = CHUNK_LEN.min(self.len - self.cursor);
        let dst: &mut [MaybeUninit<T>] = &mut scratch.as_uninit_mut()[..take];
        let dst_ptr = dst.as_mut_ptr().cast::<T>();

        let ends = self.ends.as_slice();
        let values = self.values.as_slice();

        let mut written = 0usize;
        // chunk-relative position: 0..take, encoded position: cursor+offset..
        let chunk_end_encoded = self.cursor + take + self.offset;
        while written < take {
            // ends[run] is the encoded-position end of run `run` (exclusive).
            let run_end_encoded: usize = ends[self.run].as_();
            let pos_encoded = self.cursor + written + self.offset;
            // Number of elements remaining in this run within the current chunk.
            let in_run = run_end_encoded.min(chunk_end_encoded) - pos_encoded;
            let value = values[self.run];
            // SAFETY: `written + in_run <= take ≤ CHUNK_LEN`. fill is a single store loop.
            unsafe {
                let slot = std::slice::from_raw_parts_mut(dst_ptr.add(written), in_run);
                for s in slot.iter_mut() {
                    *s = value;
                }
            }
            written += in_run;
            if run_end_encoded <= chunk_end_encoded && written < take {
                self.run += 1;
            } else if run_end_encoded == chunk_end_encoded {
                // exactly aligned; advance for the next chunk
                self.run += 1;
            }
        }

        self.cursor += take;
        // SAFETY: `take` elements written.
        Ok(Some(unsafe { std::slice::from_raw_parts(dst_ptr, take) }))
    }

    fn remaining(&self) -> usize {
        self.len.saturating_sub(self.cursor)
    }
}

fn find_run<E>(ends: &[E], target_encoded: usize) -> usize
where
    E: NativePType + num_traits::AsPrimitive<usize>,
{
    // Right binary-search: the run index is the first end > target_encoded.
    let (mut lo, mut hi) = (0usize, ends.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        let end: usize = ends[mid].as_();
        if end <= target_encoded {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo.min(ends.len().saturating_sub(1))
}

/// Helper used by encoding crates: build a [`RunEndPrimitiveProducer`] from the canonicalized
/// values + ends buffers and a logical `(offset, len)` window. Dispatches on `ends_ptype`.
pub fn build_runend_producer<T: NativePType>(
    values: PrimitiveArray,
    ends: PrimitiveArray,
    offset: usize,
    len: usize,
) -> VortexResult<Box<dyn PrimitiveChunkProducer<T>>> {
    let ends_ptype = ends.ptype();
    let values_buf = values.into_buffer::<T>();
    Ok(match_each_integer_ptype!(ends_ptype, |E| {
        Box::new(RunEndPrimitiveProducer::<T, E>::new(
            values_buf,
            ends.into_buffer::<E>(),
            offset,
            len,
        )?)
    }))
}

// ---------------------------------------------------------------------------------------
// Convenience: decode an encoded ArrayRef into a fresh `Buffer<T>` using the chunked path.
// ---------------------------------------------------------------------------------------

/// Decode `array` to a fresh `Buffer<T>` by driving the chunked producer to completion.
///
/// This is the helper used by benchmarks and tests to materialize the chunked output into
/// a comparable form against the existing executor.
pub fn decode_to_buffer<T: NativePType>(
    array: ArrayRef,
    dispatcher: &PrimitiveChunkKernelDispatcher,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>> {
    let len = array.len();
    let mut out = vortex_buffer::BufferMut::<T>::with_capacity(len);
    debug_assert!(out.spare_capacity_mut().len() >= len);
    let mut producer = build_primitive_producer::<T>(array, dispatcher, ctx)?;
    let mut scratch = Scratch::<T>::new();
    let mut written = 0usize;
    while let Some(chunk) = producer.next_chunk(&mut scratch)? {
        // SAFETY: `out` has `len` capacity; we write exactly `chunk.len()` elements per call
        // and never exceed `len` total.
        unsafe {
            let dst = out.spare_capacity_mut().as_mut_ptr().add(written).cast::<T>();
            std::ptr::copy_nonoverlapping(chunk.as_ptr(), dst, chunk.len());
        }
        written += chunk.len();
    }
    // SAFETY: we wrote exactly `written` elements into the spare capacity.
    unsafe {
        out.set_len(written);
    }
    Ok(out.freeze())
}

// ---------------------------------------------------------------------------------------
// Decode into a builder, for use by the existing builder execution path.
// ---------------------------------------------------------------------------------------

/// Decode `array` to a [`PrimitiveArray`] (non-nullable) via the chunked engine, choosing
/// the right concrete output type by inspecting the array's dtype.
///
/// Useful as a one-shot replacement for `array.execute::<PrimitiveArray>(ctx)` in code
/// paths where a non-nullable primitive output is expected and the chunked engine has a
/// fused kernel registered.
pub fn execute_to_primitive(
    array: ArrayRef,
    dispatcher: &PrimitiveChunkKernelDispatcher,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let DType::Primitive(ptype, nullability) = *array.dtype() else {
        vortex_error::vortex_bail!(
            "execute_to_primitive requires Primitive dtype, got {}",
            array.dtype()
        );
    };
    if !matches!(nullability, Nullability::NonNullable) {
        // For now, fall back to the existing executor for nullable arrays.
        return array.execute::<PrimitiveArray>(ctx);
    }
    Ok(match_each_native_ptype!(ptype, |T| {
        let buf = decode_to_buffer::<T>(array, dispatcher, ctx)?;
        PrimitiveArray::new(buf, Validity::NonNullable)
    }))
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::DictArray;

    fn ctx() -> ExecutionCtx {
        LEGACY_SESSION.create_execution_ctx()
    }

    #[test]
    fn slice_producer_round_trip() -> VortexResult<()> {
        let data = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut p = SliceProducer::<i32>::new(data.clone());
        let mut s = Scratch::<i32>::new();
        let mut out = Vec::new();
        while let Some(c) = p.next_chunk(&mut s)? {
            out.extend_from_slice(c);
        }
        assert_eq!(out, data.as_slice());
        Ok(())
    }

    #[test]
    fn dict_primitive_chunked() -> VortexResult<()> {
        // dict-encoded i32 of 4096 elements, dict size 17.
        let dict_values =
            buffer![10i32, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170];
        let codes: Vec<u8> = (0..4096).map(|i| (i % 17) as u8).collect();
        let codes_arr = PrimitiveArray::new(
            Buffer::<u8>::from_iter(codes.iter().copied()),
            Validity::NonNullable,
        );
        let values_arr = PrimitiveArray::new(dict_values.clone(), Validity::NonNullable);
        let dict = DictArray::try_new(codes_arr.into_array(), values_arr.into_array())?;

        let dispatcher = default_dispatcher();
        let result = decode_to_buffer::<i32>(dict.into_array(), &dispatcher, &mut ctx())?;
        let expected: Vec<i32> = codes
            .iter()
            .map(|c| dict_values.as_slice()[*c as usize])
            .collect();
        assert_eq!(result.as_slice(), expected.as_slice());
        Ok(())
    }

    #[test]
    fn fallback_to_canonicalize() -> VortexResult<()> {
        // Plain primitive — no kernel registered; should go through SliceProducer fallback.
        let p = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable);
        let dispatcher = default_dispatcher();
        let buf = decode_to_buffer::<i32>(p.into_array(), &dispatcher, &mut ctx())?;
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
        Ok(())
    }
}
