// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sinks for fused chunked execution.
//!
//! A sink consumes [`crate::_chunked_exec::primitive::PrimitiveChunkProducer`] chunks one
//! at a time and produces a final result. Combining a producer with a sink fuses decode
//! and the downstream operator into a single pass, with no `Buffer<T>` materialised
//! between them. The producer's L1-resident scratch flows directly into the operator's
//! register-resident accumulators / output buffer.
//!
//! See [`drive_into_sink`] for the canonical driver.
//!
//! ## Sinks shipped here
//!
//! - [`BufferSink`]: equivalent to `decode_to_buffer`. Collects chunks into a `Buffer<T>`.
//!   Useful as the "no fusion" baseline against which other sinks compare.
//! - [`SumSink`]: accumulates `sum(x) :: i64` across chunks. No output buffer at all.
//! - [`MapSink`]: applies a per-element `FnMut(T) -> U`, writing into a `Buffer<U>`. Used
//!   to express casts (`T as U`) and unary scalar functions (`x + c`, `x * c`, …) without
//!   materialising the source as a `Buffer<T>` first.
//! - [`FilterSink`]: applies a per-element predicate; surviving elements stream into a
//!   `Buffer<T>`. The mask is never materialised — selectivity is encoded directly in
//!   the output length.

use std::marker::PhantomData;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use super::Scratch;
use super::primitive::PrimitiveChunkKernelDispatcher;
use super::primitive::build_primitive_producer;
use crate::ArrayRef;
use crate::dtype::NativePType;
use crate::executor::ExecutionCtx;

/// Consumes chunks of decoded primitive values and produces a final result.
///
/// Implementors should treat each `push` call as "process this chunk in-place" — the
/// slice borrowed by `chunk` is only valid until the next `push` call, since it points
/// into the producer's scratch buffer.
pub trait PrimitiveChunkSink<T: NativePType> {
    /// The value produced once all chunks have been consumed.
    type Output;

    /// Process one chunk. Implementors typically aggregate, transform, or selectively
    /// store into an owned output buffer.
    fn push(&mut self, chunk: &[T]) -> VortexResult<()>;

    /// Finalise the sink and produce the result. Consumes the sink.
    fn finish(self) -> VortexResult<Self::Output>;
}

/// Drive a chunked producer to completion, feeding every chunk into `sink`.
///
/// This is the fused-pipeline entry point: the producer never materialises a full
/// `Buffer<T>`, the sink decides what to do with each chunk, and the only memory traffic
/// outside the input bytes is whatever the sink chooses to keep.
pub fn drive_into_sink<T, S>(
    array: ArrayRef,
    dispatcher: &PrimitiveChunkKernelDispatcher,
    mut sink: S,
    ctx: &mut ExecutionCtx,
) -> VortexResult<S::Output>
where
    T: NativePType,
    S: PrimitiveChunkSink<T>,
{
    let mut producer = build_primitive_producer::<T>(array, dispatcher, ctx)?;
    let mut scratch = Scratch::<T>::new();
    while let Some(chunk) = producer.next_chunk(&mut scratch)? {
        sink.push(chunk)?;
    }
    sink.finish()
}

// ----------------------------------------------------------------------------------
// BufferSink — the "no fusion" baseline. Collects everything into a Buffer<T>.
// ----------------------------------------------------------------------------------

/// Collects chunks into a flat [`Buffer<T>`]. Equivalent to the old `decode_to_buffer`,
/// reformulated as a sink for use with [`drive_into_sink`].
pub struct BufferSink<T: NativePType> {
    out: BufferMut<T>,
}

impl<T: NativePType> BufferSink<T> {
    /// Construct with a pre-allocated capacity of `len` elements.
    pub fn with_capacity(len: usize) -> Self {
        Self {
            out: BufferMut::<T>::with_capacity(len),
        }
    }
}

impl<T: NativePType> PrimitiveChunkSink<T> for BufferSink<T> {
    type Output = Buffer<T>;

    fn push(&mut self, chunk: &[T]) -> VortexResult<()> {
        // SAFETY: BufferMut maintains capacity invariant; we never exceed its allocation.
        // The caller pre-allocated with `with_capacity(array.len())`.
        let n = chunk.len();
        let written = self.out.as_slice().len();
        unsafe {
            let dst = self
                .out
                .spare_capacity_mut()
                .as_mut_ptr()
                .add(0)
                .cast::<T>();
            std::ptr::copy_nonoverlapping(chunk.as_ptr(), dst, n);
            self.out.set_len(written + n);
        }
        Ok(())
    }

    fn finish(self) -> VortexResult<Self::Output> {
        Ok(self.out.freeze())
    }
}

// ----------------------------------------------------------------------------------
// SumSink — proof of concept. No output buffer; just an i64 accumulator.
// ----------------------------------------------------------------------------------

/// Accumulates `sum(chunk_i) as i64` across all chunks. Produces a scalar.
///
/// Useful as the smallest possible demonstration of operator fusion: the only memory
/// touched outside the input bytes is the producer's 4 KiB scratch.
pub struct SumSink<T: NativePType> {
    acc: i64,
    _marker: PhantomData<fn() -> T>,
}

impl<T: NativePType> SumSink<T> {
    /// New sink, zero-initialised.
    pub fn new() -> Self {
        Self {
            acc: 0,
            _marker: PhantomData,
        }
    }
}

impl<T: NativePType> Default for SumSink<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> PrimitiveChunkSink<T> for SumSink<T>
where
    T: NativePType + num_traits::AsPrimitive<i64>,
{
    type Output = i64;

    fn push(&mut self, chunk: &[T]) -> VortexResult<()> {
        let mut acc = self.acc;
        for &v in chunk {
            acc = acc.wrapping_add(v.as_());
        }
        self.acc = acc;
        Ok(())
    }

    fn finish(self) -> VortexResult<Self::Output> {
        Ok(self.acc)
    }
}

// ----------------------------------------------------------------------------------
// MapSink — generic per-element transform. Used for cast + scalar funcs.
// ----------------------------------------------------------------------------------

/// Applies a per-element `FnMut(T) -> U` and writes the result to a [`Buffer<U>`].
///
/// Casts (e.g. `T as U`) and unary scalar functions (`x + c`, `x * c`, `x.abs()`, …) are
/// the same shape — one element in, one element out. The fused win over canonical is that
/// the source array is never materialised as a `Buffer<T>` between decode and map.
pub struct MapSink<T, U, F>
where
    T: NativePType,
    U: NativePType,
    F: FnMut(T) -> U,
{
    f: F,
    out: BufferMut<U>,
    _marker: PhantomData<fn(T)>,
}

impl<T, U, F> MapSink<T, U, F>
where
    T: NativePType,
    U: NativePType,
    F: FnMut(T) -> U,
{
    /// Construct with a pre-allocated output capacity of `len` elements.
    pub fn with_capacity(len: usize, f: F) -> Self {
        Self {
            f,
            out: BufferMut::<U>::with_capacity(len),
            _marker: PhantomData,
        }
    }
}

impl<T, U, F> PrimitiveChunkSink<T> for MapSink<T, U, F>
where
    T: NativePType,
    U: NativePType,
    F: FnMut(T) -> U,
{
    type Output = Buffer<U>;

    fn push(&mut self, chunk: &[T]) -> VortexResult<()> {
        let written = self.out.as_slice().len();
        let n = chunk.len();
        // SAFETY: caller-pre-allocated capacity ≥ total array length; we never exceed it.
        unsafe {
            let dst = self.out.spare_capacity_mut().as_mut_ptr().cast::<U>();
            for (i, &v) in chunk.iter().enumerate() {
                dst.add(i).write((self.f)(v));
            }
            self.out.set_len(written + n);
        }
        Ok(())
    }

    fn finish(self) -> VortexResult<Self::Output> {
        Ok(self.out.freeze())
    }
}

// ----------------------------------------------------------------------------------
// FilterSink — per-element predicate, surviving elements stream to a Buffer<T>.
// ----------------------------------------------------------------------------------

/// Applies a per-element predicate. Surviving elements stream directly into the output
/// buffer — the boolean mask is never materialised.
///
/// Canonical two-pass `decode → mask → filter` becomes a single pass that touches the
/// compressed input once and writes only the elements that survive.
pub struct FilterSink<T, P>
where
    T: NativePType,
    P: FnMut(T) -> bool,
{
    pred: P,
    out: BufferMut<T>,
}

impl<T, P> FilterSink<T, P>
where
    T: NativePType,
    P: FnMut(T) -> bool,
{
    /// Construct with a pre-allocated output capacity of `len` elements (the worst-case
    /// upper bound when every element passes the predicate).
    pub fn with_capacity(len: usize, pred: P) -> Self {
        Self {
            pred,
            out: BufferMut::<T>::with_capacity(len),
        }
    }
}

impl<T, P> PrimitiveChunkSink<T> for FilterSink<T, P>
where
    T: NativePType,
    P: FnMut(T) -> bool,
{
    type Output = Buffer<T>;

    fn push(&mut self, chunk: &[T]) -> VortexResult<()> {
        // SAFETY: BufferMut::with_capacity guaranteed at construction; never push past it.
        let mut written = self.out.as_slice().len();
        let dst = self.out.spare_capacity_mut();
        let dst_ptr = dst.as_mut_ptr().cast::<T>();
        // Track how many cells we've written into the spare-capacity region this call.
        let mut local = 0usize;
        for &v in chunk {
            if (self.pred)(v) {
                // SAFETY: total written ≤ pre-allocated capacity.
                unsafe { dst_ptr.add(local).write(v) };
                local += 1;
            }
        }
        written += local;
        // SAFETY: just wrote exactly `local` elements past the previous len.
        unsafe { self.out.set_len(written) };
        Ok(())
    }

    fn finish(self) -> VortexResult<Self::Output> {
        Ok(self.out.freeze())
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::_chunked_exec::primitive::default_dispatcher;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::validity::Validity;

    fn ctx() -> ExecutionCtx {
        LEGACY_SESSION.create_execution_ctx()
    }

    fn make_dict_i32(codes: &[u32], dict: &[i32]) -> ArrayRef {
        let dict_arr = PrimitiveArray::new(
            Buffer::<i32>::from_iter(dict.iter().copied()),
            Validity::NonNullable,
        );
        let codes_arr = PrimitiveArray::new(
            Buffer::<u32>::from_iter(codes.iter().copied()),
            Validity::NonNullable,
        );
        DictArray::try_new(codes_arr.into_array(), dict_arr.into_array())
            .unwrap()
            .into_array()
    }

    #[test]
    fn buffer_sink_round_trip() -> VortexResult<()> {
        let array = make_dict_i32(&[0, 1, 0, 2, 1], &[10, 20, 30]);
        let dispatcher = default_dispatcher();
        let buf: Buffer<i32> = drive_into_sink(
            array,
            &dispatcher,
            BufferSink::<i32>::with_capacity(5),
            &mut ctx(),
        )?;
        assert_eq!(buf.as_slice(), &[10, 20, 10, 30, 20]);
        Ok(())
    }

    #[test]
    fn sum_sink_fused() -> VortexResult<()> {
        let array = make_dict_i32(&[0, 1, 0, 2, 1, 2, 2, 0], &[10, 20, 30]);
        // Expected: 10+20+10+30+20+30+30+10 = 160.
        let dispatcher = default_dispatcher();
        let s: i64 = drive_into_sink(array, &dispatcher, SumSink::<i32>::new(), &mut ctx())?;
        assert_eq!(s, 160);
        Ok(())
    }

    #[test]
    fn map_sink_cast_i32_to_i64() -> VortexResult<()> {
        let array = make_dict_i32(&[0, 1, 2], &[100, 200, 300]);
        let dispatcher = default_dispatcher();
        let buf: Buffer<i64> = drive_into_sink(
            array,
            &dispatcher,
            MapSink::<i32, i64, _>::with_capacity(3, |x| x as i64),
            &mut ctx(),
        )?;
        assert_eq!(buf.as_slice(), &[100i64, 200, 300]);
        Ok(())
    }

    #[test]
    fn map_sink_scalar_add() -> VortexResult<()> {
        let array = make_dict_i32(&[0, 1, 0, 2], &[1, 2, 3]);
        let dispatcher = default_dispatcher();
        let buf: Buffer<i32> = drive_into_sink(
            array,
            &dispatcher,
            MapSink::<i32, i32, _>::with_capacity(4, |x| x + 100),
            &mut ctx(),
        )?;
        assert_eq!(buf.as_slice(), &[101, 102, 101, 103]);
        Ok(())
    }

    #[test]
    fn filter_sink_keeps_surviving() -> VortexResult<()> {
        let array = make_dict_i32(&[0, 1, 2, 1, 0, 2], &[5, 15, 25]);
        // Source values: [5, 15, 25, 15, 5, 25]. Predicate: > 10.
        let dispatcher = default_dispatcher();
        let buf: Buffer<i32> = drive_into_sink(
            array,
            &dispatcher,
            FilterSink::<i32, _>::with_capacity(6, |x| x > 10),
            &mut ctx(),
        )?;
        assert_eq!(buf.as_slice(), &[15, 25, 15, 25]);
        Ok(())
    }

    #[test]
    fn sink_works_on_canonical_primitive() -> VortexResult<()> {
        // Sanity check: even plain PrimitiveArray flows through the SliceProducer
        // fallback path and into the sink.
        let p = PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5], Validity::NonNullable);
        let dispatcher = default_dispatcher();
        let s = drive_into_sink(p.into_array(), &dispatcher, SumSink::<i32>::new(), &mut ctx())?;
        assert_eq!(s, 15);
        Ok(())
    }
}
