// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chunked execution engine — experimental.
//!
//! The default executor in [`crate::executor`] drives an array to a fully materialized
//! [`crate::Canonical`] one encoding at a time. Each intermediate is the size of the whole
//! array; for stacks like `Dict<RunEnd<Primitive>>` the working set is enormous, eviction
//! storms are inevitable, and any per-stage allocation runs at the rate of the *output*
//! size rather than the rate of useful work.
//!
//! This module models execution differently:
//!
//! 1. A **producer** yields decoded values into a small, driver-owned [`Scratch`] buffer
//!    sized to fit comfortably in L1d (1024 elements, 4–8 KiB for primitives).
//! 2. The driver pulls chunks until the producer is exhausted, copying each chunk into
//!    its final destination (a builder, an Arrow buffer, an aggregator, …). The scratch
//!    is reused across iterations, so the steady-state memory footprint of decode is the
//!    scratch size plus whatever fixed dictionaries the producer holds.
//! 3. Custom **chunk kernels** can be registered to fuse multiple encoding layers into a
//!    single pass — the model rule of thumb is that an encoding's chunk kernel is allowed
//!    to *materialise its own children up-front* if they are bounded in size (e.g. a
//!    dictionary's `values` slot), and then stream the unbounded `codes` chunk-by-chunk.
//!    This is the same pattern as [`crate::arrays::dict::TakeExecute`] — the fused take
//!    kernel reads `Dict.values` once and then walks `Dict.codes` chunk-by-chunk.
//!
//! See the producer traits for the contract and [`build_primitive_producer`] for dispatch.
//!
//! ## Status
//!
//! v1 spike — covers primitive output for `Dict<Primitive>`, `RunEnd<Primitive>` and the
//! fused `Dict<RunEnd<Primitive>>` stack, plus a streaming [`listview::ListChunkProducer`]
//! over `ListView<Primitive>` rows with bit-packable offsets/sizes. The module is
//! `_`-prefixed so it does not leak into the public API surface yet.

pub mod listview;
pub mod primitive;

mod scratch;

/// Re-export of the AVX2-aware take helper, hoisted into the chunked engine namespace so
/// out-of-crate kernels can call it without piercing private compute modules.
pub use crate::arrays::primitive::compute::take::take_into_uninit;

pub use scratch::Scratch;

/// Number of elements per scratch chunk.
///
/// 1024 matches the fastlanes chunk size, keeps the scratch under 8 KiB for any
/// primitive up to `u64`/`f64`, and is small enough to leave room for one fixed
/// dictionary in L1d on every CPU we care about. Empirically larger super-chunks
/// (tested at 4096) didn't move primitive workloads and regressed RunEnd at moderate
/// run lengths — the per-chunk dispatch isn't the bottleneck the profile suggested.
pub const CHUNK_LEN: usize = 1024;

/// Drive a producer to completion, invoking `sink` with each emitted chunk.
///
/// This is the canonical helper for "decode the whole thing into a downstream buffer".
/// The producer's scratch is supplied by the driver, so the same allocation is reused
/// across every chunk for the lifetime of the call.
pub fn drive_primitive<T, P, S>(
    mut producer: P,
    scratch: &mut Scratch<T>,
    mut sink: S,
) -> vortex_error::VortexResult<()>
where
    T: crate::dtype::NativePType,
    P: primitive::PrimitiveChunkProducer<T>,
    S: FnMut(&[T]),
{
    while let Some(chunk) = producer.next_chunk(scratch)? {
        sink(chunk);
    }
    Ok(())
}
