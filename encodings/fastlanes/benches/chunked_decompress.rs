// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures the composable chunked-decompression vtable path (`decompress_chunks`) for
//! FoR-over-BitPacked against:
//!
//! - `two_pass`: full fused decompression to a `PrimitiveArray` followed by a second pass over
//!   the materialized buffer (the pattern `decompress_chunks` is designed to replace).
//! - `hand_fused`: a monomorphized loop over `BitUnpackedChunks` with the reference folded in
//!   algebraically — the no-dispatch upper bound, equivalent to the fused `unpack_map` kernels.
//!
//! The gap between `chunked_vtable` and `hand_fused` is the exact price of the generic
//! mechanism: one virtual call per 1024-element chunk per encoding level plus one in-place
//! add pass over the L1-resident chunk at the FoR level.

// The `#[gat(Item)]` import expansion trips unused_imports even though the lending iterator
// machinery requires it.
#![allow(unused_imports)]

use std::hint::black_box;
use std::ops::Range;
use std::sync::LazyLock;

use divan::Bencher;
use lending_iterator::gat;
use lending_iterator::prelude::Item;
#[gat(Item)]
use lending_iterator::prelude::LendingIterator;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::chunk_iter::ChunkMut;
use vortex_array::chunk_iter::ChunkSink;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::unpack_iter::BitUnpackedChunks;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

const LEN: usize = 4 * 1024 * 1024;
const REFERENCE: u32 = 1_000_000;

/// FoR(reference=1M) over BitPacked(bit_width=10), no patches.
fn make_for_bitpacked() -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let deltas = PrimitiveArray::from_iter(
        (0..LEN as u64).map(|i| u32::try_from((i * 7) % 1000).vortex_expect("fits")),
    );
    let bp = bitpack_encode(&deltas, 10, None, &mut ctx).vortex_expect("bench");
    FoR::try_new(bp.into_array(), Scalar::from(REFERENCE))
        .vortex_expect("bench")
        .into_array()
}

struct SumSink {
    total: u64,
}

impl ChunkSink for SumSink {
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, _row_range: Range<usize>) -> VortexResult<()> {
        self.total = self
            .total
            .wrapping_add(chunk.as_slice::<u32>().iter().map(|&v| v as u64).sum());
        Ok(())
    }
}

/// Sum through the composable vtable path. With an unsigned reference over BitPacked, FoR takes
/// its fused streaming path: each block is unpacked with the reference folded into the kernel
/// and handed straight to the sink — one streaming pass, no materialization, no extra add pass.
#[divan::bench]
fn chunked_vtable_sum(bencher: Bencher) {
    let array = make_for_bitpacked();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let mut sink = SumSink { total: 0 };
            array
                .decompress_chunks(&mut ctx, &mut sink)
                .vortex_expect("bench");
            black_box(sink.total)
        });
}

/// Same as `chunked_vtable_sum` but with a signed FoR reference, which skips the fused
/// FoR+BitPacked dispatch and exercises the generic composition: BitPacked streams plain
/// unpacked blocks and FoR's sink adapter adds the reference in place per chunk.
#[divan::bench]
fn chunked_vtable_sum_generic_compose(bencher: Bencher) {
    let mut ctx = SESSION.create_execution_ctx();
    let deltas = PrimitiveArray::from_iter(
        (0..LEN as u64).map(|i| i32::try_from((i * 7) % 1000).vortex_expect("fits")),
    );
    let bp = bitpack_encode(&deltas, 10, None, &mut ctx).vortex_expect("bench");
    let array = FoR::try_new(bp.into_array(), Scalar::from(-1_000_000i32))
        .vortex_expect("bench")
        .into_array();

    struct SumSinkI32 {
        total: i64,
    }
    impl ChunkSink for SumSinkI32 {
        #[inline]
        fn accept(&mut self, chunk: ChunkMut<'_>, _row_range: Range<usize>) -> VortexResult<()> {
            self.total = self
                .total
                .wrapping_add(chunk.as_slice::<i32>().iter().map(|&v| v as i64).sum());
            Ok(())
        }
    }

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let mut sink = SumSinkI32 { total: 0 };
            array
                .decompress_chunks(&mut ctx, &mut sink)
                .vortex_expect("bench");
            black_box(sink.total)
        });
}

/// Sum by fully decompressing (the fused FoR+BitPacked execute path) and then re-reading the
/// materialized array: two passes over `LEN` values.
#[divan::bench]
fn two_pass_sum(bencher: Bencher) {
    let array = make_for_bitpacked();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let primitive = array
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("bench");
            let total: u64 = primitive
                .as_slice::<u32>()
                .iter()
                .map(|&v| v as u64)
                .fold(0, u64::wrapping_add);
            black_box(total)
        });
}

/// Monomorphized upper bound: iterate the BitPacked child's unpacked blocks directly and fold
/// the FoR reference in algebraically — no dynamic dispatch, no extra in-place pass.
#[divan::bench]
fn hand_fused_sum(bencher: Bencher) {
    let array = make_for_bitpacked();
    bencher.with_inputs(|| array.clone()).bench_values(|array| {
        let for_ = array.as_::<FoR>();
        let bp = for_.encoded().as_::<vortex_fastlanes::BitPacked>();
        let mut chunks: BitUnpackedChunks<u32> = bp.unpacked_chunks().vortex_expect("bench");
        let mut total = 0u64;
        let mut count = 0u64;
        if let Some(initial) = chunks.initial() {
            total = total.wrapping_add(initial.iter().map(|&v| v as u64).sum());
            count += initial.len() as u64;
        }
        let mut iter = chunks.full_chunks();
        while let Some(chunk) = iter.next() {
            total = total.wrapping_add(chunk.iter().map(|&v| v as u64).sum());
            count += chunk.len() as u64;
        }
        if let Some(trailer) = chunks.trailer() {
            total = total.wrapping_add(trailer.iter().map(|&v| v as u64).sum());
            count += trailer.len() as u64;
        }
        black_box(total.wrapping_add(count * REFERENCE as u64))
    });
}

/// Monomorphized version of exactly what the vtable path does per chunk (unpack, in-place
/// reference add, then fold) but with zero dynamic dispatch. The gap between this and
/// `hand_fused_sum` is the cost of the extra in-place pass; the gap between this and
/// `chunked_vtable_sum` is the pure dynamic-dispatch cost (two virtual calls per 1024 values).
#[divan::bench]
fn hand_chunked_add_pass_sum(bencher: Bencher) {
    let array = make_for_bitpacked();
    bencher.with_inputs(|| array.clone()).bench_values(|array| {
        let for_ = array.as_::<FoR>();
        let bp = for_.encoded().as_::<vortex_fastlanes::BitPacked>();
        let mut chunks: BitUnpackedChunks<u32> = bp.unpacked_chunks().vortex_expect("bench");
        let mut total = 0u64;
        let mut fold = |chunk: &mut [u32]| {
            for v in chunk.iter_mut() {
                *v = v.wrapping_add(REFERENCE);
            }
            total = total.wrapping_add(chunk.iter().map(|&v| v as u64).sum());
        };
        if let Some(initial) = chunks.initial() {
            fold(initial);
        }
        let mut iter = chunks.full_chunks();
        while let Some(chunk) = iter.next() {
            fold(chunk);
        }
        if let Some(trailer) = chunks.trailer() {
            fold(trailer);
        }
        black_box(total)
    });
}

struct WriteSink<'a> {
    out: &'a mut Vec<u32>,
}

impl ChunkSink for WriteSink<'_> {
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, _row_range: Range<usize>) -> VortexResult<()> {
        self.out.extend_from_slice(chunk.as_slice::<u32>());
        Ok(())
    }
}

/// Materialize through the chunked path (unpack to scratch, add reference in place, copy out) —
/// upper-bounds the cost of the non-fused composition when the consumer wants a full buffer.
#[divan::bench]
fn chunked_vtable_decompress_into(bencher: Bencher) {
    let array = make_for_bitpacked();
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                SESSION.create_execution_ctx(),
                Vec::<u32>::with_capacity(LEN),
            )
        })
        .bench_values(|(array, mut ctx, mut out)| {
            let mut sink = WriteSink { out: &mut out };
            array
                .decompress_chunks(&mut ctx, &mut sink)
                .vortex_expect("bench");
            black_box(out.len())
        });
}

/// Materialize through the fused execute path (`decode_into` writes unpacked+shifted values
/// straight into the output buffer) — the specialized baseline for full decompression.
#[divan::bench]
fn fused_decompress(bencher: Bencher) {
    let array = make_for_bitpacked();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let primitive = array
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("bench");
            black_box(primitive.len())
        });
}

/// Patched(Constant): the base never materializes — Constant re-emits one L1-resident scratch
/// chunk and Patched's sink adapter overwrites the sparse patch rows per chunk.
fn make_patched_constant() -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let inner = vortex_array::arrays::ConstantArray::new(42u32, LEN).into_array();
    let n_patches = LEN / 1000;
    let patches = vortex_array::patches::Patches::new(
        LEN,
        0,
        PrimitiveArray::from_iter((0..n_patches as u64).map(|i| i * 1000)).into_array(),
        PrimitiveArray::from_iter(
            (0..n_patches as u64).map(|i| u32::try_from(i % 90_000).vortex_expect("fits")),
        )
        .into_array(),
        None,
    )
    .vortex_expect("bench");
    vortex_array::arrays::Patched::from_array_and_patches(inner, &patches, &mut ctx)
        .vortex_expect("bench")
        .into_array()
}

#[divan::bench]
fn chunked_vtable_sum_patched_constant(bencher: Bencher) {
    let array = make_patched_constant();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let mut sink = SumSink { total: 0 };
            array
                .decompress_chunks(&mut ctx, &mut sink)
                .vortex_expect("bench");
            black_box(sink.total)
        });
}

#[divan::bench]
fn two_pass_sum_patched_constant(bencher: Bencher) {
    let array = make_patched_constant();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let primitive = array
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("bench");
            let total: u64 = primitive
                .as_slice::<u32>()
                .iter()
                .map(|&v| v as u64)
                .fold(0, u64::wrapping_add);
            black_box(total)
        });
}

/// Sparse decompression baseline: `execute` on Patched(Constant) canonicalizes the constant into
/// a full-length buffer, then scatters the sparse patches over it — one full-buffer write pass
/// plus sparse writes, materializing the array.
#[divan::bench]
fn sparse_decompress_patched_constant(bencher: Bencher) {
    let array = make_patched_constant();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let primitive = array
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("bench");
            black_box(primitive.len())
        });
}

/// Materialize Patched(Constant) through the chunked path: splat the constant into an 8KB
/// scratch, patch it, and copy each chunk out to the destination buffer.
#[divan::bench]
fn chunked_decompress_into_patched_constant(bencher: Bencher) {
    let array = make_patched_constant();
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                SESSION.create_execution_ctx(),
                Vec::<u32>::with_capacity(LEN),
            )
        })
        .bench_values(|(array, mut ctx, mut out)| {
            let mut sink = WriteSink { out: &mut out };
            array
                .decompress_chunks(&mut ctx, &mut sink)
                .vortex_expect("bench");
            black_box(out.len())
        });
}

// ---------------------------------------------------------------------------------------------
// Executor integration: execute::<PrimitiveArray> with the stream-to-canonical shortcut
// (execute_until step 2c) enabled vs disabled, on realistic multi-level trees.
// ---------------------------------------------------------------------------------------------

/// FoR(signed reference) over BitPacked: the common shape for signed integers. The level-wise
/// executor unpacks the full buffer, then does a second full-buffer wrapping-add pass; the
/// streaming shortcut does the add in L1 per block and writes the output once.
fn make_signed_for_bitpacked() -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let deltas = PrimitiveArray::from_iter(
        (0..LEN as u64).map(|i| i32::try_from((i * 7) % 1000).vortex_expect("fits")),
    );
    let bp = bitpack_encode(&deltas, 10, None, &mut ctx).vortex_expect("bench");
    FoR::try_new(bp.into_array(), Scalar::from(-1_000_000i32))
        .vortex_expect("bench")
        .into_array()
}

/// Patched(FoR(BitPacked)): G-ALP-style tree — bitpacked base, frame-of-reference shift, and
/// sparse exceptions patched at the top.
fn make_patched_for_bitpacked() -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let inner = make_for_bitpacked();
    let n_patches = LEN / 1000;
    let patches = vortex_array::patches::Patches::new(
        LEN,
        0,
        PrimitiveArray::from_iter((0..n_patches as u64).map(|i| i * 1000)).into_array(),
        PrimitiveArray::from_iter(
            (0..n_patches as u64)
                .map(|i| u32::try_from(2_000_000 + i % 90_000).vortex_expect("fits")),
        )
        .into_array(),
        None,
    )
    .vortex_expect("bench");
    vortex_array::arrays::Patched::from_array_and_patches(inner, &patches, &mut ctx)
        .vortex_expect("bench")
        .into_array()
}

/// Canonicalize `array`, either by streaming chunks into the builder or by level-wise execution.
/// Driving `execute_via_chunks` directly measures the mechanism itself, independent of the
/// executor's depth heuristic.
fn bench_execute(bencher: Bencher, array: ArrayRef, streaming: bool) {
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let len = if streaming {
                vortex_array::chunk_iter::execute_via_chunks(&array, &mut ctx)
                    .vortex_expect("bench")
                    .len()
            } else {
                vortex_array::chunk_iter::set_chunked_execute_enabled(false);
                let result = array
                    .execute::<PrimitiveArray>(&mut ctx)
                    .vortex_expect("bench");
                vortex_array::chunk_iter::set_chunked_execute_enabled(true);
                result.len()
            };
            black_box(len)
        });
}

#[divan::bench]
fn execute_signed_for_bp_streaming(bencher: Bencher) {
    bench_execute(bencher, make_signed_for_bitpacked(), true);
}

#[divan::bench]
fn execute_signed_for_bp_levelwise(bencher: Bencher) {
    bench_execute(bencher, make_signed_for_bitpacked(), false);
}

#[divan::bench]
fn execute_patched_for_bp_streaming(bencher: Bencher) {
    bench_execute(bencher, make_patched_for_bitpacked(), true);
}

#[divan::bench]
fn execute_patched_for_bp_levelwise(bencher: Bencher) {
    bench_execute(bencher, make_patched_for_bitpacked(), false);
}

#[divan::bench]
fn execute_fused_for_bp_streaming(bencher: Bencher) {
    bench_execute(bencher, make_for_bitpacked(), true);
}

#[divan::bench]
fn execute_fused_for_bp_levelwise(bencher: Bencher) {
    bench_execute(bencher, make_for_bitpacked(), false);
}

// ---------------------------------------------------------------------------------------------
// Filter(BitPacked): the dominant TPC-H scan tree (736x at 65,536 rows in the Q1/Q6 trace).
// Compares the streaming path (unpack a block, compact it in L1, write survivors once) against
// level-wise execution (materialize the full 64K child, then run the compaction kernel over it).
// ---------------------------------------------------------------------------------------------

const SPLIT_LEN: usize = 65_536;

/// Filter over BitPacked keeping `keep` rows out of every 16 (i.e. selectivity `keep/16`),
/// giving the mask realistic run structure rather than alternating single rows.
fn make_filter_bitpacked(keep: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter(
        (0..SPLIT_LEN as u64).map(|i| u32::try_from((i * 7) % 1000).vortex_expect("fits")),
    );
    let bp = bitpack_encode(&values, 10, None, &mut ctx)
        .vortex_expect("bench")
        .into_array();
    const PERIOD: usize = 16;
    assert!(keep < PERIOD, "mask must filter some rows out");
    let mask = vortex_mask::Mask::from_iter((0..SPLIT_LEN).map(|i| (i % PERIOD) < keep));
    bp.filter(mask).vortex_expect("bench")
}

/// `keep` rows out of every 16.
const FILTER_KEEP: &[usize] = &[1, 4, 8, 12];

#[divan::bench(args = FILTER_KEEP)]
fn execute_filter_bp_streaming(bencher: Bencher, keep: usize) {
    bench_execute(bencher, make_filter_bitpacked(keep), true);
}

#[divan::bench(args = FILTER_KEEP)]
fn execute_filter_bp_levelwise(bencher: Bencher, keep: usize) {
    bench_execute(bencher, make_filter_bitpacked(keep), false);
}

/// Consumption (not materialization): sum a Filter(BitPacked) tree. Streaming compacts each
/// block in L1 and folds it directly; the baseline canonicalizes the filtered array first and
/// then reads it back.
#[divan::bench(args = FILTER_KEEP)]
fn filter_bp_sum_streaming(bencher: Bencher, keep: usize) {
    let array = make_filter_bitpacked(keep);
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let mut sink = SumSink { total: 0 };
            array
                .decompress_chunks(&mut ctx, &mut sink)
                .vortex_expect("bench");
            black_box(sink.total)
        });
}

#[divan::bench(args = FILTER_KEEP)]
fn filter_bp_sum_execute_then_read(bencher: Bencher, keep: usize) {
    let array = make_filter_bitpacked(keep);
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            vortex_array::chunk_iter::set_chunked_execute_enabled(false);
            let primitive = array
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("bench");
            vortex_array::chunk_iter::set_chunked_execute_enabled(true);
            let total: u64 = primitive
                .as_slice::<u32>()
                .iter()
                .map(|&v| v as u64)
                .fold(0, u64::wrapping_add);
            black_box(total)
        });
}

// ---------------------------------------------------------------------------------------------
// Depth sweep: does the streaming win grow with stack depth?
//
// Each level above the innermost fused FoR+BitPacked pair is a generic composition step. For
// level-wise execution that is one extra *full-buffer* read+write pass over every value; for
// streaming it is one extra pass over an L1-resident block. If the model is right, the gap
// should widen roughly linearly with depth.
// ---------------------------------------------------------------------------------------------

/// `depth` nested FoR levels over one BitPacked leaf.
fn make_for_stack(depth: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let deltas = PrimitiveArray::from_iter(
        (0..LEN as u64).map(|i| u32::try_from((i * 7) % 1000).vortex_expect("fits")),
    );
    let mut array = bitpack_encode(&deltas, 10, None, &mut ctx)
        .vortex_expect("bench")
        .into_array();
    for _ in 0..depth {
        array = FoR::try_new(array, Scalar::from(1_000u32))
            .vortex_expect("bench")
            .into_array();
    }
    array
}

const STACK_DEPTHS: &[usize] = &[1, 2, 4, 8];

#[divan::bench(args = STACK_DEPTHS)]
fn execute_for_stack_streaming(bencher: Bencher, depth: usize) {
    bench_execute(bencher, make_for_stack(depth), true);
}

#[divan::bench(args = STACK_DEPTHS)]
fn execute_for_stack_levelwise(bencher: Bencher, depth: usize) {
    bench_execute(bencher, make_for_stack(depth), false);
}
