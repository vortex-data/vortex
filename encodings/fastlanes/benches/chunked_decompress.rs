// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures where the executor's stream-to-canonical shortcut pays: streaming through
//! `decompress_chunks` against level-wise execution, for Filter over BitPacked and for nested FoR
//! stacks of growing depth. `vortex/benches/streaming_decompress.rs` compares streaming against
//! hand-fused kernels per encoding tree.

use std::ops::Range;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::chunk_iter::ChunkMut;
use vortex_array::chunk_iter::ChunkSink;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_fastlanes::FoR;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

const LEN: usize = 4 * 1024 * 1024;

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

/// Canonicalize `array`, either by streaming chunks into the builder or by level-wise execution.
/// Driving `execute_via_chunks` directly measures the mechanism itself, independent of the
/// executor's depth heuristic.
fn bench_execute(bencher: Bencher, array: ArrayRef, streaming: bool) {
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            if streaming {
                vortex_array::chunk_iter::execute_via_chunks(&array, &mut ctx)
                    .vortex_expect("bench")
                    .len()
            } else {
                vortex_array::chunk_iter::without_chunked_execute(|| {
                    array.execute::<PrimitiveArray>(&mut ctx)
                })
                .vortex_expect("bench")
                .len()
            }
        });
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
            sink.total
        });
}

#[divan::bench(args = FILTER_KEEP)]
fn filter_bp_sum_execute_then_read(bencher: Bencher, keep: usize) {
    let array = make_filter_bitpacked(keep);
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let primitive = vortex_array::chunk_iter::without_chunked_execute(|| {
                array.execute::<PrimitiveArray>(&mut ctx)
            })
            .vortex_expect("bench");
            primitive
                .as_slice::<u32>()
                .iter()
                .map(|&v| v as u64)
                .fold(0, u64::wrapping_add)
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

/// `depth` nested FoR levels over one BitPacked leaf of `len` rows.
fn make_for_stack(len: usize, depth: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let deltas = PrimitiveArray::from_iter(
        (0..len as u64).map(|i| u32::try_from((i * 7) % 1000).vortex_expect("fits")),
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

const STACK_DEPTHS: &[usize] = &[1, 2, 3, 4, 8];

/// A scan split, whose output stays in cache, and a large array, whose output does not.
const STACK_LENS: [usize; 2] = [SPLIT_LEN, LEN];

#[divan::bench(consts = STACK_LENS, args = STACK_DEPTHS)]
fn execute_for_stack_streaming<const N: usize>(bencher: Bencher, depth: usize) {
    bench_execute(bencher, make_for_stack(N, depth), true);
}

#[divan::bench(consts = STACK_LENS, args = STACK_DEPTHS)]
fn execute_for_stack_levelwise<const N: usize>(bencher: Bencher, depth: usize) {
    bench_execute(bencher, make_for_stack(N, depth), false);
}
