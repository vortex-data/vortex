// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks decoding an RLE array of `i32` values over uncompressed values, isolating the RLE
//! decode from the decoding of its values.
//!
//! The RLE array is built directly rather than by encoding a column, so its shape is fixed here and
//! not by the encoder. Each of the 64 chunks holds runs of uniform length 6 to 18 (12 on average,
//! about 85 runs per chunk), each with a random value, and the `u8` indices count the runs within
//! their chunk, so they need 7 bits. The indices are stored the two ways the compressor writes
//! them: bit-packed at their width, and, when Delta is enabled, as FastLanes Delta over 1-bit
//! bit-packed deltas. The `fastlanes` crate benchmarks the kernels themselves at every value width;
//! this covers the decoding around them.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench rle_decode`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::Delta;
use vortex_fastlanes::DeltaArraySlotsExt;
use vortex_fastlanes::FL_CHUNK_SIZE;
use vortex_fastlanes::RLE;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>();
    vortex_fastlanes::initialize(&session);
    session
});

const LEN: usize = 64 * 1024;

#[derive(Clone, Copy, Debug)]
enum Indices {
    BitPackedU8,
    DeltaU8,
}

const INDICES: &[Indices] = &[Indices::BitPackedU8, Indices::DeltaU8];

fn bit_width(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> u8 {
    let max = array.statistics().compute_max::<u64>(ctx).unwrap();
    u8::try_from(u64::BITS - max.leading_zeros()).unwrap()
}

/// The parts of an RLE array of `LEN` values: per-chunk run values, the `u8` index of each value's
/// run within its chunk, and the offset of each chunk's first run value.
fn rle_parts() -> (Buffer<i32>, Buffer<u8>, Buffer<u64>) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut values = BufferMut::with_capacity(LEN / 4);
    let mut indices = BufferMut::with_capacity(LEN);
    let mut offsets = BufferMut::with_capacity(LEN / FL_CHUNK_SIZE);
    for _ in 0..LEN / FL_CHUNK_SIZE {
        offsets.push(values.len() as u64);
        let mut filled = 0;
        let mut run = 0usize;
        while filled < FL_CHUNK_SIZE {
            let len = rng.random_range(6..=18).min(FL_CHUNK_SIZE - filled);
            values.push(rng.random::<i32>());
            indices.extend(std::iter::repeat_n(u8::try_from(run).unwrap(), len));
            filled += len;
            run += 1;
        }
    }
    (values.freeze(), indices.freeze(), offsets.freeze())
}

fn rle_array(layout: Indices) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let (values, indices, offsets) = rle_parts();
    let expected: Buffer<i32> = indices
        .iter()
        .enumerate()
        .map(|(i, &run)| {
            values[usize::try_from(offsets[i / FL_CHUNK_SIZE]).unwrap() + usize::from(run)]
        })
        .collect();
    let indices = PrimitiveArray::new(indices, Validity::NonNullable);
    let indices = match layout {
        Indices::BitPackedU8 => {
            let width = bit_width(&indices, &mut ctx);
            assert_eq!(width, 7);
            BitPacked::encode(&indices.into_array(), width, &mut ctx)
                .unwrap()
                .into_array()
        }
        Indices::DeltaU8 => {
            let delta = Delta::try_from_primitive_array(&indices, &mut ctx).unwrap();
            let deltas = delta
                .deltas()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)
                .unwrap();
            assert_eq!(bit_width(&deltas, &mut ctx), 1);
            let deltas = BitPacked::encode(&deltas.into_array(), 1, &mut ctx).unwrap();
            Delta::try_new(
                delta.bases().clone(),
                deltas.into_array(),
                Validity::NonNullable,
                0,
                LEN,
            )
            .unwrap()
            .into_array()
        }
    };
    let array = RLE::try_new(
        PrimitiveArray::new(values, Validity::NonNullable).into_array(),
        indices,
        offsets.into_array(),
        0,
        LEN,
    )
    .unwrap()
    .into_array();
    let decoded = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
    assert_eq!(decoded.as_slice::<i32>(), expected.as_slice());
    array
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = INDICES)]
fn decode(bencher: Bencher, layout: Indices) {
    let array = rle_array(layout);
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (*array).clone().execute::<PrimitiveArray>(ctx).unwrap());
}
