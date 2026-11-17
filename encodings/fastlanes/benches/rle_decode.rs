// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks decoding an RLE array of `i32` values over uncompressed children, isolating the RLE
//! decode from the decoding of its children.
//!
//! The input has irregular runs of 6 to 18 values (12 on average). Its indices are stored the two
//! ways the compressor writes them: `u8` bit-packed at their width, and, when Delta is enabled, `u8`
//! FastLanes Delta over 1-bit bit-packed deltas. The `fastlanes` crate benchmarks the kernels
//! themselves at every value width; this covers the decoding around them.
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
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::Delta;
use vortex_fastlanes::DeltaArraySlotsExt;
use vortex_fastlanes::RLE;
use vortex_fastlanes::RLEArrayExt;
use vortex_fastlanes::RLEArraySlotsExt;
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

fn runs() -> Buffer<i32> {
    let mut rng = StdRng::seed_from_u64(8);
    let mut values = Vec::with_capacity(LEN);
    while values.len() < LEN {
        let len = rng.random_range(6..19);
        let value = rng.random_range(0..100);
        values.extend(std::iter::repeat_n(value, len));
    }
    values.truncate(LEN);
    Buffer::from(values)
}

fn rle_array(indices: Indices) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let array = PrimitiveArray::new(runs(), Validity::NonNullable);
    let rle = RLE::encode(array.as_view(), &mut ctx).unwrap();
    let rle_indices = rle
        .indices()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let narrow = rle_indices.narrow(&mut ctx).unwrap();
    let rle_indices = match indices {
        Indices::BitPackedU8 => {
            let width = bit_width(&narrow, &mut ctx);
            BitPacked::encode(&narrow.into_array(), width, &mut ctx)
                .unwrap()
                .into_array()
        }
        Indices::DeltaU8 => {
            let delta = Delta::try_from_primitive_array(&narrow, &mut ctx).unwrap();
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
                narrow.len(),
            )
            .unwrap()
            .into_array()
        }
    };
    RLE::try_new(
        rle.values().clone(),
        rle_indices,
        rle.values_idx_offsets().clone(),
        rle.offset(),
        rle.len(),
    )
    .unwrap()
    .into_array()
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = INDICES)]
fn decode(bencher: Bencher, indices: Indices) {
    let array = rle_array(indices);
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (*array).clone().execute::<PrimitiveArray>(ctx).unwrap());
}
