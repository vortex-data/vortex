// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks decoding an RLE array over uncompressed children, isolating the RLE kernel from the
//! decoding of its children.
//!
//! The input has irregular runs of 6 to 18 values (12 on average), so `u8` indices need 7 bits. Indices are benchmarked at
//! `u8`, the width the compressor narrows them to when a chunk has at most 256 runs, and at
//! `u16`, the width `RLE::encode` produces; and stored as `u8` bit-packed at their width, and as
//! `u8` FastLanes Delta over 1-bit bit-packed deltas, as the compressor stores them when Delta is
//! enabled.
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
use vortex_array::dtype::NativePType;
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
    U8,
    U16,
    BitPackedU8,
    DeltaU8,
}

const INDICES: &[Indices] = &[
    Indices::U8,
    Indices::U16,
    Indices::BitPackedU8,
    Indices::DeltaU8,
];

fn bit_width(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> u8 {
    let max = array.statistics().compute_max::<u64>(ctx).unwrap();
    u8::try_from(u64::BITS - max.leading_zeros()).unwrap()
}

fn runs<T: NativePType + TryFrom<i64>>() -> Buffer<T> {
    let mut rng = StdRng::seed_from_u64(8);
    let mut values = Vec::with_capacity(LEN);
    while values.len() < LEN {
        let len = rng.random_range(6..19);
        let value = T::try_from(rng.random_range(0..100i64)).ok().unwrap();
        values.extend(std::iter::repeat_n(value, len));
    }
    values.truncate(LEN);
    Buffer::from(values)
}

fn rle_array<T: NativePType + TryFrom<i64>>(indices: Indices) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let array = PrimitiveArray::new(runs::<T>(), Validity::NonNullable);
    let rle = RLE::encode(array.as_view(), &mut ctx).unwrap();
    let rle_indices = rle
        .indices()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let rle_indices = match indices {
        Indices::U16 => rle_indices.into_array(),
        Indices::U8 => rle_indices.narrow(&mut ctx).unwrap().into_array(),
        Indices::BitPackedU8 => {
            let narrow = rle_indices.narrow(&mut ctx).unwrap();
            let width = bit_width(&narrow, &mut ctx);
            BitPacked::encode(&narrow.into_array(), width, &mut ctx)
                .unwrap()
                .into_array()
        }
        Indices::DeltaU8 => {
            let narrow = rle_indices.narrow(&mut ctx).unwrap();
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
#[divan::bench(types = [u8, i16, i32, i64], args = INDICES)]
fn decode<T: NativePType + TryFrom<i64>>(bencher: Bencher, indices: Indices) {
    let array = rle_array::<T>(indices);
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (*array).clone().execute::<PrimitiveArray>(ctx).unwrap());
}
