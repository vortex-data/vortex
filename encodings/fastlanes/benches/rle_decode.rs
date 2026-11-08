// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks decoding an RLE array over uncompressed children, isolating the RLE kernel from the
//! decoding of its children.
//!
//! The input has irregular runs of 1 to 15 values (8 on average). Indices are benchmarked at
//! `u8`, the width the compressor narrows them to when a chunk has at most 256 runs, and at
//! `u16`, the width `RLE::encode` produces.
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
}

const INDICES: &[Indices] = &[Indices::U8, Indices::U16];

fn runs<T: NativePType + TryFrom<i64>>() -> Buffer<T> {
    let mut rng = StdRng::seed_from_u64(8);
    let mut values = Vec::with_capacity(LEN);
    while values.len() < LEN {
        let len = rng.random_range(1..16);
        let value = T::try_from(rng.random_range(0..1i64 << 30)).ok().unwrap();
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
        Indices::U8 => rle_indices.narrow(&mut ctx).unwrap(),
        Indices::U16 => rle_indices,
    };
    RLE::try_new(
        rle.values().clone(),
        rle_indices.into_array(),
        rle.values_idx_offsets().clone(),
        rle.offset(),
        rle.len(),
    )
    .unwrap()
    .into_array()
}

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [i32, i64], args = INDICES)]
fn decode<T: NativePType + TryFrom<i64>>(bencher: Bencher, indices: Indices) {
    let array = rle_array::<T>(indices);
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (*array).clone().execute::<PrimitiveArray>(ctx).unwrap());
}
