// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks decoding a Delta array over a BitPacked child. The fused decode unpacks and
//! prefix-sums each chunk in one pass, while the unfused decode first unpacks the whole child into
//! a primitive array and then undoes the deltas, as Delta did before the fused kernel. Both
//! untranspose each chunk once.
//!
//! Every benchmark carries `#[cpu_features]`, so it is measured on each walltime CPU-feature leg
//! rather than in simulation: the loops under test are auto-vectorized, so the build decides
//! their speed.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench delta_decode`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::Delta;
use vortex_fastlanes::DeltaArray;
use vortex_fastlanes::DeltaArraySlotsExt;
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

/// Bytes of values per input array.
const INPUT_BYTES: &[usize] = &[512 * 1024];

/// Enough bits for the deltas between consecutive values.
const BIT_WIDTH: u8 = 4;

/// Increasing values whose consecutive deltas are below 16, wrapping around at the width of `T`.
fn values<T: NativePType + TryFrom<usize>>(len: usize) -> Buffer<T> {
    let mask = usize::MAX >> (usize::BITS as usize - 8 * size_of::<T>());
    (0..len)
        .scan(0usize, |acc, i| {
            *acc = acc.wrapping_add((i * 7919) % 16);
            Some(T::try_from(*acc & mask).ok().unwrap())
        })
        .collect()
}

fn delta_bitpacked<T: NativePType + TryFrom<usize>>(len: usize) -> DeltaArray {
    let mut ctx = SESSION.create_execution_ctx();
    let array = PrimitiveArray::new(values::<T>(len), Validity::NonNullable);
    let delta = Delta::try_from_primitive_array(&array, &mut ctx).unwrap();
    let packed = BitPacked::encode(delta.deltas(), BIT_WIDTH, &mut ctx).unwrap();
    Delta::try_new(
        delta.bases().clone(),
        packed.into_array(),
        Validity::NonNullable,
        0,
        len,
    )
    .unwrap()
}

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [u16, u32, u64], args = INPUT_BYTES)]
fn decode_fused<T: NativePType + TryFrom<usize>>(bencher: Bencher, bytes: usize) {
    let len = bytes / size_of::<T>();
    let array = delta_bitpacked::<T>(len).into_array();
    bencher
        .counter(ItemsCount::new(len))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (*array).clone().execute::<PrimitiveArray>(ctx).unwrap());
}

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [u16, u32, u64], args = INPUT_BYTES)]
fn decode_unfused<T: NativePType + TryFrom<usize>>(bencher: Bencher, bytes: usize) {
    let len = bytes / size_of::<T>();
    let array = delta_bitpacked::<T>(len);
    bencher
        .counter(ItemsCount::new(len))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            let deltas: ArrayRef = array
                .deltas()
                .clone()
                .execute::<PrimitiveArray>(ctx)
                .unwrap()
                .into_array();
            Delta::try_new(
                array.bases().clone(),
                deltas,
                Validity::NonNullable,
                0,
                array.len(),
            )
            .unwrap()
            .into_array()
            .execute::<PrimitiveArray>(ctx)
            .unwrap()
        });
}
