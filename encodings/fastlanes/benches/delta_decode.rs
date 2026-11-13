// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks decoding and reading a Delta array.
//!
//! - `decode_bitpacked_deltas` decodes a Delta array whose deltas are bit-packed, and
//!   `decode_primitive_deltas` one whose deltas are a primitive array.
//! - The probe benchmarks read single rows, through a one-off probe for each row or through one
//!   repeated probe.
//! - `filter` selects one row in `stride`, and `take` takes scattered rows.
//!
//! Only `decode_bitpacked_deltas` carries `#[cpu_features]`, so it is measured on each walltime
//! CPU-feature leg: the build decides which transpose and decode kernels it runs. The others are
//! measured once, in simulation.
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
use vortex_mask::Mask;
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
fn decode_bitpacked_deltas<T: NativePType + TryFrom<usize>>(bencher: Bencher, bytes: usize) {
    let len = bytes / size_of::<T>();
    let array = delta_bitpacked::<T>(len).into_array();
    bencher
        .counter(ItemsCount::new(len))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (*array).clone().execute::<PrimitiveArray>(ctx).unwrap());
}

#[divan::bench(args = INPUT_BYTES)]
fn decode_primitive_deltas(bencher: Bencher, bytes: usize) {
    let len = bytes / size_of::<u32>();
    let mut ctx = SESSION.create_execution_ctx();
    let array = PrimitiveArray::new(values::<u32>(len), Validity::NonNullable);
    let array = Delta::try_from_primitive_array(&array, &mut ctx)
        .unwrap()
        .into_array();
    bencher
        .counter(ItemsCount::new(len))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (*array).clone().execute::<PrimitiveArray>(ctx).unwrap());
}

/// Rows read by the probe benchmarks: the first four chunks, in order.
const PROBE_ROWS: usize = 4096;

/// Reads each row with a one-off probe.
#[divan::bench]
fn probe_once(bencher: Bencher) {
    let array = delta_bitpacked::<u32>(PROBE_ROWS).into_array();
    bencher
        .counter(ItemsCount::new(PROBE_ROWS))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            for index in 0..PROBE_ROWS {
                divan::black_box(array.probe().execute_scalar(index, ctx).unwrap());
            }
        });
}

/// Reads each row through one repeated probe.
#[divan::bench]
fn probe_repeated(bencher: Bencher) {
    let array = delta_bitpacked::<u32>(PROBE_ROWS).into_array();
    bencher
        .counter(ItemsCount::new(PROBE_ROWS))
        .with_inputs(|| (array.repeated_probe(), SESSION.create_execution_ctx()))
        .bench_refs(|(probe, ctx)| {
            for index in 0..PROBE_ROWS {
                divan::black_box(probe.execute_scalar(index, ctx).unwrap());
            }
        });
}

/// A u32 Delta array of `INPUT_BYTES` over bit-packed deltas.
fn delta_bitpacked_u32() -> ArrayRef {
    delta_bitpacked::<u32>(INPUT_BYTES[0] / size_of::<u32>()).into_array()
}

/// A mask that selects one row in `stride`.
fn strided_mask(len: usize, stride: usize) -> Mask {
    Mask::from_indices(len, (0..len).step_by(stride))
}

/// Filters the Delta array.
#[divan::bench(args = [64, 4096])]
fn filter(bencher: Bencher, stride: usize) {
    let array = delta_bitpacked_u32();
    let mask = strided_mask(array.len(), stride);
    bencher
        .counter(ItemsCount::new(array.len()))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            (*array)
                .clone()
                .filter(mask.clone())
                .unwrap()
                .execute::<PrimitiveArray>(ctx)
                .unwrap()
        });
}

/// `count` indices scattered over `len` rows, in no particular order.
fn scattered_indices(len: usize, count: usize) -> ArrayRef {
    PrimitiveArray::from_iter((0..count).map(|i| u32::try_from((i * 7919 + 13) % len).unwrap()))
        .into_array()
}

/// Takes from the Delta array.
#[divan::bench(args = [1024])]
fn take(bencher: Bencher, count: usize) {
    let array = delta_bitpacked_u32();
    let indices = scattered_indices(array.len(), count);
    bencher
        .counter(ItemsCount::new(count))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            (*array)
                .clone()
                .take(indices.clone())
                .unwrap()
                .execute::<PrimitiveArray>(ctx)
                .unwrap()
        });
}
