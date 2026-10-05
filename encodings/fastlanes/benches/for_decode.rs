// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks decoding a FoR array with a reference per 1024-element chunk, over a primitive child
//! and over a BitPacked child. An unsigned BitPacked child decodes through a fused unpack, and a
//! signed one through BitPacked decoding followed by adding the references in place.
//!
//! Every benchmark carries `#[cpu_features]`, so it is measured on each walltime CPU-feature leg
//! rather than in simulation: the loops under test are auto-vectorized, so the build decides
//! their speed.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench for_decode`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArraySlotsExt;
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

/// Bytes of values per input array.
const INPUT_BYTES: &[usize] = &[512 * 1024];

/// Enough bits for the values' spread above their minimum.
const BIT_WIDTH: u8 = 7;

fn values<T: NativePType + TryFrom<usize>>(len: usize) -> Buffer<T> {
    (0..len)
        .map(|i| T::try_from(1000 + (i * 7919) % 100).ok().unwrap())
        .collect()
}

fn for_array<T: NativePType + TryFrom<usize>>(len: usize, bitpacked: bool) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let array = PrimitiveArray::new(values::<T>(len), Validity::NonNullable);
    let for_array = FoR::encode_chunked(array, &mut ctx).unwrap();
    if !bitpacked {
        return for_array.into_array();
    }
    let packed = BitPacked::encode(for_array.encoded(), BIT_WIDTH, &mut ctx).unwrap();
    FoR::try_new_chunked(packed.into_array(), for_array.references().clone(), 0)
        .unwrap()
        .into_array()
}

fn run<T: NativePType + TryFrom<usize>>(bencher: Bencher, bytes: usize, bitpacked: bool) {
    let len = bytes / size_of::<T>();
    let array = for_array::<T>(len, bitpacked);
    bencher
        .counter(ItemsCount::new(len))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (*array).clone().execute::<PrimitiveArray>(ctx).unwrap());
}

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [i64], args = INPUT_BYTES)]
fn decode_chunked<T: NativePType + TryFrom<usize>>(bencher: Bencher, bytes: usize) {
    run::<T>(bencher, bytes, false);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [u32, i64], args = INPUT_BYTES)]
fn decode_bitpacked_chunked<T: NativePType + TryFrom<usize>>(bencher: Bencher, bytes: usize) {
    run::<T>(bencher, bytes, true);
}
