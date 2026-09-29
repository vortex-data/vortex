// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare decoding a FoR array with one reference against one with a reference per 1024-element
//! chunk, over both a primitive child and a BitPacked child, which decode through a fused unpack.
//!
//! Every chunk spans the same range of values, so both encodings pack at the same bit width and
//! the difference between them is the cost of the per-chunk references.
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

const LENS: &[usize] = &[1024, 16 * 1024];

/// Enough bits for the values' spread above their minimum.
const BIT_WIDTH: u8 = 7;

fn values<T: NativePType + TryFrom<usize>>(len: usize) -> Buffer<T> {
    (0..len)
        .map(|i| T::try_from(1000 + (i * 7919) % 100).ok().unwrap())
        .collect()
}

fn for_array<T: NativePType + TryFrom<usize>>(
    len: usize,
    chunked: bool,
    bitpacked: bool,
) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let array = PrimitiveArray::new(values::<T>(len), Validity::NonNullable);
    let for_array = if chunked {
        FoR::encode_chunked(array, &mut ctx).unwrap()
    } else {
        FoR::encode(array, &mut ctx).unwrap()
    };
    if !bitpacked {
        return for_array.into_array();
    }
    let packed = BitPacked::encode(for_array.encoded(), BIT_WIDTH, &mut ctx).unwrap();
    FoR::try_new_chunked(packed.into_array(), for_array.references().clone(), 0)
        .unwrap()
        .into_array()
}

fn run<T: NativePType + TryFrom<usize>>(
    bencher: Bencher,
    len: usize,
    chunked: bool,
    bitpacked: bool,
) {
    let array = for_array::<T>(len, chunked, bitpacked);
    bencher
        .counter(ItemsCount::new(len))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (*array).clone().execute::<PrimitiveArray>(ctx).unwrap());
}

#[divan::bench(types = [u32, i64], args = LENS)]
fn decode<T: NativePType + TryFrom<usize>>(bencher: Bencher, len: usize) {
    run::<T>(bencher, len, false, false);
}

#[divan::bench(types = [u32, i64], args = LENS)]
fn decode_chunked<T: NativePType + TryFrom<usize>>(bencher: Bencher, len: usize) {
    run::<T>(bencher, len, true, false);
}

#[divan::bench(types = [u32, i64], args = LENS)]
fn decode_bitpacked<T: NativePType + TryFrom<usize>>(bencher: Bencher, len: usize) {
    run::<T>(bencher, len, false, true);
}

#[divan::bench(types = [u32, i64], args = LENS)]
fn decode_bitpacked_chunked<T: NativePType + TryFrom<usize>>(bencher: Bencher, len: usize) {
    run::<T>(bencher, len, true, true);
}
