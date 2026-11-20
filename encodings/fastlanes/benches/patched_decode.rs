// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decompression of bit-packed integers with exceptions, comparing patches stored inside the
//! BitPacked array against a Patched array wrapping a patchless BitPacked array, and comparisons
//! of each against a constant.

#![expect(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitWidths;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
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

const LEN: usize = if cfg!(codspeed) { 1 << 16 } else { 1 << 20 };
const BIT_WIDTH: u8 = 9;
/// One exception every `STRIDE` values.
const STRIDES: &[usize] = &[1000, 100, 20];

/// Values that fit `BIT_WIDTH` bits, except one in every `stride`.
fn data(stride: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LEN as u32).map(|i| {
        if i as usize % stride == stride / 2 {
            u32::MAX - i
        } else {
            i % (1 << BIT_WIDTH)
        }
    }))
}

/// The data bit-packed with its exceptions as interior patches.
fn interior(stride: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    bitpack_encode(&data(stride), BIT_WIDTH, None, &mut ctx)
        .unwrap()
        .into_array()
}

/// The data as a Patched array over a patchless bit-packed array.
fn patched(stride: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let packed = bitpack_encode(&data(stride), BIT_WIDTH, None, &mut ctx).unwrap();
    let mut parts = BitPacked::into_parts(packed);
    let patches = parts.patches.take().unwrap();
    let BitWidths::Global(bit_width) = parts.bit_widths else {
        unreachable!()
    };
    let inner = BitPacked::try_new(
        parts.packed,
        vortex_array::dtype::PType::U32,
        parts.validity,
        None,
        bit_width,
        parts.len,
        parts.offset,
    )
    .unwrap();
    Patched::from_array_and_patches(inner.into_array(), &patches, &mut ctx)
        .unwrap()
        .into_array()
}

#[divan::bench(args = STRIDES)]
fn decode_interior_patches(bencher: Bencher, stride: usize) {
    let array = interior(stride);
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| array.clone().execute::<PrimitiveArray>(ctx).unwrap());
}

#[divan::bench(args = STRIDES)]
fn decode_patched_array(bencher: Bencher, stride: usize) {
    let array = patched(stride);
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| array.clone().execute::<PrimitiveArray>(ctx).unwrap());
}

fn compare(array: &ArrayRef) -> ArrayRef {
    array
        .binary(
            ConstantArray::new(7u32, array.len()).into_array(),
            Operator::Eq,
        )
        .unwrap()
}

#[divan::bench(args = STRIDES)]
fn compare_interior_patches(bencher: Bencher, stride: usize) {
    let array = interior(stride);
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| compare(array).execute::<BoolArray>(ctx).unwrap());
}

#[divan::bench(args = STRIDES)]
fn compare_patched_array(bencher: Bencher, stride: usize) {
    let array = patched(stride);
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| compare(array).execute::<BoolArray>(ctx).unwrap());
}
