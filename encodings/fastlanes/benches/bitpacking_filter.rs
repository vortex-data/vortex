// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks filtering bit-packed arrays, both directly and as the elements of a
//! `FixedSizeList<i32>`, where the list-level selection expands into runs of selected elements.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPackedData;
use vortex_mask::Mask;
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

const NUM_ELEMENTS: usize = 1 << 20;
const BIT_WIDTH: u8 = 16;
const DENSITIES: &[f64] = &[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.9];

fn bitpacked_i32(len: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..len)
        .map(|_| rng.random_range(0..(1i32 << BIT_WIDTH)))
        .collect::<BufferMut<i32>>();
    let array = PrimitiveArray::new(values, Validity::NonNullable).into_array();
    BitPackedData::encode(&array, BIT_WIDTH, &mut SESSION.create_execution_ctx())
        .unwrap()
        .into_array()
}

fn random_bits(len: usize, density: f64) -> BitBuffer {
    let mut rng = StdRng::seed_from_u64(1);
    BitBuffer::from_iter((0..len).map(|_| rng.random_bool(density)))
}

fn bench_filter(bencher: Bencher, array: &ArrayRef, bits: &BitBuffer) {
    // Build a fresh mask for every iteration so that representations cached on the mask, such as
    // its indices, are not reused across iterations.
    bencher
        .with_inputs(|| {
            (
                array,
                Mask::from_buffer(bits.clone()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mask, mut ctx)| {
            array
                .filter(mask)
                .unwrap()
                .execute::<RecursiveCanonical>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench(args = DENSITIES)]
fn primitive_i32(bencher: Bencher, density: f64) {
    let array = bitpacked_i32(NUM_ELEMENTS);
    bench_filter(bencher, &array, &random_bits(NUM_ELEMENTS, density));
}

#[divan::bench(consts = [2, 4, 16, 64], args = DENSITIES)]
fn fsl_i32<const LIST_SIZE: u32>(bencher: Bencher, density: f64) {
    let len = NUM_ELEMENTS / LIST_SIZE as usize;
    let elements = bitpacked_i32(NUM_ELEMENTS);
    let array =
        FixedSizeListArray::new(elements, LIST_SIZE, Validity::NonNullable, len).into_array();
    bench_filter(bencher, &array, &random_bits(len, density));
}
