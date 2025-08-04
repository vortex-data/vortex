// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(unexpected_cfgs)]

use arrow_buffer::BooleanBuffer;
use bitvec::order::Msb0;
use bitvec::vec::BitVec;
use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex::experiment::array::Array;
use vortex::experiment::buffers::ByteBufferHandle;
use vortex::experiment::encodings::bitpacked::BitPackedEncoding;
use vortex::{IntoArray, ToCanonical};
use vortex_array::compute::filter;
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_fastlanes::bitpack_to_best_bit_width;
use vortex_mask::Mask;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub fn main() {
    divan::main();
}

// #[divan::bench(types = [i8, i16, i32, i64], args = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])]
#[divan::bench(types = [i8, i16, i32, i64], args = [0.005, 0.01, 0.0105, 0.02, 0.03, 0.04, 0.05])]
pub fn decompress_bitpacking_early_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..100_000)
        .map(|_| T::from(rng.random_range(0..100)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .to_primitive()
        .unwrap();
    let array = bitpack_to_best_bit_width(&values).unwrap();

    let mask = (0..100_000)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BooleanBuffer>();
    let mask = &Mask::from_buffer(mask);

    bencher.bench(|| {
        filter(array.as_ref(), mask)
            .unwrap()
            .to_canonical()
            .unwrap()
    });
}

// #[divan::bench(types = [i8, i16, i32, i64], args = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])]
#[divan::bench(types = [i8, i16, i32, i64], args = [0.005, 0.01, 0.0105, 0.02, 0.03, 0.04, 0.05])]
pub fn decompress_bitpacking_late_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..100_000)
        .map(|_| T::from(rng.random_range(0..100)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .to_primitive()
        .unwrap();

    let array = bitpack_to_best_bit_width(&values).unwrap();

    let mask = (0..100_000)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BooleanBuffer>();
    let mask = &Mask::from_buffer(mask);

    bencher
        .with_inputs(|| array.clone())
        .bench_values(|array| filter(array.to_canonical().unwrap().as_ref(), mask).unwrap());
}

// #[divan::bench(types = [i8, i16, i32, i64], args = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])]
#[divan::bench(types = [i8, i16, i32, i64], args = [0.005, 0.01, 0.0105, 0.02, 0.03, 0.04, 0.05])]
pub fn decompress_bitpacking_fused_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..100_000)
        .map(|_| T::from(rng.random_range(0..100)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .to_primitive()
        .unwrap();
    let array = bitpack_to_best_bit_width(&values).unwrap();

    // Create a V2 array.
    let enc = BitPackedEncoding::new(
        array.bit_width() as usize,
        ByteBufferHandle::new(array.packed().clone()),
    );
    let array2 = Array::new(
        array.len(),
        array.dtype().clone(),
        array.statistics().to_owned(),
        Box::new(enc),
    );

    let mask = (0..100_000)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BitVec<u64, Msb0>>();

    bencher.bench_local(|| array2.to_canonical(&mask).unwrap());
}
