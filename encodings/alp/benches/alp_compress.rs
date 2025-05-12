#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng as _};
use vortex_alp::{ALPFloat, ALPRDFloat, RDEncoder, alp_encode};
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_dtype::NativePType;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, f64, f64)] = &[
    // length, fraction_patch, fraction_valid
    (1_000, 0.0, 0.25),
    (1_000, 0.01, 0.25),
    (1_000, 0.1, 0.25),
    (1_000, 0.0, 0.95),
    (1_000, 0.01, 0.95),
    (1_000, 0.1, 0.95),
    (1_000, 0.0, 1.0),
    (1_000, 0.01, 1.0),
    (1_000, 0.1, 1.0),
    (10_000, 0.0, 0.25),
    (10_000, 0.01, 0.25),
    (10_000, 0.1, 0.25),
    (10_000, 0.0, 0.95),
    (10_000, 0.01, 0.95),
    (10_000, 0.1, 0.95),
    (10_000, 0.0, 1.0),
    (10_000, 0.01, 1.0),
    (10_000, 0.1, 1.0),
];

#[divan::bench(types = [f32, f64], args = BENCH_ARGS)]
fn compress_alp<T: ALPFloat + NativePType>(bencher: Bencher, args: (usize, f64, f64)) {
    let (n, fraction_patch, fraction_valid) = args;
    let mut rng = StdRng::seed_from_u64(0);
    let mut values = buffer![T::from(1.234).unwrap(); n].into_mut();
    if fraction_patch > 0.0 {
        for index in 0..values.len() {
            if rng.random_bool(fraction_patch) {
                values[index] = T::from(1000.0).unwrap()
            }
        }
    }
    let validity = if fraction_valid < 1.0 {
        Validity::from_iter((0..values.len()).map(|_| rng.random_bool(fraction_valid)))
    } else {
        Validity::NonNullable
    };
    let values = values.freeze();

    bencher
        .with_inputs(|| (values.clone(), validity.clone()))
        .bench_values(|(values, validity)| {
            alp_encode(&PrimitiveArray::new(values, validity), None).unwrap()
        })
}

#[divan::bench(types = [f32, f64], args = BENCH_ARGS)]
fn decompress_alp<T: ALPFloat + NativePType>(bencher: Bencher, args: (usize, f64, f64)) {
    let (n, fraction_patch, fraction_valid) = args;
    let mut rng = StdRng::seed_from_u64(0);
    let mut values = buffer![T::from(1.234).unwrap(); n].into_mut();
    if fraction_patch > 0.0 {
        for index in 0..values.len() {
            if rng.random_bool(fraction_patch) {
                values[index] = T::from(1000.0).unwrap()
            }
        }
    }
    let validity = if fraction_valid < 1.0 {
        Validity::from_iter((0..values.len()).map(|_| rng.random_bool(fraction_valid)))
    } else {
        Validity::NonNullable
    };
    let values = values.freeze();
    let array = alp_encode(&PrimitiveArray::new(values, validity), None).unwrap();
    bencher
        .with_inputs(|| array.clone())
        .bench_values(|array| array.to_canonical().unwrap());
}

#[divan::bench(types = [f32, f64], args = [10_000, 100_000])]
fn compress_rd<T: ALPRDFloat>(bencher: Bencher, n: usize) {
    let primitive = PrimitiveArray::new(buffer![T::from(1.23).unwrap(); n], Validity::NonNullable);
    let encoder = RDEncoder::new(&[T::from(1.23).unwrap()]);
    bencher.bench(|| encoder.encode(&primitive));
}

#[divan::bench(types = [f32, f64], args = [10_000, 100_000])]
fn decompress_rd<T: ALPRDFloat>(bencher: Bencher, n: usize) {
    let primitive = PrimitiveArray::new(buffer![T::from(1.23).unwrap(); n], Validity::NonNullable);
    let encoder = RDEncoder::new(&[T::from(1.23).unwrap()]);
    let encoded = encoder.encode(&primitive);

    bencher
        .with_inputs(move || encoded.clone())
        .bench_values(|encoded| encoded.to_canonical().unwrap());
}
