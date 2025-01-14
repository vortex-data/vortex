#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng as _};
use vortex_alp::{alp_encode, ALPFloat, ALPRDFloat, RDEncoder};
use vortex_array::array::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::IntoCanonical;
use vortex_buffer::buffer;
use vortex_dtype::NativePType;

fn main() {
    divan::main();
}

#[divan::bench(types = [f32, f64], args = [
    (100_000, 1.0),
    (10_000_000, 1.0),
    (100_000, 0.25),
    (10_000_000, 0.25),
    (100_000, 0.95),
    (10_000_000, 0.95),
])]
fn compress_alp<T: ALPFloat + NativePType>(bencher: Bencher, args: (usize, f64)) -> () {
    let (n, fraction_valid) = args;
    let mut rng = StdRng::seed_from_u64(0);
    let values = buffer![T::from(1.234).unwrap(); n];
    let validity = if fraction_valid < 1.0 {
        Validity::from_iter((0..values.len()).map(|_| rng.gen_bool(fraction_valid)))
    } else {
        Validity::NonNullable
    };
    bencher.bench_local(move || {
        alp_encode(&PrimitiveArray::new(values.clone(), validity.clone())).unwrap()
    })
}

#[divan::bench(types = [f32, f64], args = [
    (100_000, 1.0),
    (10_000_000, 1.0),
    (100_000, 0.25),
    (10_000_000, 0.25),
    (100_000, 0.95),
    (10_000_000, 0.95),
])]
fn decompress_alp<T: ALPFloat + NativePType>(bencher: Bencher, args: (usize, f64)) {
    let (n, fraction_valid) = args;
    let mut rng = StdRng::seed_from_u64(0);
    let values = buffer![T::from(1.234).unwrap(); n];
    let validity = if fraction_valid < 1.0 {
        Validity::from_iter((0..values.len()).map(|_| rng.gen_bool(fraction_valid)))
    } else {
        Validity::NonNullable
    };
    let array = alp_encode(&PrimitiveArray::new(values, validity)).unwrap();
    bencher.bench_local(move || array.clone().into_canonical().unwrap());
}

#[divan::bench(types = [f32, f64], args = [100_000, 10_000_000])]
fn compress_rd<T: ALPRDFloat>(bencher: Bencher, n: usize) {
    let primitive = PrimitiveArray::new(buffer![T::from(1.23).unwrap(); n], Validity::NonNullable);
    let encoder = RDEncoder::new(&[T::from(1.23).unwrap()]);

    bencher.bench_local(|| encoder.encode(&primitive));
}

#[divan::bench(types = [f32, f64], args = [100_000, 1_000_000, 10_000_000])]
fn decompress_rd<T: ALPRDFloat>(bencher: Bencher, n: usize) {
    let primitive = PrimitiveArray::new(buffer![T::from(1.23).unwrap(); n], Validity::NonNullable);
    let encoder = RDEncoder::new(&[T::from(1.23).unwrap()]);
    let encoded = encoder.encode(&primitive);

    bencher
        .with_inputs(move || encoded.clone())
        .bench_local_values(|encoded| encoded.into_canonical().unwrap());
}
