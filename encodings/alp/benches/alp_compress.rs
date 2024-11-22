#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_alp::{ALPFloat, ALPRDFloat, Exponents, RDEncoder};
use vortex_array::array::PrimitiveArray;
use vortex_array::IntoCanonical;

fn main() {
    divan::main();
}

#[divan::bench(types = [f32, f64], args = [100_000, 10_000_000])]
fn compress_alp<T: ALPFloat>(n: usize) -> (Exponents, Vec<T::ALPInt>, Vec<u64>, Vec<T>) {
    let values: Vec<T> = vec![T::from(1.234).unwrap(); n];
    T::encode(values.as_slice(), None)
}

#[divan::bench(types = [f32, f64], args = [100_000, 10_000_000])]
fn decompress_alp<T: ALPFloat>(bencher: Bencher, n: usize) {
    let values: Vec<T> = vec![T::from(1.234).unwrap(); n];
    let (exponents, encoded, ..) = T::encode(values.as_slice(), None);
    bencher.bench_local(move || T::decode(&encoded, exponents));
}

#[divan::bench(types = [f32, f64], args = [100_000, 10_000_000])]
fn compress_rd<T: ALPRDFloat>(bencher: Bencher, n: usize) {
    let values: Vec<T> = vec![T::from(1.23).unwrap(); n];
    let primitive = PrimitiveArray::from(values);
    let encoder = RDEncoder::new(&[T::from(1.23).unwrap()]);

    bencher.bench_local(|| encoder.encode(&primitive));
}

#[divan::bench(types = [f32, f64], args = [100_000, 1_000_000, 10_000_000])]
fn decompress_rd<T: ALPRDFloat>(bencher: Bencher, n: usize) {
    let values: Vec<T> = vec![T::from(1.23).unwrap(); n];
    let primitive = PrimitiveArray::from(values);
    let encoder = RDEncoder::new(&[T::from(1.23).unwrap()]);
    let encoded = encoder.encode(&primitive);

    bencher
        .with_inputs(move || encoded.clone())
        .bench_local_values(|encoded| encoded.into_canonical().unwrap());
}
