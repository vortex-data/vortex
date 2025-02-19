#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::builders::{ArrayBuilder, PrimitiveBuilder};
use vortex_array::{IntoArray, IntoCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::Nullability;
use vortex_scalar::Scalar;
use vortex_sparse::SparseArray;

fn generate_sparse_array(len: usize, sparsity: f64) -> SparseArray {
    // Choose a small number of indices from the full len to hold values.
    let indices: BufferMut<u64> = (0..len as u64)
        .filter(|_| rand::random::<f64>() < sparsity)
        .collect();

    let values = indices.clone();

    SparseArray::try_new(
        indices.into_array(),
        values.into_array(),
        len,
        Scalar::primitive(u64::MAX, Nullability::NonNullable),
    )
    .unwrap()
}

#[divan::bench(args = [0.001, 0.01, 0.05, 0.1], sample_count = 1000)]
fn into_canonical(bencher: Bencher, sparsity: f64) {
    bencher
        .with_inputs(|| generate_sparse_array(64_000, sparsity))
        .bench_values(|sparse_array| sparse_array.into_canonical().unwrap())
}

#[divan::bench(args = [0.001, 0.01, 0.05, 0.1], sample_count = 1000)]
fn canonicalize_into(bencher: Bencher, sparsity: f64) {
    bencher
        .with_inputs(|| generate_sparse_array(64_000, sparsity))
        .bench_values(|sparse_array| {
            let mut output: PrimitiveBuilder<u64> =
                PrimitiveBuilder::with_capacity(Nullability::NonNullable, 64_000);
            sparse_array.canonicalize_into(&mut output).unwrap();
            output.finish()
        })
}

fn main() {
    divan::main()
}
