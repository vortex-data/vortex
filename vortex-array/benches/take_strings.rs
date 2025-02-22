#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::arrays::VarBinArray;
use vortex_array::compute::take;
use vortex_array::{ArrayRef, IntoArray, ToCanonical};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Nullability};

fn main() {
    divan::main();
}

#[divan::bench]
fn varbin(bencher: Bencher) {
    let array = fixture(65_535);
    let indices = indices(1024);

    bencher
        .with_inputs(|| (&array, &indices))
        .bench_refs(|(array, indices)| take(*array, *indices).unwrap());
}

#[divan::bench]
fn varbinview(bencher: Bencher) {
    let array = fixture(65_535).to_varbinview().unwrap();
    let indices = indices(1024);

    bencher
        .with_inputs(|| (&array, &indices))
        .bench_refs(|(array, indices)| take(*array, *indices).unwrap());
}

fn fixture(len: usize) -> VarBinArray {
    VarBinArray::from_iter(
        [Some("inlined"), None, Some("verylongstring--notinlined")]
            .into_iter()
            .cycle()
            .take(len),
        DType::Utf8(Nullability::Nullable),
    )
}

// Fraction of the indices to take.
fn indices(len: usize) -> ArrayRef {
    Buffer::from_iter((0..len).filter_map(|x| (x % 2 == 0).then_some(x as u64))).into_array()
}
