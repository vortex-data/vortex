// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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
    let indices = indices(1024, 65_535);

    bencher
        .with_inputs(|| (&array, &indices))
        .bench_refs(|(array, indices)| take(array.as_ref(), indices.as_ref()).unwrap());
}

#[divan::bench]
fn varbinview(bencher: Bencher) {
    let array = fixture(65_535).to_varbinview();
    let indices = indices(1024, 65_535);

    bencher
        .with_inputs(|| (&array, &indices))
        .bench_refs(|(array, indices)| take(array.as_ref(), indices.as_ref()).unwrap());
}

#[divan::bench]
fn varbin_non_null(bencher: Bencher) {
    let array = non_null_fixutre(65_535);
    let indices = indices(1024, 65_535);

    bencher
        .with_inputs(|| (&array, &indices))
        .bench_refs(|(array, indices)| take(array.as_ref(), indices.as_ref()).unwrap());
}

#[divan::bench]
fn varbinview_non_null(bencher: Bencher) {
    let array = non_null_fixutre(65_535).to_varbinview();
    let indices = indices(1024, 65_535);

    bencher
        .with_inputs(|| (&array, &indices))
        .bench_refs(|(array, indices)| take(array.as_ref(), indices.as_ref()).unwrap());
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

fn non_null_fixutre(len: usize) -> VarBinArray {
    VarBinArray::from_iter(
        [Some("inlined"), Some("verylongstring--notinlined")]
            .into_iter()
            .cycle()
            .take(len),
        DType::Utf8(Nullability::Nullable),
    )
}

// Fraction of the indices to take.
fn indices(desired: usize, range: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    Buffer::from_iter((0..desired).map(|_| rng.random_range(0..range) as u64)).into_array()
}
