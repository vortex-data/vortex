// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::BoolArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;

fn main() {
    divan::main();
}

const INPUT_SIZE: &[usize] = &[1024, 16_384, 65_536, 262_144];

fn make_bool_array(n: usize) -> BoolArray {
    BoolArray::new(
        BitBuffer::from_iter((0..n).map(|i| i % 3 == 0)),
        Validity::from_iter((0..n).map(|i| i % 7 != 0)),
    )
}

#[divan::bench(args = INPUT_SIZE)]
fn bool_fill_null_true(bencher: Bencher, n: usize) {
    let arr = make_bool_array(n);
    let fill = Scalar::from(true);
    bencher
        .with_inputs(|| (arr.clone().into_array(), fill.clone()))
        .bench_values(|(a, f)| a.fill_null(f).unwrap().to_canonical().unwrap());
}

#[divan::bench(args = INPUT_SIZE)]
fn bool_fill_null_false(bencher: Bencher, n: usize) {
    let arr = make_bool_array(n);
    let fill = Scalar::from(false);
    bencher
        .with_inputs(|| (arr.clone().into_array(), fill.clone()))
        .bench_values(|(a, f)| a.fill_null(f).unwrap().to_canonical().unwrap());
}
