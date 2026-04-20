// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::dict::dict_encode;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_compressor::builtins::integer_dictionary_encode;
use vortex_compressor::stats::IntegerStats;

fn make_array() -> PrimitiveArray {
    let values: BufferMut<i32> = (0..50).cycle().take(64_000).collect();

    let nulls = BoolArray::from_iter(
        [true, true, true, true, true, true, false]
            .into_iter()
            .cycle()
            .take(64_000),
    )
    .into_array();

    PrimitiveArray::new(values, Validity::Array(nulls))
}

#[cfg(not(codspeed))]
#[divan::bench]
fn encode_generic(bencher: Bencher) {
    let array = make_array().into_array();
    bencher
        .with_inputs(|| &array)
        .bench_refs(|array| dict_encode(array).unwrap());
}

#[cfg(not(codspeed))]
#[divan::bench]
fn encode_specialized(bencher: Bencher) {
    let array = make_array();
    let stats = IntegerStats::generate(&array);
    bencher
        .with_inputs(|| &stats)
        .bench_refs(|stats| integer_dictionary_encode(array.as_view(), stats));
}

fn main() {
    divan::main()
}
