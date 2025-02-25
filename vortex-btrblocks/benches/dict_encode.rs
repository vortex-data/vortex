#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::Array;
use vortex_array::arrays::{BoolArray, PrimitiveArray};
use vortex_array::validity::Validity;
use vortex_btrblocks::CompressorStats;
use vortex_btrblocks::integer::IntegerStats;
use vortex_btrblocks::integer::dictionary::dictionary_encode;
use vortex_buffer::BufferMut;
use vortex_dict::builders::dict_encode;

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

#[divan::bench]
fn encode_generic(bencher: Bencher) {
    bencher
        .with_inputs(|| make_array().into_array())
        .bench_values(|array| dict_encode(&array).unwrap());
}

#[divan::bench]
fn encode_specialized(bencher: Bencher) {
    bencher
        .with_inputs(|| IntegerStats::generate(&make_array()))
        .bench_values(|stats| dictionary_encode(&stats).unwrap());
}

fn main() {
    divan::main()
}
