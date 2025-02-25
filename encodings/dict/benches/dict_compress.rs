#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::distr::{Distribution, StandardUniform};
use vortex_array::Array;
use vortex_array::arrays::{PrimitiveArray, VarBinArray, VarBinViewArray};
use vortex_array::validity::Validity::NonNullable;
use vortex_buffer::buffer;
use vortex_dict::DictArray;
use vortex_dict::builders::dict_encode;
use vortex_dict::test::{gen_primitive_for_dict, gen_varbin_words};
use vortex_dtype::NativePType;
use vortex_runend::RunEndArray;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, usize)] = &[
    // length, unique_values
    (1_000, 2),
    (1_000, 4),
    (1_000, 8),
    (1_000, 32),
    (1_000, 128),
    (1_000, 512),
    (10_000, 2),
    (10_000, 4),
    (10_000, 8),
    (10_000, 32),
    (10_000, 128),
    (10_000, 512),
];

#[divan::bench(types = [u8, f32, i64], args = BENCH_ARGS)]
fn encode_primitives<T>(bencher: Bencher, (len, unique_values): (usize, usize))
where
    T: NativePType,
    StandardUniform: Distribution<T>,
{
    let primitive_arr = gen_primitive_for_dict::<T>(len, unique_values);

    bencher
        .with_inputs(|| primitive_arr.clone())
        .bench_refs(|arr| dict_encode(arr));
}

#[divan::bench(args = BENCH_ARGS)]
fn encode_varbin(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let varbin_arr = VarBinArray::from(gen_varbin_words(len, unique_values));

    bencher
        .with_inputs(|| varbin_arr.clone())
        .bench_refs(|arr| dict_encode(arr));
}

#[divan::bench(args = BENCH_ARGS)]
fn encode_varbinview(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(len, unique_values));

    bencher
        .with_inputs(|| varbinview_arr.clone())
        .bench_refs(|arr| dict_encode(arr));
}

#[divan::bench(types = [u8, f32, i64], args = BENCH_ARGS)]
fn decode_primitives<T>(bencher: Bencher, (len, unique_values): (usize, usize))
where
    T: NativePType,
    StandardUniform: Distribution<T>,
{
    let primitive_arr = gen_primitive_for_dict::<T>(len, unique_values);
    let dict = dict_encode(&primitive_arr).unwrap();

    bencher
        .with_inputs(|| dict.clone())
        .bench_values(|dict| dict.to_canonical());
}

#[divan::bench(args = BENCH_ARGS)]
fn decode_varbin(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let varbin_arr = VarBinArray::from(gen_varbin_words(len, unique_values));
    let dict = dict_encode(&varbin_arr).unwrap();

    bencher
        .with_inputs(|| dict.clone())
        .bench_values(|dict| dict.to_canonical());
}

#[divan::bench(args = BENCH_ARGS)]
fn decode_varbinview(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(len, unique_values));
    let dict = dict_encode(&varbinview_arr).unwrap();

    bencher
        .with_inputs(|| dict.clone())
        .bench_values(|dict| dict.to_canonical());
}

#[divan::bench(args = &[
    100,
    1000,
    10_000,
    100_000,
])]
fn decode_dict_with_runend_codes(bencher: Bencher, length: usize) {
    let ends = PrimitiveArray::new(
        buffer![(length / 3) as u32, (length / 2) as u32, length as u32],
        NonNullable,
    )
    .into_array();

    let codes = PrimitiveArray::new(buffer![0u32, 1, 2], NonNullable).into_array();
    let runend_codes = RunEndArray::try_new(ends, codes).unwrap();
    let dict_values = PrimitiveArray::new(buffer![100u32, 200, 300], NonNullable).into_array();
    let dict_array = DictArray::try_new(runend_codes.into_array(), dict_values).unwrap();

    bencher
        .with_inputs(|| dict_array.clone())
        .bench_refs(|array| array.to_canonical())
}
