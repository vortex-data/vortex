#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::distr::{Distribution, StandardUniform};
use vortex_array::arrays::{VarBinArray, VarBinViewArray};
use vortex_dict::builders::dict_encode;
use vortex_dict::test::{gen_primitive_for_dict, gen_varbin_words};
use vortex_dtype::NativePType;

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
        .bench_refs(|arr| dict_encode(arr.as_ref()));
}

#[divan::bench(args = BENCH_ARGS)]
fn encode_varbin(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let varbin_arr = VarBinArray::from(gen_varbin_words(len, unique_values));

    bencher
        .with_inputs(|| varbin_arr.clone())
        .bench_refs(|arr| dict_encode(arr.as_ref()));
}

#[divan::bench(args = BENCH_ARGS)]
fn encode_varbinview(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(len, unique_values));

    bencher
        .with_inputs(|| varbinview_arr.clone())
        .bench_refs(|arr| dict_encode(arr.as_ref()));
}

#[divan::bench(types = [u8, f32, i64], args = BENCH_ARGS)]
fn decode_primitives<T>(bencher: Bencher, (len, unique_values): (usize, usize))
where
    T: NativePType,
    StandardUniform: Distribution<T>,
{
    let primitive_arr = gen_primitive_for_dict::<T>(len, unique_values);
    let dict = dict_encode(primitive_arr.as_ref()).unwrap();

    bencher
        .with_inputs(|| dict.clone())
        .bench_values(|dict| dict.to_canonical());
}

#[divan::bench(args = BENCH_ARGS)]
fn decode_varbin(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let varbin_arr = VarBinArray::from(gen_varbin_words(len, unique_values));
    let dict = dict_encode(varbin_arr.as_ref()).unwrap();

    bencher
        .with_inputs(|| dict.clone())
        .bench_values(|dict| dict.to_canonical());
}

#[divan::bench(args = BENCH_ARGS)]
fn decode_varbinview(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(len, unique_values));
    let dict = dict_encode(varbinview_arr.as_ref()).unwrap();

    bencher
        .with_inputs(|| dict.clone())
        .bench_values(|dict| dict.to_canonical());
}
