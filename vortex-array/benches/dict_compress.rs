// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use rand::distr::Distribution;
use rand::distr::StandardUniform;
use vortex_array::IntoArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict_test::gen_primitive_for_dict;
use vortex_array::arrays::dict_test::gen_varbin_words;
use vortex_array::builders::dict::dict_encode;
use vortex_array::dtype::NativePType;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, usize)] = &[
    // length, unique_values
    (1_000, 2),
    (1_000, 4),
    (1_000, 8),
    (1_000, 32),
    (1_000, 512),
    (10_000, 2),
    (10_000, 4),
    (10_000, 8),
    (10_000, 32),
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
        .with_inputs(|| &primitive_arr)
        .bench_refs(|arr| dict_encode(&arr.clone().into_array()));
}

#[divan::bench(args = BENCH_ARGS)]
fn encode_varbin(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let varbin_arr = VarBinArray::from(gen_varbin_words(len, unique_values));

    bencher
        .with_inputs(|| &varbin_arr)
        .bench_refs(|arr| dict_encode(&arr.clone().into_array()));
}

#[divan::bench(args = BENCH_ARGS)]
fn encode_varbinview(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(len, unique_values));

    bencher
        .with_inputs(|| &varbinview_arr)
        .bench_refs(|arr| dict_encode(&arr.clone().into_array()));
}

#[divan::bench(types = [u8, f32, i64], args = BENCH_ARGS)]
fn decode_primitives<T>(bencher: Bencher, (len, unique_values): (usize, usize))
where
    T: NativePType,
    StandardUniform: Distribution<T>,
{
    let primitive_arr = gen_primitive_for_dict::<T>(len, unique_values);
    let dict = dict_encode(&primitive_arr.into_array())
        .unwrap()
        .into_array();

    bencher
        .with_inputs(|| &dict)
        .bench_refs(|dict| dict.to_canonical());
}

#[divan::bench(args = BENCH_ARGS)]
fn decode_varbin(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let varbin_arr = VarBinArray::from(gen_varbin_words(len, unique_values));
    let dict = dict_encode(&varbin_arr.into_array()).unwrap().into_array();

    bencher
        .with_inputs(|| &dict)
        .bench_refs(|dict| dict.to_canonical());
}

#[divan::bench(args = BENCH_ARGS)]
fn decode_varbinview(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(len, unique_values));
    let dict = dict_encode(&varbinview_arr.into_array())
        .unwrap()
        .into_array();

    bencher
        .with_inputs(|| &dict)
        .bench_refs(|dict| dict.to_canonical());
}
