#![allow(clippy::unwrap_used)]

use std::str::from_utf8;

use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::{ConstantArray, VarBinArray, VarBinViewArray};
use vortex_array::compute::{Operator, compare};
use vortex_dict::builders::dict_encode;
use vortex_dict::test::{gen_primitive_for_dict, gen_varbin_words};

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, usize)] = &[
    // length, unique_values
    (10_000, 2),
    (10_000, 4),
    (10_000, 8),
    (10_000, 32),
    (10_000, 128),
    (10_000, 512),
    (10_000, 2048),
    (100_000, 2),
    (100_000, 4),
    (100_000, 8),
    (100_000, 32),
    (100_000, 128),
    (100_000, 512),
    (100_000, 2048),
];

#[divan::bench(args = BENCH_ARGS)]
fn bench_compare_primitive(bencher: divan::Bencher, (len, uniqueness): (usize, usize)) {
    let primitive_arr = gen_primitive_for_dict::<i32>(len, uniqueness);
    let dict = dict_encode(&primitive_arr).unwrap();
    let value = primitive_arr.as_slice::<i32>()[0];

    bencher
        .with_inputs(|| dict.clone())
        .bench_refs(|dict| compare(dict, &ConstantArray::new(value, len), Operator::Eq).unwrap())
}

#[divan::bench(args = BENCH_ARGS)]
fn bench_compare_varbin(bencher: divan::Bencher, (len, uniqueness): (usize, usize)) {
    let varbin_arr = VarBinArray::from(gen_varbin_words(len, uniqueness));
    let dict = dict_encode(&varbin_arr).unwrap();
    let bytes = varbin_arr
        .with_iterator(|i| i.next().unwrap().unwrap().to_vec())
        .unwrap();
    let value = from_utf8(bytes.as_slice()).unwrap();

    bencher
        .with_inputs(|| dict.clone())
        .bench_refs(|dict| compare(dict, &ConstantArray::new(value, len), Operator::Eq).unwrap())
}

#[divan::bench(args = BENCH_ARGS)]
fn bench_compare_varbinview(bencher: divan::Bencher, (len, uniqueness): (usize, usize)) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(len, uniqueness));
    let dict = dict_encode(&varbinview_arr).unwrap();
    let bytes = varbinview_arr
        .with_iterator(|i| i.next().unwrap().unwrap().to_vec())
        .unwrap();
    let value = from_utf8(bytes.as_slice()).unwrap();
    bencher
        .with_inputs(|| dict.clone())
        .bench_refs(|dict| compare(dict, &ConstantArray::new(value, len), Operator::Eq).unwrap())
}
