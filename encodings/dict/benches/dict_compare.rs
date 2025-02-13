#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::str::from_utf8;

use rand::distributions::{Alphanumeric, Uniform};
use rand::prelude::SliceRandom;
use rand::{thread_rng, Rng};
use vortex_array::accessor::ArrayAccessor;
use vortex_array::array::{ConstantArray, PrimitiveArray, VarBinArray, VarBinViewArray};
use vortex_array::compute::{compare, Operator};
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_dict::builders::dict_encode;

fn gen_primitive_dict(len: usize, uniqueness: f64) -> PrimitiveArray {
    let mut rng = thread_rng();
    let value_range = len as f64 * uniqueness;
    let range = Uniform::new(-(value_range / 2.0) as i32, (value_range / 2.0) as i32);
    let data: Buffer<i32> = (0..len).map(|_| rng.sample(range)).collect();
    PrimitiveArray::new(data, Validity::NonNullable)
}

fn gen_varbin_words(len: usize, uniqueness: f64) -> Vec<String> {
    let mut rng = thread_rng();
    let uniq_cnt = (len as f64 * uniqueness) as usize;
    let dict: Vec<String> = (0..uniq_cnt)
        .map(|_| {
            (&mut rng)
                .sample_iter(&Alphanumeric)
                .take(16)
                .map(char::from)
                .collect()
        })
        .collect();
    (0..len)
        .map(|_| dict.choose(&mut rng).unwrap().clone())
        .collect()
}

#[divan::bench(args = BENCH_ARGS)]
fn bench_compare_primitive(bencher: divan::Bencher, (len, uniqueness): (usize, f64)) {
    let primitive_arr = gen_primitive_dict(len, uniqueness);
    let dict = dict_encode(primitive_arr.as_ref()).unwrap();
    let value = primitive_arr.as_slice::<i32>()[0];

    bencher
        .with_inputs(|| dict.clone())
        .bench_local_values(|dict| {
            compare(dict, ConstantArray::new(value, len), Operator::Eq).unwrap()
        })
}

#[divan::bench(args = BENCH_ARGS)]
fn bench_compare_varbin(bencher: divan::Bencher, (len, uniqueness): (usize, f64)) {
    let varbin_arr = VarBinArray::from(gen_varbin_words(len, uniqueness));
    let dict = dict_encode(varbin_arr.as_ref()).unwrap();
    let bytes = varbin_arr
        .with_iterator(|i| i.next().unwrap().unwrap().to_vec())
        .unwrap();
    let value = from_utf8(bytes.as_slice()).unwrap();

    bencher
        .with_inputs(|| dict.clone())
        .bench_local_values(|dict| {
            compare(dict, ConstantArray::new(value, len), Operator::Eq).unwrap()
        })
}

#[divan::bench(args = BENCH_ARGS)]
fn bench_compare_varbinview(bencher: divan::Bencher, (len, uniqueness): (usize, f64)) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(len, uniqueness));
    let dict = dict_encode(varbinview_arr.as_ref()).unwrap();
    let bytes = varbinview_arr
        .with_iterator(|i| i.next().unwrap().unwrap().to_vec())
        .unwrap();
    let value = from_utf8(bytes.as_slice()).unwrap();
    bencher
        .with_inputs(|| dict.clone())
        .bench_local_values(|dict| {
            compare(dict, ConstantArray::new(value, len), Operator::Eq).unwrap()
        })
}

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, f64)] = &[
    (1_000_000, 0.00005),
    (10_000_000, 0.00005),
    (100_000_000, 0.00005),
    (1_000_000, 0.05),
    (10_000_000, 0.05),
    (100_000_000, 0.05),
];
