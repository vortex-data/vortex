#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::str::from_utf8;

use rand::distributions::Alphanumeric;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use vortex_array::accessor::ArrayAccessor;
use vortex_array::array::{ConstantArray, VarBinArray, VarBinViewArray};
use vortex_array::compute::{compare, Operator};
use vortex_dict::builders::dict_encode;
use vortex_dict::test::gen_primitive_for_dict;

fn gen_varbin_words(len: usize, unique_count: usize) -> Vec<String> {
    let mut rng = thread_rng();
    let dict: Vec<String> = (0..unique_count)
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
fn bench_compare_primitive(bencher: divan::Bencher, (len, uniqueness): (usize, usize)) {
    let mut rng = StdRng::seed_from_u64(3342);
    let primitive_arr = gen_primitive_for_dict::<i32>(&mut rng, len, uniqueness);
    let dict = dict_encode(primitive_arr.as_ref()).unwrap();
    let value = primitive_arr.as_slice::<i32>()[0];

    bencher
        .with_inputs(|| dict.clone())
        .bench_local_values(|dict| {
            compare(dict, ConstantArray::new(value, len), Operator::Eq).unwrap()
        })
}

#[divan::bench(args = BENCH_ARGS)]
fn bench_compare_varbin(bencher: divan::Bencher, (len, uniqueness): (usize, usize)) {
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
fn bench_compare_varbinview(bencher: divan::Bencher, (len, uniqueness): (usize, usize)) {
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

const BENCH_ARGS: &[(usize, usize)] = &[
    (1_000_000, 50),
    (10_000_000, 50),
    (100_000_000, 50),
    (1_000_000, 5000),
    (10_000_000, 5000),
    (100_000_000, 5000),
];
