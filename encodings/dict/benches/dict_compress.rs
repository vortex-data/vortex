#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::prelude::StdRng;
use rand::SeedableRng;
use vortex_array::array::{VarBinArray, VarBinViewArray};
use vortex_array::IntoCanonical;
use vortex_dict::builders::dict_encode;
use vortex_dict::test::{gen_primitive_for_dict, gen_varbin_words};

fn encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("dict_encode");

    let mut rng = StdRng::seed_from_u64(334);

    let primitive_arr = gen_primitive_for_dict::<i32>(&mut rng, 1_000_000, 50);
    group.throughput(Throughput::Bytes(primitive_arr.nbytes() as u64));
    group.bench_function("dict_encode_primitives", |b| {
        b.iter(|| dict_encode(primitive_arr.as_ref()));
    });

    let varbin_arr = VarBinArray::from(gen_varbin_words(&mut rng, 1_000_000, 50));
    group.throughput(Throughput::Bytes(varbin_arr.nbytes() as u64));
    group.bench_function("dict_encode_varbin", |b| {
        b.iter(|| dict_encode(varbin_arr.as_ref()));
    });

    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(&mut rng, 1_000_000, 50));
    group.throughput(Throughput::Bytes(varbinview_arr.nbytes() as u64));
    group.bench_function("dict_encode_view", |b| {
        b.iter(|| dict_encode(varbinview_arr.as_ref()));
    });
}

fn decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("dict_decode");

    let mut rng = StdRng::seed_from_u64(334);

    let primitive_arr = gen_primitive_for_dict::<u8>(&mut rng, 1_000_000, 32);
    let dict = dict_encode(primitive_arr.as_ref()).unwrap();
    group.throughput(Throughput::Bytes(primitive_arr.nbytes() as u64));
    group.bench_function("dict_decode_primitives_u8", |b| {
        b.iter_with_setup(|| dict.clone(), |dict| dict.into_canonical().unwrap())
    });

    let primitive_arr = gen_primitive_for_dict::<i32>(&mut rng, 1_000_000, 32);
    let dict = dict_encode(primitive_arr.as_ref()).unwrap();
    group.throughput(Throughput::Bytes(primitive_arr.nbytes() as u64));
    group.bench_function("dict_decode_primitives_i32", |b| {
        b.iter_with_setup(|| dict.clone(), |dict| dict.into_canonical().unwrap())
    });

    let varbin_arr = VarBinArray::from(gen_varbin_words(&mut rng, 1_000_000, 50));
    let dict = dict_encode(varbin_arr.as_ref()).unwrap();
    group.throughput(Throughput::Bytes(varbin_arr.nbytes() as u64));
    group.bench_function("dict_decode_varbin", |b| {
        b.iter_with_setup(|| dict.clone(), |dict| dict.into_canonical().unwrap())
    });

    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(&mut rng, 1_000_000, 50));
    let dict = dict_encode(varbinview_arr.as_ref()).unwrap();
    group.throughput(Throughput::Bytes(varbin_arr.nbytes() as u64));
    group.bench_function("dict_decode_view", |b| {
        b.iter_with_setup(|| dict.clone(), |dict| dict.into_canonical().unwrap())
    });
}

criterion_group!(benches, encode, decode);
criterion_main!(benches);
