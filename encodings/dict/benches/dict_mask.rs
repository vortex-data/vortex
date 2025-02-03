#![allow(clippy::unwrap_used)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng as _};
use vortex_array::array::PrimitiveArray;
use vortex_array::compute::mask;
use vortex_array::IntoArray as _;
use vortex_buffer::buffer;
use vortex_dict::DictArray;
use vortex_mask::Mask;

fn filter_mask(len: usize, fraction_masked: f64, rng: &mut StdRng) -> Mask {
    let indices = (0..len)
        .filter(|_| rng.gen_bool(fraction_masked))
        .collect::<Vec<usize>>();
    Mask::from_indices(len, indices)
}

#[allow(clippy::cast_possible_truncation)]
fn bench_dict_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_dict_mask");
    let mut rng = StdRng::seed_from_u64(0);

    let len = 65_535;
    // for fraction_valid in [0.5, 0.1, 0.01, 0.001, 0.0001] {
    for fraction_valid in [0.1] {
        let codes =
            PrimitiveArray::from_iter((0..len).map(|_| (!rng.gen_bool(fraction_valid)) as u64))
                .into_array();
        let values = buffer![1].into_array();
        let array = DictArray::try_new(codes, values).unwrap().into_array();
        // for fraction_masked in [0.1, 0.01, 0.001, 0.0001] {
        for fraction_masked in [0.9, 0.5, 0.1, 0.0001] {
            let filter_mask = filter_mask(len, fraction_masked, &mut rng);
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "fraction_valid={}, fraction_masked={}",
                    fraction_valid, fraction_masked
                )),
                &(&array, filter_mask),
                |b, (array, filter_mask)| b.iter(|| mask(array, filter_mask.clone()).unwrap()),
            );
        }
    }
    group.finish()
}

criterion_group!(benches, bench_dict_mask);
criterion_main!(benches);
