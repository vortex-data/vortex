#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::mask;
use vortex_dict::DictArray;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

fn filter_mask(len: usize, fraction_masked: f64, rng: &mut StdRng) -> Mask {
    let indices = (0..len)
        .filter(|_| rng.random_bool(fraction_masked))
        .collect::<Vec<usize>>();
    Mask::from_indices(len, indices)
}

#[divan::bench(args = [
    (0.9, 0.9),
    (0.9, 0.5),
    (0.9, 0.1),
    (0.9, 0.01),
    (0.5, 0.9),
    (0.5, 0.5),
    (0.5, 0.1),
    (0.5, 0.01),
    (0.1, 0.9),
    (0.1, 0.5),
    (0.1, 0.1),
    (0.1, 0.01),
    (0.01, 0.9),
    (0.01, 0.5),
    (0.01, 0.1),
    (0.01, 0.01),
])]
fn bench_dict_mask(bencher: Bencher, (fraction_valid, fraction_masked): (f64, f64)) {
    let mut rng = StdRng::seed_from_u64(0);

    let len = 65_535;
    let codes = PrimitiveArray::from_iter((0..len).map(|_| {
        if rng.random_bool(fraction_valid) {
            1u64
        } else {
            0u64
        }
    }))
    .into_array();
    let values = PrimitiveArray::from_option_iter([None, Some(42i32)]).into_array();
    let array = DictArray::try_new(codes, values).unwrap().into_array();
    let filter_mask = filter_mask(len, fraction_masked, &mut rng);
    bencher
        .with_inputs(|| (&array, filter_mask.clone()))
        .bench_values(|(array, filter_mask)| mask(array, &filter_mask).unwrap());
}
