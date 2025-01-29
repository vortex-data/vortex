#![allow(clippy::unwrap_used)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng as _};
use vortex_array::patches::Patches;
use vortex_array::{ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_buffer::Buffer;

fn fixture(len: usize, sparsity: f64, rng: &mut StdRng) -> Patches {
    // NB: indices are always ordered
    let indices = (0..len)
        .filter(|_| rng.gen_bool(sparsity))
        .map(|x| x as u64)
        .collect::<Buffer<u64>>();
    let sparse_len = indices.len();
    let values = Buffer::from_iter((0..sparse_len).map(|x| x as u64)).into_array();
    Patches::new(len, indices.into_array(), values)
}

fn indices(array_len: usize, n_indices: usize, rng: &mut StdRng) -> ArrayData {
    Buffer::from_iter((0..n_indices).map(|_| rng.gen_range(0..(array_len as u64)))).into_array()
}

#[allow(clippy::cast_possible_truncation)]
fn bench_take(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_take");
    let mut rng = StdRng::seed_from_u64(0);

    for patches_sparsity in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001] {
        let patches = fixture(65_535, patches_sparsity, &mut rng);
        for index_multiple in [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001] {
            let indices = indices(
                patches.array_len(),
                (patches.array_len() as f64 * index_multiple) as usize,
                &mut rng,
            );
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "take_search: array_len={}, n_patches={} (~{}%), n_indices={} ({}%)",
                    patches.array_len(),
                    patches.num_patches(),
                    patches_sparsity,
                    indices.len(),
                    index_multiple * 100.0
                )),
                &(&patches, &indices),
                |b, (patches, indices)| {
                    b.iter(|| {
                        patches.take_search(
                            <&ArrayData>::clone(indices)
                                .clone()
                                .into_primitive()
                                .unwrap(),
                        )
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "take_map: array_len={}, n_patches={} (~{}%), n_indices={} ({}%)",
                    patches.array_len(),
                    patches.num_patches(),
                    patches_sparsity,
                    indices.len(),
                    index_multiple * 100.0
                )),
                &(&patches, &indices),
                |b, (patches, indices)| {
                    b.iter(|| {
                        patches.take_map(
                            <&ArrayData>::clone(indices)
                                .clone()
                                .into_primitive()
                                .unwrap(),
                        )
                    })
                },
            );
        }
    }
    group.finish()
}

criterion_group!(benches, bench_take);
criterion_main!(benches);
