// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use divan::Bencher;
use rand::Rng;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const ARRAY_LEN: usize = 1 << 16; // 65536

#[derive(Clone, Copy, Debug)]
enum Distribution {
    /// Patch indices are uniformly random across the array.
    Random,
    /// Patch indices are clustered together in a contiguous region.
    Clustered,
}

/// Combined benchmark arguments: (density, distribution).
const BENCH_ARGS: &[(f64, Distribution)] = &[
    (0.01, Distribution::Random),
    (0.01, Distribution::Clustered),
    (0.1, Distribution::Random),
    (0.1, Distribution::Clustered),
];

fn make_base_array(len: usize) -> PrimitiveArray {
    let buffer = Buffer::from_iter(0..len as u32);
    PrimitiveArray::new(buffer, Validity::NonNullable)
}

fn make_patches(array_len: usize, density: f64, dist: Distribution, rng: &mut StdRng) -> Patches {
    let num_patches = (array_len as f64 * density) as usize;

    let indices: Vec<u64> = match dist {
        Distribution::Random => {
            let mut raw: Vec<u64> = (0..num_patches)
                .map(|_| rng.random_range(0..array_len as u64))
                .collect();
            raw.sort_unstable();
            raw.dedup();
            raw
        }
        Distribution::Clustered => {
            // Place patches in a contiguous cluster starting at a random offset.
            let max_start = array_len.saturating_sub(num_patches);
            let start = rng.random_range(0..=max_start as u64);
            (start..start + num_patches as u64).collect()
        }
    };

    let n = indices.len();
    let values = Buffer::from_iter((0..n).map(|i| i as u32)).into_array();
    Patches::new(
        array_len,
        0,
        Buffer::from(indices).into_array(),
        values,
        None,
    )
    .unwrap()
}

#[divan::bench(args = BENCH_ARGS)]
fn patch_inplace(bencher: Bencher, &(density, dist): &(f64, Distribution)) {
    let mut rng = StdRng::seed_from_u64(42);
    let patches = make_patches(ARRAY_LEN, density, dist, &mut rng);

    bencher
        .with_inputs(|| {
            (
                make_base_array(ARRAY_LEN),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.patch(&patches, &mut ctx).unwrap());
}

#[divan::bench(args = BENCH_ARGS)]
fn patch_copy_to_buffer(bencher: Bencher, &(density, dist): &(f64, Distribution)) {
    let mut rng = StdRng::seed_from_u64(42);
    let patches = make_patches(ARRAY_LEN, density, dist, &mut rng);

    bencher
        .with_inputs(|| {
            (
                make_base_array(ARRAY_LEN),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| {
            let arr_ref = array.clone();
            (arr_ref.patch(&patches, &mut ctx).unwrap(), array)
        });
}
