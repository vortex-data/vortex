// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(unexpected_cfgs)]

#[cfg(not(codspeed))]
mod benchmarks {
    use divan::Bencher;
    use divan::counter::BytesCount;
    use divan::counter::ItemsCount;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_btrblocks::BtrBlocksCompressor;
    use vortex_buffer::buffer_mut;
    use vortex_utils::aliases::hash_set::HashSet;

    fn make_clickbench_window_name() -> ArrayRef {
        // A test that's meant to mirror the WindowName column from ClickBench.
        let mut values = buffer_mut![-1i32; 65_536];
        let mut visited = HashSet::new();
        let mut rng = StdRng::seed_from_u64(1u64);
        while visited.len() < 223 {
            let random = (rng.next_u32() as usize) % 65_536;
            if visited.contains(&random) {
                continue;
            }
            visited.insert(random);
            // Pick 100 random values to insert.
            values[random] = 5 * (rng.next_u64() % 100) as i32;
        }

        // Ok, now let's compress
        values.freeze().into_array()
    }

    #[divan::bench]
    fn btrblocks(bencher: Bencher) {
        let array = make_clickbench_window_name().to_primitive();
        let compressor = BtrBlocksCompressor::default();
        bencher
            .with_inputs(|| &array)
            .input_counter(|array| ItemsCount::new(array.len()))
            .input_counter(|array| BytesCount::of_many::<i32>(array.len()))
            .bench_refs(|array| compressor.compress(&array.clone().into_array()).unwrap());
    }
}

fn main() {
    divan::main()
}
