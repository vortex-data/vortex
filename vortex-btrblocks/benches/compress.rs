// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

#[cfg(not(codspeed))]
mod benchmarks {
    use std::sync::LazyLock;

    use divan::Bencher;
    use divan::counter::BytesCount;
    use divan::counter::ItemsCount;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_btrblocks::BtrBlocksCompressor;
    use vortex_buffer::buffer_mut;
    use vortex_session::VortexSession;
    use vortex_utils::aliases::hash_set::HashSet;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

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
        let mut ctx = SESSION.create_execution_ctx();
        let array = make_clickbench_window_name()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let compressor = BtrBlocksCompressor::default();
        bencher
            .with_inputs(|| (&array, SESSION.create_execution_ctx()))
            .input_counter(|(array, _)| ItemsCount::new(array.len()))
            .input_counter(|(array, _)| BytesCount::of_many::<i32>(array.len()))
            .bench_refs(|(array, ctx)| {
                compressor
                    .compress(&array.clone().into_array(), ctx)
                    .unwrap()
            });
    }
}

fn main() {
    divan::main()
}
