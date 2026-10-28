// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use mimalloc::MiMalloc;

#[cfg(not(codspeed))]
#[divan::bench_group(items_count = 64_000u32)]
mod benchmarks {
    use std::sync::LazyLock;

    use divan::Bencher;
    use num_traits::AsPrimitive;
    use rand::prelude::*;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::session::AggregateFnSession;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::NativePType;
    use vortex_array::optimizer::kernels::KernelSession;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_compressor::stats::GenerateStatsOptions;
    use vortex_compressor::stats::IntegerStats;
    use vortex_session::VortexSession;

    pub(super) static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        VortexSession::empty()
            .with::<KernelSession>()
            .with::<AggregateFnSession>()
    });

    const LEN: usize = 64_000;

    #[derive(Debug, Copy, Clone)]
    enum Distribution {
        Constant,
        LowCardinality,
        ShortRuns,
        LongRuns,
        VeryLongRuns,
    }

    const DISTRIBUTIONS: [Distribution; 5] = [
        Distribution::Constant,
        Distribution::LowCardinality,
        Distribution::ShortRuns,
        Distribution::LongRuns,
        Distribution::VeryLongRuns,
    ];

    fn generate_runs(max_run: u32, distinct: u32) -> Vec<u32> {
        let mut rng = StdRng::seed_from_u64(0);
        let mut output = Vec::with_capacity(LEN);
        let mut run = 0;
        let mut value = 0;
        for _ in 0..LEN {
            if run == 0 {
                value = rng.random::<u32>() % distinct;
                run = std::cmp::max(rng.random::<u32>() % max_run, 1);
            }
            output.push(value);
            run -= 1;
        }
        output
    }

    fn generate<T>(distribution: Distribution) -> Buffer<T>
    where
        T: NativePType,
        u32: AsPrimitive<T>,
    {
        let values: Vec<u32> = match distribution {
            Distribution::Constant => vec![7; LEN],
            Distribution::LowCardinality => (0..1024).cycle().take(LEN).collect(),
            Distribution::ShortRuns => generate_runs(4, 1024),
            Distribution::LongRuns => generate_runs(64, 1024),
            Distribution::VeryLongRuns => generate_runs(512, 1024),
        };
        values.into_iter().map(|v| v.as_()).collect()
    }

    fn bench_stats<T>(bencher: Bencher, distribution: Distribution, count_distinct_values: bool)
    where
        T: NativePType,
        u32: AsPrimitive<T>,
    {
        let values = generate::<T>(distribution);
        bencher
            .with_inputs(|| {
                (
                    PrimitiveArray::new(values.clone(), Validity::NonNullable),
                    SESSION.create_execution_ctx(),
                )
            })
            .bench_refs(|(values, ctx)| {
                IntegerStats::generate_opts(
                    values,
                    GenerateStatsOptions {
                        count_distinct_values,
                    },
                    ctx,
                )
            });
    }

    #[divan::bench(types = [u8, u32], args = DISTRIBUTIONS)]
    fn stats_dict_on<T>(bencher: Bencher, distribution: Distribution)
    where
        T: NativePType,
        u32: AsPrimitive<T>,
    {
        bench_stats::<T>(bencher, distribution, true);
    }

    #[divan::bench(types = [u8, u32], args = DISTRIBUTIONS)]
    fn stats_dict_off<T>(bencher: Bencher, distribution: Distribution)
    where
        T: NativePType,
        u32: AsPrimitive<T>,
    {
        bench_stats::<T>(bencher, distribution, false);
    }
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    #[cfg(not(codspeed))]
    std::sync::LazyLock::force(&benchmarks::SESSION);
    divan::main();
}
