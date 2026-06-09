// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(not(codspeed))]
#[divan::bench_group(items_count = 64_000u32, bytes_count = 256_000u32)]
mod benchmarks {
    use std::sync::LazyLock;

    use divan::Bencher;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;
    use vortex_compressor::stats::GenerateStatsOptions;
    use vortex_compressor::stats::IntegerStats;
    use vortex_session::VortexSession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn generate_dataset(max_run: u32, distinct: u32) -> Buffer<u32> {
        let mut output = BufferMut::with_capacity(64_000);
        let mut run = 0;
        let mut value = 0;
        for _ in 0..64_000 {
            if run == 0 {
                value = rand::random::<u32>() % distinct;
                run = std::cmp::max(rand::random::<u32>() % max_run, 1);
            }
            output.push(value);
            run -= 1;
        }

        output.freeze()
    }

    #[derive(Debug, Copy, Clone)]
    enum Distribution {
        LowCardinality,
        ShortRuns,
        LongRuns,
    }

    fn generate_low_cardinality() -> PrimitiveArray {
        let values: Buffer<u32> = (0..1024).cycle().take(64_000).collect();
        PrimitiveArray::new(values, Validity::NonNullable)
    }

    fn generate_runs(max_run: u32) -> PrimitiveArray {
        let values = generate_dataset(max_run, 1024);
        PrimitiveArray::new(values, Validity::NonNullable)
    }

    #[divan::bench(args = [Distribution::LowCardinality, Distribution::ShortRuns, Distribution::LongRuns])]
    fn stats_dict_on(bencher: Bencher, distribution: Distribution) {
        let values = match distribution {
            Distribution::LowCardinality => generate_low_cardinality(),
            Distribution::ShortRuns => generate_runs(4),
            Distribution::LongRuns => generate_runs(64),
        };

        bencher
            .with_inputs(|| (&values, SESSION.create_execution_ctx()))
            .bench_refs(|(values, ctx)| {
                IntegerStats::generate_opts(values, GenerateStatsOptions::default(), ctx);
            });
    }

    #[divan::bench(args = [Distribution::LowCardinality, Distribution::ShortRuns, Distribution::LongRuns])]
    fn stats_dict_off(bencher: Bencher, distribution: Distribution) {
        let values = match distribution {
            Distribution::LowCardinality => generate_low_cardinality(),
            Distribution::ShortRuns => generate_runs(4),
            Distribution::LongRuns => generate_runs(64),
        };

        bencher
            .with_inputs(|| (&values, SESSION.create_execution_ctx()))
            .bench_refs(|(values, ctx)| {
                IntegerStats::generate_opts(
                    values,
                    GenerateStatsOptions {
                        count_distinct_values: false,
                    },
                    ctx,
                );
            });
    }
}

fn main() {
    divan::main();
}
