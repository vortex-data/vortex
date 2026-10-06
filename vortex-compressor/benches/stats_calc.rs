// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use mimalloc::MiMalloc;

#[cfg(not(codspeed))]
#[divan::bench_group]
mod benchmarks {
    use std::sync::LazyLock;

    use divan::Bencher;
    use divan::counter::BytesCount;
    use num_traits::AsPrimitive;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::NativePType;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_compressor::stats::GenerateStatsOptions;
    use vortex_compressor::stats::IntegerStats;
    use vortex_session::VortexSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    /// The uncompressed size of each benchmarked array, matching a writer chunk.
    const BYTES: usize = 4 << 20;

    #[derive(Debug, Copy, Clone)]
    enum Distribution {
        Constant,
        LowCardinality,
        ShortRuns,
        LongRuns,
        VeryLongRuns,
        /// A multiple of the position, so every value differs.
        Sequence,
        /// Uniformly random over 200 values.
        SmallRange,
        /// Uniformly random over a range wider than the dense distinct-counting limit.
        WideRandom,
    }

    const DISTRIBUTIONS: [Distribution; 8] = [
        Distribution::Constant,
        Distribution::LowCardinality,
        Distribution::ShortRuns,
        Distribution::LongRuns,
        Distribution::VeryLongRuns,
        Distribution::Sequence,
        Distribution::SmallRange,
        Distribution::WideRandom,
    ];

    /// A seeded xorshift generator, so that every build benchmarks the same data.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }
    }

    fn generate_runs(len: usize, max_run: u32, distinct: u32) -> Vec<u32> {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut output = Vec::with_capacity(len);
        let mut run = 0;
        let mut value = 0;
        for _ in 0..len {
            if run == 0 {
                value = rng.next() % distinct;
                run = std::cmp::max(rng.next() % max_run, 1);
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
        let len = BYTES / size_of::<T>();
        let values: Vec<u32> = match distribution {
            Distribution::Constant => vec![7; len],
            Distribution::LowCardinality => (0..1024).cycle().take(len).collect(),
            Distribution::ShortRuns => generate_runs(len, 4, 1024),
            Distribution::LongRuns => generate_runs(len, 64, 1024),
            Distribution::VeryLongRuns => generate_runs(len, 512, 1024),
            Distribution::Sequence => (0u32..).take(len).map(|i| i.wrapping_mul(3)).collect(),
            Distribution::SmallRange => {
                let mut rng = Rng(0x2545_F491_4F6C_DD1D);
                (0..len).map(|_| 1000 + rng.next() % 200).collect()
            }
            Distribution::WideRandom => {
                let mut rng = Rng(0x2545_F491_4F6C_DD1D);
                (0..len).map(|_| rng.next() % 1_000_000).collect()
            }
        };
        values.into_iter().map(|v| v.as_()).collect()
    }

    fn bench_stats<T>(
        bencher: Bencher,
        distribution: Distribution,
        count_distinct_values: bool,
        nullable: bool,
    ) where
        T: NativePType,
        u32: AsPrimitive<T>,
    {
        let values = generate::<T>(distribution);
        let validity = if nullable {
            let mut rng = Rng(0xD1B5_4A32_D192_ED03);
            Validity::from(BitBuffer::from_iter(
                (0..values.len()).map(|_| !rng.next().is_multiple_of(10)),
            ))
        } else {
            Validity::NonNullable
        };
        // A fresh array per iteration, so min/max cached on the array by a previous iteration is
        // not reused.
        bencher
            .counter(BytesCount::new(BYTES))
            .with_inputs(|| {
                (
                    PrimitiveArray::new(values.clone(), validity.clone()),
                    SESSION.create_execution_ctx(),
                )
            })
            .bench_refs(|(array, ctx)| {
                IntegerStats::generate_opts(
                    array,
                    GenerateStatsOptions {
                        count_distinct_values,
                    },
                    ctx,
                )
            });
    }

    #[divan::bench(types = [u8, u16, u32, u64, i32, i64], args = DISTRIBUTIONS)]
    fn stats_dict_on<T>(bencher: Bencher, distribution: Distribution)
    where
        T: NativePType,
        u32: AsPrimitive<T>,
    {
        bench_stats::<T>(bencher, distribution, true, false);
    }

    #[divan::bench(types = [u8, u16, u32, u64, i32, i64], args = DISTRIBUTIONS)]
    fn stats_dict_off<T>(bencher: Bencher, distribution: Distribution)
    where
        T: NativePType,
        u32: AsPrimitive<T>,
    {
        bench_stats::<T>(bencher, distribution, false, false);
    }

    #[divan::bench(types = [u8, u32, u64], args = DISTRIBUTIONS)]
    fn stats_nullable_dict_on<T>(bencher: Bencher, distribution: Distribution)
    where
        T: NativePType,
        u32: AsPrimitive<T>,
    {
        bench_stats::<T>(bencher, distribution, true, true);
    }

    #[divan::bench(types = [u8, u32, u64], args = DISTRIBUTIONS)]
    fn stats_nullable_dict_off<T>(bencher: Bencher, distribution: Distribution)
    where
        T: NativePType,
        u32: AsPrimitive<T>,
    {
        bench_stats::<T>(bencher, distribution, false, true);
    }
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}
