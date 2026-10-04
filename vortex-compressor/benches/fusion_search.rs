// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures which statistics gain from fusing, for choosing `Fused` groups.
//!
//! For every pair of statistics, `fused` runs both in one loop per block and `blocked` runs one
//! loop each over every block. A pair whose fused time is clearly below its blocked time has an
//! affinity for fusion.

use mimalloc::MiMalloc;

#[cfg(not(codspeed))]
#[divan::bench_group]
mod benchmarks {
    use divan::Bencher;
    use divan::black_box;
    use divan::counter::BytesCount;
    use num_traits::AsPrimitive;
    use vortex_buffer::Buffer;
    use vortex_compressor::stats::accumulator::BitWidthHistogram;
    use vortex_compressor::stats::accumulator::CommonBits;
    use vortex_compressor::stats::accumulator::DeltaRange;
    use vortex_compressor::stats::accumulator::Fused;
    use vortex_compressor::stats::accumulator::IntValue;
    use vortex_compressor::stats::accumulator::MinMax;
    use vortex_compressor::stats::accumulator::RunCount;
    use vortex_compressor::stats::accumulator::Sorted;
    use vortex_compressor::stats::accumulator::Sum;
    use vortex_compressor::stats::accumulator::accumulate;
    use vortex_mask::Mask;

    /// The uncompressed size of each benchmarked array, matching a writer chunk.
    const BYTES: usize = 4 << 20;

    /// Every value valid, or roughly one in ten values null.
    const NULLABLE: [bool; 2] = [false, true];

    /// Runs of up to 64 equal values drawn from 1024 distinct values, from a seeded xorshift.
    fn generate<T>() -> Buffer<T>
    where
        T: Copy + 'static,
        u32: AsPrimitive<T>,
    {
        let len = BYTES / size_of::<T>();
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 32) as u32
        };
        let mut output = Vec::with_capacity(len);
        while output.len() < len {
            let value = next() % 1024;
            let run = (next() % 64).max(1) as usize;
            output.extend(std::iter::repeat_n(
                value.as_(),
                run.min(len - output.len()),
            ));
        }
        output.into_iter().collect()
    }

    fn bench<T, R>(bencher: Bencher, nullable: bool, f: impl Fn(&[T], &Mask) -> R + Sync)
    where
        T: Copy + Sync + Send + 'static,
        u32: AsPrimitive<T>,
    {
        let values = generate::<T>();
        let len = values.len();
        let mask = if nullable {
            Mask::from_iter((0..len).map(|i| !(i * 2_654_435_761).is_multiple_of(10)))
        } else {
            Mask::new_true(len)
        };
        bencher
            .counter(BytesCount::new(BYTES))
            .bench(|| f(black_box(&values), black_box(&mask)));
    }

    /// Benchmarks a pair of statistics fused and blocked.
    macro_rules! pair {
        ($name:ident, $a:expr, $b:expr) => {
            mod $name {
                use super::*;

                #[divan::bench(types = [u8, u16, u32, u64, i64], args = NULLABLE)]
                fn fused<T>(bencher: Bencher, nullable: bool)
                where
                    T: IntValue + Sync + Send,
                    u32: AsPrimitive<T>,
                {
                    bench::<T, _>(bencher, nullable, |v: &[T], m: &Mask| {
                        accumulate(v, m, Fused(($a, $b)))
                    });
                }

                #[divan::bench(types = [u8, u16, u32, u64, i64], args = NULLABLE)]
                fn blocked<T>(bencher: Bencher, nullable: bool)
                where
                    T: IntValue + Sync + Send,
                    u32: AsPrimitive<T>,
                {
                    bench::<T, _>(bencher, nullable, |v: &[T], m: &Mask| {
                        accumulate(v, m, ($a, $b))
                    });
                }
            }
        };
    }

    pair!(min_max_x_run_count, MinMax::new(), RunCount::new());
    pair!(min_max_x_sorted, MinMax::new(), Sorted::new());
    pair!(min_max_x_sum, MinMax::new(), Sum::new());
    pair!(min_max_x_common_bits, MinMax::new(), CommonBits::new());
    pair!(min_max_x_histogram, MinMax::new(), BitWidthHistogram::new());
    pair!(min_max_x_delta, MinMax::new(), DeltaRange::new());
    pair!(run_count_x_sorted, RunCount::new(), Sorted::new());
    pair!(run_count_x_sum, RunCount::new(), Sum::new());
    pair!(run_count_x_common_bits, RunCount::new(), CommonBits::new());
    pair!(
        run_count_x_histogram,
        RunCount::new(),
        BitWidthHistogram::new()
    );
    pair!(run_count_x_delta, RunCount::new(), DeltaRange::new());
    pair!(sorted_x_sum, Sorted::new(), Sum::new());
    pair!(sorted_x_common_bits, Sorted::new(), CommonBits::new());
    pair!(sorted_x_histogram, Sorted::new(), BitWidthHistogram::new());
    pair!(sorted_x_delta, Sorted::new(), DeltaRange::new());
    pair!(sum_x_common_bits, Sum::new(), CommonBits::new());
    pair!(sum_x_histogram, Sum::new(), BitWidthHistogram::new());
    pair!(sum_x_delta, Sum::new(), DeltaRange::new());
    pair!(
        common_bits_x_histogram,
        CommonBits::new(),
        BitWidthHistogram::new()
    );
    pair!(common_bits_x_delta, CommonBits::new(), DeltaRange::new());
    pair!(
        histogram_x_delta,
        BitWidthHistogram::new(),
        DeltaRange::new()
    );
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}
