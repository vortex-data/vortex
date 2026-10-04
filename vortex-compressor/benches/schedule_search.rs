// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Times every schedule of the seven statistics the integer schemes use, to find the best.
//!
//! The statistics are ordered so that the pairs `fusion_search` measured as gaining from fusion
//! are adjacent, and every one of the 64 ways to split them into consecutive loops is timed. A
//! schedule is printed as its `BREAKS` constant: bit `i` set starts a new loop at statistic `i`.

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
    use vortex_compressor::stats::accumulator::IntValue;
    use vortex_compressor::stats::accumulator::MinMax;
    use vortex_compressor::stats::accumulator::RunCount;
    use vortex_compressor::stats::accumulator::Schedule;
    use vortex_compressor::stats::accumulator::Sorted;
    use vortex_compressor::stats::accumulator::Sum;
    use vortex_compressor::stats::accumulator::accumulate;
    use vortex_mask::Mask;

    /// The uncompressed size of each benchmarked array, matching a writer chunk.
    const BYTES: usize = 4 << 20;

    /// Every value valid, or roughly one in ten values null.
    const NULLABLE: [bool; 2] = [false, true];

    /// Every schedule of seven statistics, each combination of loop starts at statistics 1 to 6,
    /// as four sets of 16 because a benchmark takes at most 20 constants.
    const fn schedules(set: u64) -> [u64; 16] {
        let mut schedules = [0; 16];
        let mut i = 0;
        while i < 16 {
            schedules[i] = (set * 16 + i as u64) << 1;
            i += 1;
        }
        schedules
    }

    const SCHEDULES_0: [u64; 16] = schedules(0);
    const SCHEDULES_1: [u64; 16] = schedules(1);
    const SCHEDULES_2: [u64; 16] = schedules(2);
    const SCHEDULES_3: [u64; 16] = schedules(3);

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

    /// The seven statistics, ordered so that pairs with an affinity for fusion are adjacent.
    #[allow(clippy::type_complexity)]
    fn stats<T: IntValue>() -> (
        Sum<T>,
        MinMax<T>,
        CommonBits<T>,
        BitWidthHistogram<T>,
        DeltaRange<T>,
        Sorted<T>,
        RunCount<T>,
    ) {
        (
            Sum::new(),
            MinMax::new(),
            CommonBits::new(),
            BitWidthHistogram::new(),
            DeltaRange::new(),
            Sorted::new(),
            RunCount::new(),
        )
    }

    fn run_schedule<T, const BREAKS: u64>(bencher: Bencher, nullable: bool)
    where
        T: IntValue + Sync + Send,
        u32: AsPrimitive<T>,
    {
        bench::<T, _>(bencher, nullable, |v: &[T], m: &Mask| {
            accumulate(v, m, Schedule::<_, BREAKS>::new(stats::<T>()))
        });
    }

    /// Benchmarks one set of 16 schedules.
    macro_rules! schedule_set {
        ($name:ident, $schedules:ident) => {
            #[divan::bench(types = [u8, u16, u32, u64, i64], consts = $schedules, args = NULLABLE)]
            fn $name<T, const BREAKS: u64>(bencher: Bencher, nullable: bool)
            where
                T: IntValue + Sync + Send,
                u32: AsPrimitive<T>,
            {
                run_schedule::<T, BREAKS>(bencher, nullable);
            }
        };
    }

    schedule_set!(schedule_0, SCHEDULES_0);
    schedule_set!(schedule_1, SCHEDULES_1);
    schedule_set!(schedule_2, SCHEDULES_2);
    schedule_set!(schedule_3, SCHEDULES_3);

    /// The reference: one pass over the array per statistic.
    #[divan::bench(types = [u8, u16, u32, u64, i64], args = NULLABLE)]
    fn separate<T>(bencher: Bencher, nullable: bool)
    where
        T: IntValue + Sync + Send,
        u32: AsPrimitive<T>,
    {
        bench::<T, _>(bencher, nullable, |v: &[T], m: &Mask| {
            (
                accumulate(v, m, Sum::new()),
                accumulate(v, m, MinMax::new()),
                accumulate(v, m, CommonBits::new()),
                accumulate(v, m, BitWidthHistogram::new()),
                accumulate(v, m, DeltaRange::new()),
                accumulate(v, m, Sorted::new()),
                accumulate(v, m, RunCount::new()),
            )
        });
    }
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}
