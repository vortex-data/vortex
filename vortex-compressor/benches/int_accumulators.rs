// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The cost of composing integer statistics into one fused pass, against separate passes.

use mimalloc::MiMalloc;

#[cfg(not(codspeed))]
#[divan::bench_group]
mod benchmarks {
    use std::hash::Hash;
    use std::sync::LazyLock;

    use divan::Bencher;
    use divan::black_box;
    use divan::counter::BytesCount;
    use num_traits::AsPrimitive;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::min_max::min_max;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::primitive::NativeValue;
    use vortex_array::dtype::IntegerPType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_compressor::stats::accumulator::BitWidthHistogram;
    use vortex_compressor::stats::accumulator::CommonBits;
    use vortex_compressor::stats::accumulator::DeltaRange;
    use vortex_compressor::stats::accumulator::Distinct;
    use vortex_compressor::stats::accumulator::EACH;
    use vortex_compressor::stats::accumulator::FUSED;
    use vortex_compressor::stats::accumulator::IntValue;
    use vortex_compressor::stats::accumulator::MinMax;
    use vortex_compressor::stats::accumulator::RunCount;
    use vortex_compressor::stats::accumulator::Schedule;
    use vortex_compressor::stats::accumulator::Sorted;
    use vortex_compressor::stats::accumulator::Sum;
    use vortex_compressor::stats::accumulator::accumulate;
    use vortex_compressor::stats::accumulator::compute;
    use vortex_compressor::stats::accumulator::groups;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    /// The uncompressed size of each benchmarked array, matching a writer chunk.
    const BYTES: usize = 4 << 20;

    /// The number of `T` values in [`BYTES`].
    fn len<T>() -> usize {
        BYTES / size_of::<T>()
    }

    /// Runs of up to 64 equal values drawn from 1024 distinct values, from a seeded xorshift.
    fn generate<T>() -> Buffer<T>
    where
        T: Copy + 'static,
        u32: AsPrimitive<T>,
    {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 32) as u32
        };
        let len = len::<T>();
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

    /// Every value valid, or roughly one in ten values null.
    const NULLABLE: [bool; 2] = [false, true];

    fn bench<T, R>(bencher: Bencher, nullable: bool, f: impl Fn(&[T], &Mask) -> R + Sync)
    where
        T: Copy + Sync + Send + 'static,
        u32: AsPrimitive<T>,
    {
        let values = generate::<T>();
        let mask = if nullable {
            Mask::from_iter((0..len::<T>()).map(|i| !(i * 2_654_435_761).is_multiple_of(10)))
        } else {
            Mask::new_true(len::<T>())
        };
        bencher
            .counter(BytesCount::new(BYTES))
            .bench(|| f(black_box(&values), black_box(&mask)));
    }

    macro_rules! int_bench {
        ($name:ident, | $values:ident, $mask:ident | $body:expr) => {
            #[divan::bench(types = [u8, u16, u32, u64, i64], args = NULLABLE)]
            fn $name<T>(bencher: Bencher, nullable: bool)
            where
                T: IntValue + Sync + Send,
                NativeValue<T>: Eq + Hash,
                u32: AsPrimitive<T>,
            {
                bench::<T, _>(bencher, nullable, |$values: &[T], $mask: &Mask| $body);
            }
        };
    }

    // Each statistic alone.
    int_bench!(min_max_only, |v, m| accumulate(v, m, MinMax::new()));
    int_bench!(run_count_only, |v, m| accumulate(v, m, RunCount::new()));
    int_bench!(sorted_only, |v, m| accumulate(v, m, Sorted::new()));
    int_bench!(distinct_only, |v, m| accumulate(
        v,
        m,
        Distinct::new(T::zero(), 1023u32.as_(), len::<T>())
    ));

    int_bench!(sum_only, |v, m| accumulate(v, m, Sum::new()));
    int_bench!(common_bits_only, |v, m| accumulate(v, m, CommonBits::new()));
    int_bench!(bit_width_histogram_only, |v, m| accumulate(
        v,
        m,
        BitWidthHistogram::new()
    ));
    int_bench!(delta_range_only, |v, m| accumulate(v, m, DeltaRange::new()));

    // Compositions: a plain tuple runs one loop per statistic over each block.
    int_bench!(fused_min_max_runs, |v, m| accumulate(
        v,
        m,
        (MinMax::new(), RunCount::new())
    ));
    int_bench!(chunk_fused_min_max_runs, |v, m| accumulate(
        v,
        m,
        Schedule::<_, FUSED>::new((MinMax::new(), RunCount::new()))
    ));
    int_bench!(fused_min_max_runs_sorted, |v, m| accumulate(
        v,
        m,
        (MinMax::new(), RunCount::new(), Sorted::new())
    ));
    int_bench!(fused_all, |v, m| accumulate(
        v,
        m,
        (
            MinMax::new(),
            RunCount::new(),
            Sorted::new(),
            Distinct::new(T::zero(), 1023u32.as_(), len::<T>()),
        )
    ));

    /// The seven statistics the integer schemes use, in the order the schedules below group them.
    #[allow(clippy::type_complexity)]
    fn compressor_set<T: IntValue>() -> (
        MinMax<T>,
        Sum<T>,
        CommonBits<T>,
        RunCount<T>,
        Sorted<T>,
        BitWidthHistogram<T>,
        DeltaRange<T>,
    ) {
        (
            MinMax::new(),
            Sum::new(),
            CommonBits::new(),
            RunCount::new(),
            Sorted::new(),
            BitWidthHistogram::new(),
            DeltaRange::new(),
        )
    }

    /// The cheap reductions in one loop, then one loop each.
    const GROUPED: u64 = groups(&[3, 4, 5, 6]);

    /// The cheap reductions in one loop, the two neighbour comparisons in another, then one loop
    /// each, chosen from the `fusion_search` affinities with every value valid.
    const PLANNED: u64 = groups(&[3, 5, 6]);

    // The same seven statistics in different schedules, which changes only the loops: one loop
    // each over every L1-sized block, one loop for all, groups in between, and separate passes
    // that read the array once per statistic.
    int_bench!(blocked_compressor_set, |v, m| accumulate(
        v,
        m,
        Schedule::<_, EACH>::new(compressor_set::<T>())
    ));
    int_bench!(chunk_fused_compressor_set, |v, m| accumulate(
        v,
        m,
        Schedule::<_, FUSED>::new(compressor_set::<T>())
    ));
    int_bench!(grouped_compressor_set, |v, m| accumulate(
        v,
        m,
        Schedule::<_, GROUPED>::new(compressor_set::<T>())
    ));
    // With nulls, fusing measured no better, so the plan runs one loop each.
    int_bench!(planned_compressor_set, |v, m| if m.all_true() {
        accumulate(v, m, Schedule::<_, PLANNED>::new(compressor_set::<T>()))
    } else {
        accumulate(v, m, Schedule::<_, EACH>::new(compressor_set::<T>()))
    });
    int_bench!(erased_grouped_compressor_set, |v, m| compute(
        v,
        m,
        Schedule::<_, GROUPED>::new(compressor_set::<T>())
    ));
    int_bench!(separate_compressor_set, |v, m| (
        accumulate(v, m, MinMax::new()),
        accumulate(v, m, RunCount::new()),
        accumulate(v, m, Sorted::new()),
        accumulate(v, m, Sum::new()),
        accumulate(v, m, CommonBits::new()),
        accumulate(v, m, BitWidthHistogram::new()),
        accumulate(v, m, DeltaRange::new()),
    ));

    // The cheap statistics alone: fused in one loop, one loop each per block, and separate passes.
    int_bench!(fused_cheap, |v, m| accumulate(
        v,
        m,
        Schedule::<_, FUSED>::new((MinMax::new(), Sum::new(), CommonBits::new()))
    ));
    int_bench!(blocked_cheap, |v, m| accumulate(
        v,
        m,
        Schedule::<_, EACH>::new((MinMax::new(), Sum::new(), CommonBits::new()))
    ));
    int_bench!(separate_cheap, |v, m| (
        accumulate(v, m, MinMax::new()),
        accumulate(v, m, Sum::new()),
        accumulate(v, m, CommonBits::new()),
    ));

    // The same statistics as separate passes.
    int_bench!(separate_min_max_runs_sorted, |v, m| (
        accumulate(v, m, MinMax::new()),
        accumulate(v, m, RunCount::new()),
        accumulate(v, m, Sorted::new()),
    ));

    /// The bit width histogram that the bit packing scheme computes in its own pass.
    #[divan::bench(types = [u8, u16, u32, u64, i64])]
    fn fastlanes_bit_width_histogram<T>(bencher: Bencher)
    where
        T: IntegerPType + Sync + Send,
        u32: AsPrimitive<T>,
    {
        let array = PrimitiveArray::new(generate::<T>(), Validity::NonNullable);
        bencher
            .counter(BytesCount::new(BYTES))
            .with_inputs(|| SESSION.create_execution_ctx())
            .bench_refs(|ctx| {
                vortex_fastlanes::bitpack_compress::bit_width_histogram(array.as_view(), ctx)
            });
    }

    /// The hand-written min/max kernel in `vortex-array`, on a fresh array each iteration so its
    /// result is not cached.
    #[divan::bench(types = [u8, u16, u32, u64, i64])]
    fn vortex_array_min_max<T>(bencher: Bencher)
    where
        T: IntegerPType + Sync + Send,
        u32: AsPrimitive<T>,
    {
        let values = generate::<T>();
        bencher
            .counter(BytesCount::new(BYTES))
            .with_inputs(|| {
                (
                    PrimitiveArray::new(values.clone(), Validity::NonNullable).into_array(),
                    SESSION.create_execution_ctx(),
                )
            })
            .bench_refs(|(array, ctx)| min_max(array, ctx, NumericalAggregateOpts::default()));
    }
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}
