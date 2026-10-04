// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The cost of composing integer statistics into one fused pass, against separate passes.

use mimalloc::MiMalloc;

#[cfg(not(codspeed))]
#[divan::bench_group(items_count = 64_000u32)]
mod benchmarks {
    use std::hash::Hash;
    use std::sync::LazyLock;

    use divan::Bencher;
    use divan::black_box;
    use num_traits::AsPrimitive;
    use num_traits::PrimInt;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::min_max::min_max;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::primitive::NativeValue;
    use vortex_array::dtype::IntegerPType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_compressor::stats::accumulator::Distinct;
    use vortex_compressor::stats::accumulator::MinMax;
    use vortex_compressor::stats::accumulator::RunCount;
    use vortex_compressor::stats::accumulator::Sorted;
    use vortex_compressor::stats::accumulator::accumulate;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    const LEN: usize = 64_000;

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
        let mut output = Vec::with_capacity(LEN);
        while output.len() < LEN {
            let value = next() % 1024;
            let run = (next() % 64).max(1) as usize;
            output.extend(std::iter::repeat_n(
                value.as_(),
                run.min(LEN - output.len()),
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
            Mask::from_iter((0..LEN).map(|i| !(i * 2_654_435_761).is_multiple_of(10)))
        } else {
            Mask::new_true(LEN)
        };
        bencher.bench(|| f(black_box(&values), black_box(&mask)));
    }

    macro_rules! int_bench {
        ($name:ident, | $values:ident, $mask:ident | $body:expr) => {
            #[divan::bench(types = [u8, u16, u32, u64, i64], args = NULLABLE)]
            fn $name<T>(bencher: Bencher, nullable: bool)
            where
                T: IntegerPType + PrimInt + Sync + Send,
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
        Distinct::new(T::zero(), 1023u32.as_(), LEN)
    ));

    // Fused compositions.
    int_bench!(fused_min_max_runs, |v, m| accumulate(
        v,
        m,
        (MinMax::new(), RunCount::new())
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
            Distinct::new(T::zero(), 1023u32.as_(), LEN),
        )
    ));

    // The same statistics as separate passes.
    int_bench!(separate_min_max_runs_sorted, |v, m| (
        accumulate(v, m, MinMax::new()),
        accumulate(v, m, RunCount::new()),
        accumulate(v, m, Sorted::new()),
    ));

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
