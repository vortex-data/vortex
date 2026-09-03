// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end benchmarks for filtering RunEnd arrays.
//!
//! The benchmarks compare production dispatch, direct take, the prior range scan, and the
//! sequential scan. The array-length and run-length matrix calibrates the dispatch threshold.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::expect_used)]

use std::fmt;
use std::ops::AddAssign;
use std::sync::LazyLock;

use divan::Bencher;
use num_traits::AsPrimitive;
use num_traits::NumCast;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::buffer_mut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_runend::_benchmarking::filter_run_end_sequential;
use vortex_runend::_benchmarking::take_indices_unchecked;
use vortex_runend::RunEnd;
use vortex_runend::RunEndArray;
use vortex_runend::RunEndArrayExt;
use vortex_runend::RunEndArraySlotsExt;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_runend::initialize(&session);
    session
});

#[derive(Clone, Copy)]
enum FilterStrategy {
    Dispatch,
    DirectTake,
    LegacyRunScan,
    SequentialRunScan,
}

impl fmt::Display for FilterStrategy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dispatch => formatter.write_str("dispatch"),
            Self::DirectTake => formatter.write_str("direct_take"),
            Self::LegacyRunScan => formatter.write_str("legacy_run_scan"),
            Self::SequentialRunScan => formatter.write_str("sequential_run_scan"),
        }
    }
}

#[derive(Clone, Copy)]
enum MaskPattern {
    Random,
    Clustered,
}

impl fmt::Display for MaskPattern {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Random => formatter.write_str("random"),
            Self::Clustered => formatter.write_str("clustered"),
        }
    }
}

#[derive(Clone, Copy)]
enum RunPattern {
    Uniform,
    Skewed,
}

impl fmt::Display for RunPattern {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uniform => formatter.write_str("uniform"),
            Self::Skewed => formatter.write_str("skewed"),
        }
    }
}

#[derive(Clone, Copy)]
enum ValuesShape {
    Primitive,
    Dictionary,
}

impl fmt::Display for ValuesShape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Primitive => formatter.write_str("primitive"),
            Self::Dictionary => formatter.write_str("dictionary"),
        }
    }
}

#[derive(Clone, Copy)]
struct FilterBenchArgs {
    strategy: FilterStrategy,
    length: usize,
    run_length: usize,
    density_percent: usize,
    mask_pattern: MaskPattern,
    run_pattern: RunPattern,
    values_shape: ValuesShape,
}

impl fmt::Display for FilterBenchArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}_len{}_run{}_density{}_{}_{}_{}",
            self.strategy,
            self.length,
            self.run_length,
            self.density_percent,
            self.mask_pattern,
            self.run_pattern,
            self.values_shape,
        )
    }
}

const fn filter_case(
    strategy: FilterStrategy,
    length: usize,
    run_length: usize,
    density_percent: usize,
    mask_pattern: MaskPattern,
    run_pattern: RunPattern,
    values_shape: ValuesShape,
) -> FilterBenchArgs {
    FilterBenchArgs {
        strategy,
        length,
        run_length,
        density_percent,
        mask_pattern,
        run_pattern,
        values_shape,
    }
}

fn filter_args() -> Vec<FilterBenchArgs> {
    let mut args = Vec::new();

    for run_length in [1, 4, 16, 64, 128, 256, 512] {
        for density_percent in [1, 50, 95] {
            for strategy in [
                FilterStrategy::LegacyRunScan,
                FilterStrategy::SequentialRunScan,
            ] {
                args.push(filter_case(
                    strategy,
                    65_536,
                    run_length,
                    density_percent,
                    MaskPattern::Random,
                    RunPattern::Uniform,
                    ValuesShape::Primitive,
                ));
            }
        }
    }

    for (length, run_length) in [(128, 4), (128, 64), (4_096, 4), (4_096, 64), (4_096, 256)] {
        for strategy in [
            FilterStrategy::LegacyRunScan,
            FilterStrategy::SequentialRunScan,
        ] {
            args.push(filter_case(
                strategy,
                length,
                run_length,
                50,
                MaskPattern::Random,
                RunPattern::Uniform,
                ValuesShape::Primitive,
            ));
        }
    }

    for run_length in [4, 64, 256] {
        for strategy in [
            FilterStrategy::LegacyRunScan,
            FilterStrategy::SequentialRunScan,
        ] {
            args.push(filter_case(
                strategy,
                65_536,
                run_length,
                50,
                MaskPattern::Random,
                RunPattern::Skewed,
                ValuesShape::Primitive,
            ));
        }
    }

    for run_length in [4, 256] {
        for strategy in [
            FilterStrategy::LegacyRunScan,
            FilterStrategy::SequentialRunScan,
        ] {
            args.push(filter_case(
                strategy,
                65_536,
                run_length,
                50,
                MaskPattern::Clustered,
                RunPattern::Uniform,
                ValuesShape::Primitive,
            ));
        }

        for density_percent in [1, 50] {
            for strategy in [FilterStrategy::Dispatch, FilterStrategy::DirectTake] {
                args.push(filter_case(
                    strategy,
                    65_536,
                    run_length,
                    density_percent,
                    MaskPattern::Random,
                    RunPattern::Uniform,
                    ValuesShape::Primitive,
                ));
            }
        }
    }

    for strategy in [
        FilterStrategy::LegacyRunScan,
        FilterStrategy::SequentialRunScan,
    ] {
        args.push(filter_case(
            strategy,
            65_536,
            4,
            50,
            MaskPattern::Random,
            RunPattern::Uniform,
            ValuesShape::Dictionary,
        ));
    }

    args
}

#[divan::bench(args = filter_args())]
fn filter_materialized(bencher: Bencher, args: FilterBenchArgs) {
    let array = run_end_array(args);

    bencher
        .with_inputs(|| {
            (
                array.clone(),
                filter_mask(args),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, mask, execution_ctx)| {
            filter_with_strategy(array, mask, args.strategy, execution_ctx)
                .expect("filter")
                .execute::<RecursiveCanonical>(execution_ctx)
                .expect("materialize")
        });
}

fn filter_with_strategy(
    array: &RunEndArray,
    mask: &Mask,
    strategy: FilterStrategy,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match strategy {
        FilterStrategy::Dispatch => array.clone().into_array().filter(mask.clone()),
        FilterStrategy::DirectTake => {
            let mask_values = mask
                .values()
                .vortex_expect("forced strategies require a non-trivial mask");
            take_indices_unchecked(
                array.as_view(),
                mask_values.indices(),
                &Validity::NonNullable,
                ctx,
            )
        }
        FilterStrategy::LegacyRunScan => filter_with_run_scan(array, mask, false, ctx),
        FilterStrategy::SequentialRunScan => filter_with_run_scan(array, mask, true, ctx),
    }
}

fn filter_with_run_scan(
    array: &RunEndArray,
    mask: &Mask,
    use_sequential_scan: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mask_values = mask
        .values()
        .vortex_expect("forced strategies require a non-trivial mask");
    let primitive_run_ends = array.ends().clone().execute::<PrimitiveArray>(ctx)?;
    let (filtered_run_ends, values_mask) =
        match_each_unsigned_integer_ptype!(primitive_run_ends.ptype(), |P| {
            if use_sequential_scan {
                Ok(filter_run_end_sequential(
                    primitive_run_ends.as_slice::<P>(),
                    array.offset() as u64,
                    array.len() as u64,
                    mask_values.bit_buffer(),
                ))
            } else {
                legacy_filter_run_ends(
                    primitive_run_ends.as_slice::<P>(),
                    array.offset() as u64,
                    array.len() as u64,
                    mask_values.bit_buffer(),
                )
            }
        })?;
    let filtered_values = array.values().filter(values_mask)?;

    // SAFETY: Both scan implementations return one increasing end for each retained value.
    Ok(unsafe {
        RunEnd::new_unchecked(
            filtered_run_ends.into_array(),
            filtered_values,
            0,
            mask_values.true_count(),
        )
        .into_array()
    })
}

/// Preserves the previous per-run range-popcount implementation as a benchmark baseline.
fn legacy_filter_run_ends<R>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
) -> VortexResult<(PrimitiveArray, Mask)>
where
    R: NativePType + AddAssign + From<bool> + AsPrimitive<u64>,
{
    let mut filtered_run_ends = buffer_mut![R::zero(); run_ends.len()];
    let mut run_start = 0u64;
    let mut retained_run_count = 0;
    let mut filtered_end = R::zero();

    let values_mask = BitBuffer::collect_bool(run_ends.len(), |run_index| {
        let run_end = run_ends[run_index].as_() - offset;
        let run_end = run_end.min(length);
        let selected_in_run = mask.count_range(run_start as usize, run_end as usize);
        filtered_end += <R as NumCast>::from(selected_in_run)
            .vortex_expect("run popcount must fit in run-end native type");
        let retain_run = selected_in_run > 0;
        filtered_run_ends[retained_run_count] = filtered_end;
        retained_run_count += retain_run as usize;
        run_start = run_end;
        retain_run
    })
    .into();

    filtered_run_ends.truncate(retained_run_count);
    Ok((
        PrimitiveArray::new(filtered_run_ends, Validity::NonNullable),
        values_mask,
    ))
}

fn run_end_array(args: FilterBenchArgs) -> RunEndArray {
    let ends = run_ends(args);
    let run_count = ends.len();
    let values = match args.values_shape {
        ValuesShape::Primitive => {
            PrimitiveArray::from_iter((0..run_count).map(|run_index| run_index as u64)).into_array()
        }
        ValuesShape::Dictionary => DictArray::try_new(
            (0..run_count)
                .map(|run_index| (run_index % 16) as u8)
                .collect::<Buffer<_>>()
                .into_array(),
            PrimitiveArray::from_iter(0u64..16).into_array(),
        )
        .expect("dictionary")
        .into_array(),
    };
    RunEnd::new(ends, values, &mut SESSION.create_execution_ctx())
}

fn run_ends(args: FilterBenchArgs) -> ArrayRef {
    let mut run_ends = Vec::new();
    let mut run_end = 0usize;
    let mut run_index = 0usize;
    while run_end < args.length {
        let run_length = match args.run_pattern {
            RunPattern::Uniform => args.run_length,
            RunPattern::Skewed if run_index.is_multiple_of(2) => 1,
            RunPattern::Skewed => args.run_length.saturating_mul(2).saturating_sub(1),
        };
        run_end = run_end.saturating_add(run_length).min(args.length);
        run_ends.push(run_end);
        run_index += 1;
    }

    if args.length <= u8::MAX as usize {
        PrimitiveArray::from_iter(
            run_ends
                .into_iter()
                .map(|run_end| u8::try_from(run_end).vortex_expect("run end must fit in u8")),
        )
        .into_array()
    } else if args.length <= u16::MAX as usize {
        PrimitiveArray::from_iter(
            run_ends
                .into_iter()
                .map(|run_end| u16::try_from(run_end).vortex_expect("run end must fit in u16")),
        )
        .into_array()
    } else {
        PrimitiveArray::from_iter(
            run_ends
                .into_iter()
                .map(|run_end| u32::try_from(run_end).vortex_expect("run end must fit in u32")),
        )
        .into_array()
    }
}

fn filter_mask(args: FilterBenchArgs) -> Mask {
    let selected = args.length * args.density_percent / 100;
    let mut bits = vec![false; args.length];

    match args.mask_pattern {
        MaskPattern::Random => {
            bits[..selected].fill(true);
            bits.shuffle(&mut StdRng::seed_from_u64(0x5eed));
        }
        MaskPattern::Clustered => {
            let cluster_count = 8.min(selected.max(1));
            let cluster_span = args.length.div_ceil(cluster_count);
            let selected_per_cluster = selected.div_ceil(cluster_count);
            for cluster_index in 0..cluster_count {
                let begin = cluster_index * cluster_span;
                let end = (begin + selected_per_cluster).min(args.length);
                bits[begin..end].fill(true);
            }
        }
    }

    Mask::from_iter(bits)
}
