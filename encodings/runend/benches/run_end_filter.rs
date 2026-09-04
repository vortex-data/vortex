// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for the run-end filter inner loop.
//!
//! `filter_run_end` retains the historical production benchmark. The two materialization
//! benchmarks compare the range and sequential implementations on each CPU feature runner.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::cast_precision_loss)]
#![expect(clippy::cast_sign_loss)]
#![expect(clippy::expect_used)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
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
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_runend::_benchmarking::filter_run_end_primitive;
use vortex_runend::_benchmarking::filter_run_end_ranges;
use vortex_runend::_benchmarking::filter_run_end_sequential;
use vortex_runend::RunEnd;
use vortex_runend::RunEndArray;
use vortex_runend::RunEndArrayExt;
use vortex_runend::RunEndArraySlotsExt;
use vortex_session::VortexSession;

// Filtering allocates run ends and value masks inside the timed region. Use the same allocator on
// each benchmark runner so that allocator differences do not obscure the scan comparison.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_runend::initialize(&session);
    session
});

#[derive(Clone, Copy)]
struct FilterBenchArgs {
    length: usize,
    run_length: usize,
    density: f64,
}

impl fmt::Display for FilterBenchArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "len={}_run={}_density={:.1}",
            self.length, self.run_length, self.density
        )
    }
}

const FILTER_ARGS: &[FilterBenchArgs] = &[
    FilterBenchArgs {
        length: 4_096,
        run_length: 16,
        density: 0.1,
    },
    FilterBenchArgs {
        length: 4_096,
        run_length: 16,
        density: 0.5,
    },
    FilterBenchArgs {
        length: 4_096,
        run_length: 16,
        density: 0.9,
    },
    FilterBenchArgs {
        length: 16_384,
        run_length: 16,
        density: 0.1,
    },
    FilterBenchArgs {
        length: 16_384,
        run_length: 16,
        density: 0.5,
    },
    FilterBenchArgs {
        length: 16_384,
        run_length: 16,
        density: 0.9,
    },
];

fn build_run_ends(length: usize, run_length: usize) -> Vec<u32> {
    (0..length.div_ceil(run_length))
        .map(|run_index| (((run_index + 1) * run_length).min(length)) as u32)
        .collect()
}

fn build_mask(length: usize, density: f64) -> BitBuffer {
    let selected = (length as f64 * density).round() as usize;
    let mut bits = vec![false; length];
    bits[..selected].fill(true);
    bits.shuffle(&mut StdRng::seed_from_u64(0x5eed));
    BitBuffer::from(bits)
}

#[divan::bench(args = FILTER_ARGS)]
fn filter_run_end(bencher: Bencher, args: FilterBenchArgs) {
    let run_ends = build_run_ends(args.length, args.run_length);
    let mask = build_mask(args.length, args.density);
    let length = args.length as u64;
    bencher
        .with_inputs(|| (run_ends.clone(), mask.clone()))
        .bench_refs(|(run_ends, mask)| {
            filter_run_end_primitive::<u32>(run_ends, 0, length, mask).expect("filter")
        });
}

#[derive(Clone, Copy)]
struct StrategyBenchArgs {
    length: usize,
    run_length: usize,
    density_percent: usize,
    shape: ValuesShape,
    offset: usize,
}

impl fmt::Display for StrategyBenchArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "len{}_run{}_density{}_{}_offset{}",
            self.length, self.run_length, self.density_percent, self.shape, self.offset
        )
    }
}

#[derive(Clone, Copy)]
enum ValuesShape {
    Primitive,
    Dictionary,
    Irregular,
}

impl fmt::Display for ValuesShape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Primitive => formatter.write_str("primitive"),
            Self::Dictionary => formatter.write_str("dictionary"),
            Self::Irregular => formatter.write_str("irregular"),
        }
    }
}

const STRATEGY_ARGS: &[StrategyBenchArgs] = &[
    strategy_case(65_536, 16, 50),
    strategy_case(65_536, 32, 50),
    strategy_case(65_536, 64, 1),
    strategy_case(65_536, 64, 50),
    strategy_case(65_536, 64, 95),
    strategy_case(65_536, 96, 50),
    strategy_case(65_536, 128, 50),
    strategy_case(65_536, 256, 50),
    strategy_case(65_536, 512, 50),
    strategy_case(128, 64, 50),
    strategy_case(4_096, 64, 50),
    StrategyBenchArgs {
        length: 65_536,
        run_length: 64,
        density_percent: 50,
        shape: ValuesShape::Irregular,
        offset: 37,
    },
    StrategyBenchArgs {
        length: 65_536,
        run_length: 64,
        density_percent: 50,
        shape: ValuesShape::Dictionary,
        offset: 0,
    },
    StrategyBenchArgs {
        length: 65_536,
        run_length: 128,
        density_percent: 50,
        shape: ValuesShape::Dictionary,
        offset: 0,
    },
];

const fn strategy_case(
    length: usize,
    run_length: usize,
    density_percent: usize,
) -> StrategyBenchArgs {
    StrategyBenchArgs {
        length,
        run_length,
        density_percent,
        shape: ValuesShape::Primitive,
        offset: 0,
    }
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = STRATEGY_ARGS, sample_size = 1)]
fn filter_materialized_range(bencher: Bencher, args: StrategyBenchArgs) {
    benchmark_filter(bencher, args, filter_run_end_ranges::<u32>);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = STRATEGY_ARGS, sample_size = 1)]
fn filter_materialized_sequential(bencher: Bencher, args: StrategyBenchArgs) {
    benchmark_filter(bencher, args, |run_ends, offset, length, mask| {
        Ok(filter_run_end_sequential(run_ends, offset, length, mask))
    });
}

fn benchmark_filter<F>(bencher: Bencher, args: StrategyBenchArgs, filter_run_ends: F)
where
    F: Fn(&[u32], u64, u64, &BitBuffer) -> VortexResult<(PrimitiveArray, Mask)>
        + Copy
        + Send
        + Sync
        + 'static,
{
    let array = run_end_array(args);
    let mask: Mask = build_mask(args.length, args.density_percent as f64 / 100.0).into();
    bencher
        .with_inputs(|| (array.clone(), mask.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(array, mask, execution_ctx)| {
            filter_with_strategy(array, mask, filter_run_ends, execution_ctx)
                .expect("filter")
                .execute::<RecursiveCanonical>(execution_ctx)
                .expect("materialize")
        });
}

fn run_end_array(args: StrategyBenchArgs) -> RunEndArray {
    let source_length = args.length + args.offset;
    let mut run_ends = match args.shape {
        ValuesShape::Primitive | ValuesShape::Dictionary => {
            build_run_ends(source_length, args.run_length)
        }
        ValuesShape::Irregular => build_irregular_run_ends(source_length, args.run_length),
    };
    let first_visible_run = run_ends.partition_point(|&run_end| run_end < args.offset as u32);
    run_ends.drain(..first_visible_run);
    let values = match args.shape {
        ValuesShape::Primitive | ValuesShape::Irregular => {
            PrimitiveArray::from_iter(0..run_ends.len() as u64).into_array()
        }
        ValuesShape::Dictionary => DictArray::try_new(
            (0..run_ends.len())
                .map(|run_index| (run_index % 16) as u8)
                .collect::<Buffer<_>>()
                .into_array(),
            PrimitiveArray::from_iter(0u64..16).into_array(),
        )
        .expect("dictionary")
        .into_array(),
    };
    RunEnd::try_new_offset_length(
        PrimitiveArray::from_iter(run_ends).into_array(),
        values,
        args.offset,
        args.length,
        &mut SESSION.create_execution_ctx(),
    )
    .expect("run-end array")
}

fn build_irregular_run_ends(length: usize, run_length: usize) -> Vec<u32> {
    let mut run_ends = Vec::new();
    let mut run_end = 0;
    let mut run_index = 0;
    while run_end < length {
        let next_run_length = if run_index % 2 == 0 {
            1
        } else {
            run_length * 2 - 1
        };
        run_end = (run_end + next_run_length).min(length);
        run_ends.push(run_end as u32);
        run_index += 1;
    }
    run_ends
}

fn filter_with_strategy<F>(
    array: &RunEndArray,
    mask: &Mask,
    filter_run_ends: F,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    F: Fn(&[u32], u64, u64, &BitBuffer) -> VortexResult<(PrimitiveArray, Mask)>,
{
    let mask_values = mask
        .values()
        .vortex_expect("benchmark mask must be non-trivial");
    let primitive_run_ends = array.ends().clone().execute::<PrimitiveArray>(ctx)?;
    let (filtered_run_ends, values_mask) = filter_run_ends(
        primitive_run_ends.as_slice::<u32>(),
        array.offset() as u64,
        array.len() as u64,
        mask_values.bit_buffer(),
    )?;
    let filtered_values = array.values().filter(values_mask)?;

    // SAFETY: Both scan functions return one increasing end for each retained value.
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
