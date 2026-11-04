// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::AllocProfiler;
use divan::Bencher;
use divan::black_box;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::count::Count;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::builder_with_capacity_in;
use vortex_session::VortexSession;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    divan::main();
}

// Repeated finalization isolates result construction from input scanning and accumulation.
// Allocation counts cover only the timed section; output buffers are preallocated.
#[divan::bench(consts = [false, true], args = ["count", "sum", "min_max"])]
fn finalize<const DIRECT: bool>(bencher: Bencher, name: &str) {
    let function: AggregateFnRef = match name {
        "count" => Count.bind(NumericalAggregateOpts::default()),
        "sum" => Sum.bind(NumericalAggregateOpts::default()),
        "min_max" => MinMax.bind(NumericalAggregateOpts::default()),
        _ => unreachable!(),
    };
    let session = VortexSession::empty();
    let input = PrimitiveArray::from_iter([1i64, 2, 3]).into_array();
    let dtype = function.return_dtype(input.dtype()).unwrap();
    bencher
        .with_inputs(|| {
            let mut ctx = session.create_execution_ctx();
            let mut accumulator = function.accumulator(input.dtype()).unwrap();
            accumulator.accumulate(&input, &mut ctx).unwrap();
            (
                accumulator,
                builder_with_capacity_in(&dtype, 1024, ctx.allocator()),
            )
        })
        .bench_refs(|(accumulator, builder)| {
            for _ in 0..1024 {
                if DIRECT {
                    accumulator.finalize_into(builder.as_mut()).unwrap();
                } else {
                    builder
                        .append_scalar(&accumulator.final_scalar().unwrap())
                        .unwrap();
                }
            }
            black_box(builder.len());
        });
}
