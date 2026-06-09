// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use itertools::repeat_n;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::IntegerPType;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_runend::RunEnd;
use vortex_runend::compress::runend_encode;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const BENCH_ARGS: &[(usize, usize)] = &[
    (1000, 4),
    (1000, 16),
    (1000, 256),
    (10_000, 4),
    (10_000, 16),
    (10_000, 256),
    (10_000, 1024),
    (100_000, 4),
    (100_000, 16),
    (100_000, 256),
    (100_000, 1024),
    (100_000, 4096),
];

#[divan::bench(args = BENCH_ARGS)]
fn compress(bencher: Bencher, (length, run_step): (usize, usize)) {
    let values = PrimitiveArray::new(
        (0..length)
            .step_by(run_step)
            .flat_map(|idx| repeat_n(idx as u64, run_step))
            .collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    bencher
        .with_inputs(|| (&values, SESSION.create_execution_ctx()))
        .bench_refs(|(values, ctx)| runend_encode(values.as_view(), ctx));
}

#[divan::bench(types = [u8, u16, u32, u64], args = BENCH_ARGS)]
fn decompress<T: IntegerPType>(bencher: Bencher, (length, run_step): (usize, usize)) {
    let ends = (0..=length)
        .step_by(run_step)
        .map(|x| x as u64)
        .collect::<Buffer<_>>()
        .into_array();

    let values = (0..ends.len())
        .map(|x| T::from(x % T::max_value().to_usize().unwrap()).unwrap())
        .collect::<Buffer<_>>()
        .into_array();

    let run_end_array = RunEnd::new(ends, values, &mut SESSION.create_execution_ctx());
    let array = run_end_array.into_array();

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut execution_ctx)| {
            array
                .execute::<RecursiveCanonical>(&mut execution_ctx)
                .unwrap()
        });
}

#[divan::bench(args = BENCH_ARGS)]
#[expect(clippy::cast_possible_truncation)]
fn take_indices(bencher: Bencher, (length, run_step): (usize, usize)) {
    let values = PrimitiveArray::new(
        (0..length)
            .step_by(run_step)
            .flat_map(|idx| repeat_n(idx as u64, run_step))
            .collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    let source_array = PrimitiveArray::from_iter(0..(length as i32)).into_array();
    let mut encode_ctx = SESSION.create_execution_ctx();
    let (ends, values) = runend_encode(values.as_view(), &mut encode_ctx);
    let runend_array = RunEnd::try_new(ends.into_array(), values, &mut encode_ctx)
        .unwrap()
        .into_array();

    bencher
        .with_inputs(|| (&source_array, &runend_array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, execution_ctx)| {
            array
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        });
}

#[divan::bench(args = BENCH_ARGS)]
fn decompress_utf8(bencher: Bencher, (length, run_step): (usize, usize)) {
    let num_runs = length.div_ceil(run_step);
    let ends = (0..num_runs)
        .map(|i| ((i + 1) * run_step).min(length) as u64)
        .collect::<Buffer<_>>()
        .into_array();

    let values = VarBinViewArray::from_iter_str((0..num_runs).map(|i| format!("run_value_{i}")))
        .into_array();

    let run_end_array = RunEnd::new(ends, values, &mut SESSION.create_execution_ctx());
    let array = run_end_array.into_array();

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut execution_ctx)| {
            array
                .execute::<RecursiveCanonical>(&mut execution_ctx)
                .unwrap()
        });
}
