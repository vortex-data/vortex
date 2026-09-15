// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum_v2::sum_v2;
use vortex_array::arrays::PrimitiveArray;
use vortex_runend::RunEnd;
use vortex_session::VortexSession;

const LEN: usize = 2_048;
const RUN_LENGTH: usize = 64;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_runend::initialize(&session);
    session
});

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

fn runend_with_null_runs() -> ArrayRef {
    let ends =
        PrimitiveArray::from_iter((RUN_LENGTH..=LEN).step_by(RUN_LENGTH).map(|end| end as u64));
    let values = PrimitiveArray::from_option_iter(
        (0..ends.len()).map(|index| (index % 5 != 0).then_some(i32::try_from(index).unwrap())),
    );

    RunEnd::try_new(
        ends.into_array(),
        values.into_array(),
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap()
    .into_array()
}

#[divan::bench]
fn whole_array_sum_partially_valid(bencher: Bencher) {
    let array = runend_with_null_runs();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| sum_v2(&array, ctx).unwrap());
}
