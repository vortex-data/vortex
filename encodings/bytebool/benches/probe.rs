// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_bytebool::ByteBool;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const LEN: usize = 65_536;
const ACCESSES: usize = 256;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

#[divan::bench]
fn one_off(bencher: Bencher) {
    let array = ByteBool::from_option_vec(
        (0..LEN)
            .map(|index| (index % 7 != 0).then_some(index % 2 == 0))
            .collect(),
    )
    .into_array();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for read in 0..ACCESSES {
                divan::black_box(array.execute_scalar((read * 7919) % LEN, ctx).unwrap());
            }
        });
}

#[divan::bench]
fn repeated(bencher: Bencher) {
    let array = ByteBool::from_option_vec(
        (0..LEN)
            .map(|index| (index % 7 != 0).then_some(index % 2 == 0))
            .collect(),
    )
    .into_array();
    bencher
        .with_inputs(|| (SESSION.create_execution_ctx(), array.repeated_probe()))
        .bench_refs(|(ctx, probe)| {
            for read in 0..ACCESSES {
                divan::black_box(probe.execute_scalar((read * 7919) % LEN, ctx).unwrap());
            }
        });
}
