// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DecimalDType;
use vortex_decimal_byte_parts::DecimalByteParts;
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

fn fixture(lower_count: usize) -> ArrayRef {
    let msp = PrimitiveArray::from_option_iter(
        (0..LEN).map(|index| (index % 7 != 0).then_some(index as i64)),
    )
    .into_array();
    let lower = (0..lower_count)
        .map(|part| {
            PrimitiveArray::from_iter((0..LEN).map(|index| (index + part) as u64)).into_array()
        })
        .collect();
    let dtype = DecimalDType::new(if lower_count <= 1 { 38 } else { 76 }, 0);
    DecimalByteParts::try_new_with_lower_parts(msp, lower, dtype)
        .unwrap()
        .into_array()
}

#[divan::bench(args = [0, 1, 3])]
fn one_off(bencher: Bencher, lower_count: usize) {
    let array = fixture(lower_count);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for read in 0..ACCESSES {
                divan::black_box(array.execute_scalar((read * 7919) % LEN, ctx).unwrap());
            }
        });
}

#[divan::bench(args = [0, 1, 3])]
fn repeated(bencher: Bencher, lower_count: usize) {
    let array = fixture(lower_count);
    bencher
        .with_inputs(|| (SESSION.create_execution_ctx(), array.repeated_probe()))
        .bench_refs(|(ctx, probe)| {
            for read in 0..ACCESSES {
                divan::black_box(probe.execute_scalar((read * 7919) % LEN, ctx).unwrap());
            }
        });
}
