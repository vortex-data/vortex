// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A/B benchmark for the "lift validity into MaskedArray" experiment.
//!
//! Each case builds nullable (50% null) primitive arrays and runs them through
//! `execute::<Canonical>`, which triggers optimization. On the lift branch a nullable primitive is
//! rewritten into a `MaskedArray` wrapper; on the baseline it stays a primitive with embedded
//! validity. This isolates the wrapper-indirection cost.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const ARRAY_SIZE: usize = 65_536;

fn nullable_i64(seed: i64) -> vortex_array::ArrayRef {
    PrimitiveArray::from_option_iter(
        (0..ARRAY_SIZE as i64).map(|i| if i % 2 == 0 { Some(i + seed) } else { None }),
    )
    .into_array()
}

#[divan::bench]
fn canonicalize_nullable(bencher: Bencher) {
    let arr = nullable_i64(0);
    let session = VortexSession::empty();
    bencher
        .with_inputs(|| (arr.clone(), session.create_execution_ctx()))
        .bench_values(|(arr, mut ctx)| arr.execute::<Canonical>(&mut ctx));
}

#[divan::bench]
fn add_nullable(bencher: Bencher) {
    let lhs = nullable_i64(0);
    let rhs = nullable_i64(7);
    let session = VortexSession::empty();
    bencher
        .with_inputs(|| (lhs.clone(), rhs.clone(), session.create_execution_ctx()))
        .bench_values(|(lhs, rhs, mut ctx)| {
            lhs.binary(rhs, Operator::Add)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
        });
}

#[divan::bench]
fn compare_nullable(bencher: Bencher) {
    let lhs = nullable_i64(0);
    let rhs = nullable_i64(7);
    let session = VortexSession::empty();
    bencher
        .with_inputs(|| (lhs.clone(), rhs.clone(), session.create_execution_ctx()))
        .bench_values(|(lhs, rhs, mut ctx)| {
            lhs.binary(rhs, Operator::Gte)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
        });
}
