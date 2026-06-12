// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const LEN: usize = 65_536;

/// Fragmented (alternating) mask: the worst case for the generic per-run builder this kernel
/// replaces. The branchless bitmap blend is mask-shape-independent, so one shape suffices.
fn mask() -> Mask {
    Mask::from_iter((0..LEN).map(|i| i.is_multiple_of(2)))
}

#[divan::bench]
fn nonnull(bencher: Bencher) {
    let if_true = BoolArray::from_iter((0..LEN).map(|i| i.is_multiple_of(2))).into_array();
    let if_false = BoolArray::from_iter((0..LEN).map(|i| i.is_multiple_of(3))).into_array();
    run(bencher, if_true, if_false);
}

#[divan::bench]
fn nullable(bencher: Bencher) {
    let if_true = BoolArray::from_iter(
        (0..LEN).map(|i| (!i.is_multiple_of(7)).then_some(i.is_multiple_of(2))),
    )
    .into_array();
    let if_false = BoolArray::from_iter(
        (0..LEN).map(|i| (!i.is_multiple_of(5)).then_some(i.is_multiple_of(3))),
    )
    .into_array();
    run(bencher, if_true, if_false);
}

fn run(bencher: Bencher, if_true: ArrayRef, if_false: ArrayRef) {
    let mask = mask();
    bencher
        .with_inputs(|| {
            (
                if_true.clone(),
                if_false.clone(),
                mask.clone().into_array(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(t, f, m, ctx)| {
            m.zip(t.clone(), f.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap();
        });
}
