// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "benchmark fixtures use indices that fit in the chosen widths"
)]

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_buffer::BufferMut;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

// Sized so the bench stays well under a few hundred microseconds under CodSpeed's instruction-count
// simulation, which runs ~10x the local walltime; the branchless value blend is still exercised.
const LEN: usize = 16_384;

/// Fragmented (alternating) mask: the worst case for the generic run/slice copy path this kernel
/// replaces. The branchless per-row blend is mask-shape-independent, so one shape suffices.
fn mask() -> Mask {
    Mask::from_iter((0..LEN).map(|i| i.is_multiple_of(2)))
}

#[divan::bench]
fn nonnull(bencher: Bencher) {
    let if_true = nonnull_array(0).into_array();
    let if_false = nonnull_array(1_000_000).into_array();
    run(bencher, if_true, if_false);
}

#[divan::bench]
fn nullable(bencher: Bencher) {
    let if_true = nullable_array(0, 7).into_array();
    let if_false = nullable_array(1_000_000, 5).into_array();
    run(bencher, if_true, if_false);
}

fn run(bencher: Bencher, if_true: vortex_array::ArrayRef, if_false: vortex_array::ArrayRef) {
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

fn nonnull_array(base: i64) -> PrimitiveArray {
    let mut values = BufferMut::<i64>::with_capacity(LEN);
    values.extend((0..LEN as i64).map(|i| base + i));
    PrimitiveArray::new(
        values.freeze(),
        vortex_array::validity::Validity::NonNullable,
    )
}

fn nullable_array(base: i64, null_every: usize) -> PrimitiveArray {
    PrimitiveArray::from_option_iter(
        (0..LEN as i64).map(|i| (!(i as usize).is_multiple_of(null_every)).then_some(base + i)),
    )
}
