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
use vortex_array::arrays::ListViewArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

// Smaller than the value-path benches: listview zip cost is dominated by element concatenation and
// per-list canonicalization, so a few thousand lists already exercises the select while keeping the
// benchmark well under a few hundred microseconds.
const LEN: usize = 8_192;

/// Fragmented (alternating) mask: the worst case for the per-element branch this kernel replaces.
/// The branchless chunked select is mask-shape-independent, so one shape suffices.
fn mask() -> Mask {
    Mask::from_iter((0..LEN).map(|i| i.is_multiple_of(2)))
}

#[divan::bench]
fn nonnull(bencher: Bencher) {
    run(bencher, list_view(0, false), list_view(1_000_000, false));
}

#[divan::bench]
fn nullable(bencher: Bencher) {
    run(bencher, list_view(0, true), list_view(1_000_000, true));
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

/// `LEN` single-element lists: `list[i] = [base + i]`. When `nullable`, every 7th list is null
/// (list-level validity backed by a `BoolArray`), exercising the `zip_validity` path.
fn list_view(base: i64, nullable: bool) -> ArrayRef {
    let mut elements = BufferMut::<i64>::with_capacity(LEN);
    elements.extend((0..LEN as i64).map(|i| base + i));
    let offsets: BufferMut<u64> = (0..LEN as u64).collect();
    let sizes: BufferMut<u64> = std::iter::repeat_n(1u64, LEN).collect();

    let validity = if nullable {
        Validity::Array(BoolArray::from_iter((0..LEN).map(|i| !i.is_multiple_of(7))).into_array())
    } else {
        Validity::NonNullable
    };

    ListViewArray::try_new(
        elements.freeze().into_array(),
        offsets.freeze().into_array(),
        sizes.freeze().into_array(),
        validity,
    )
    .unwrap()
    .into_array()
}
