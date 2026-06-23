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

const LEN: usize = 65_536;

#[divan::bench(args = [MaskShape::Fragmented, MaskShape::Block, MaskShape::Sparse, MaskShape::Dense])]
fn nonnull(bencher: Bencher, shape: MaskShape) {
    run(
        bencher,
        list_view(0, false),
        list_view(1_000_000, false),
        shape,
    );
}

#[divan::bench(args = [MaskShape::Fragmented, MaskShape::Block, MaskShape::Sparse, MaskShape::Dense])]
fn nullable(bencher: Bencher, shape: MaskShape) {
    run(
        bencher,
        list_view(0, true),
        list_view(1_000_000, true),
        shape,
    );
}

fn run(bencher: Bencher, if_true: ArrayRef, if_false: ArrayRef, shape: MaskShape) {
    let mask = shape.mask(LEN);
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

#[derive(Clone, Copy, Debug)]
enum MaskShape {
    Fragmented,
    Block,
    Sparse,
    Dense,
}

impl MaskShape {
    fn mask(self, len: usize) -> Mask {
        match self {
            MaskShape::Fragmented => Mask::from_iter((0..len).map(|i| i.is_multiple_of(2))),
            MaskShape::Block => Mask::from_iter((0..len).map(|i| (i / 128).is_multiple_of(2))),
            MaskShape::Sparse => Mask::from_iter((0..len).map(|i| i.is_multiple_of(10))),
            MaskShape::Dense => Mask::from_iter((0..len).map(|i| !i.is_multiple_of(10))),
        }
    }
}
