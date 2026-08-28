// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks filtering lists with different element widths and outer-mask locality.

#![expect(clippy::cast_possible_truncation, clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_mask::Mask;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

// Keep wide cases large enough to exercise element-mask expansion while remaining inexpensive for
// CodSpeed's instruction-count simulation.
const LIST_COUNT: usize = 128;
const SHORT_LIST_WIDTH: usize = 32;
const WIDE_LIST_WIDTH: usize = 2_048;
const CLUSTERED_LIST_COUNT: usize = 8;

#[derive(Clone, Copy, Debug)]
enum MaskSetup {
    AllSelected,
    NoneSelected,
    Prefix,
    Suffix,
    EveryFifth,
    Clustered,
    Sparse,
    EdgeSpanning,
}

const MASK_SETUPS: &[MaskSetup] = &[
    MaskSetup::AllSelected,
    MaskSetup::NoneSelected,
    MaskSetup::Prefix,
    MaskSetup::Suffix,
    MaskSetup::EveryFifth,
    MaskSetup::Clustered,
    MaskSetup::Sparse,
    MaskSetup::EdgeSpanning,
];

fn primitive_list(width: usize) -> ArrayRef {
    let element_count = LIST_COUNT * width;
    let elements = PrimitiveArray::from_iter(0..element_count as u32).into_array();
    let offsets =
        Buffer::from_iter((0..=LIST_COUNT).map(|index| (index * width) as u32)).into_array();
    ListArray::try_new(elements, offsets, Validity::NonNullable)
        .unwrap()
        .into_array()
}

fn dictionary_list(width: usize) -> ArrayRef {
    let element_count = LIST_COUNT * width;
    let codes = Buffer::from_iter((0..element_count).map(|index| (index % 16) as u8)).into_array();
    let values = PrimitiveArray::from_iter(0..16u32).into_array();
    let elements = DictArray::try_new(codes, values).unwrap().into_array();
    let offsets =
        Buffer::from_iter((0..=LIST_COUNT).map(|index| (index * width) as u32)).into_array();
    ListArray::try_new(elements, offsets, Validity::NonNullable)
        .unwrap()
        .into_array()
}

fn string_list(width: usize) -> ArrayRef {
    let element_count = LIST_COUNT * width;
    let elements = VarBinViewArray::from_iter_str(
        (0..element_count).map(|index| format!("list-element-{index:08}")),
    )
    .into_array();
    let offsets =
        Buffer::from_iter((0..=LIST_COUNT).map(|index| (index * width) as u32)).into_array();
    ListArray::try_new(elements, offsets, Validity::NonNullable)
        .unwrap()
        .into_array()
}

fn selection_mask(setup: MaskSetup) -> Mask {
    match setup {
        MaskSetup::AllSelected => Mask::new_true(LIST_COUNT),
        MaskSetup::NoneSelected => Mask::new_false(LIST_COUNT),
        MaskSetup::Prefix => Mask::from_slices(LIST_COUNT, vec![(0, CLUSTERED_LIST_COUNT)]),
        MaskSetup::Suffix => Mask::from_slices(
            LIST_COUNT,
            vec![(LIST_COUNT - CLUSTERED_LIST_COUNT, LIST_COUNT)],
        ),
        MaskSetup::EveryFifth => Mask::from_indices(LIST_COUNT, (0..LIST_COUNT).step_by(5)),
        MaskSetup::Clustered => {
            let start = (LIST_COUNT - CLUSTERED_LIST_COUNT) / 2;
            Mask::from_slices(LIST_COUNT, vec![(start, start + CLUSTERED_LIST_COUNT)])
        }
        MaskSetup::Sparse => Mask::from_indices(LIST_COUNT, [LIST_COUNT / 2]),
        MaskSetup::EdgeSpanning => Mask::from_indices(LIST_COUNT, [0, LIST_COUNT - 1]),
    }
}

fn run(bencher: Bencher, array: ArrayRef, mask: Mask) {
    bencher
        .with_inputs(|| (array.clone(), mask.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mask, mut ctx)| {
            divan::black_box(
                array
                    .filter(mask)
                    .unwrap()
                    .execute::<RecursiveCanonical>(&mut ctx)
                    .unwrap(),
            );
        });
}

#[divan::bench(args = MASK_SETUPS)]
fn primitive_short(bencher: Bencher, setup: MaskSetup) {
    run(
        bencher,
        primitive_list(SHORT_LIST_WIDTH),
        selection_mask(setup),
    );
}

#[divan::bench(args = MASK_SETUPS)]
fn primitive_wide(bencher: Bencher, setup: MaskSetup) {
    run(
        bencher,
        primitive_list(WIDE_LIST_WIDTH),
        selection_mask(setup),
    );
}

#[divan::bench(args = MASK_SETUPS)]
fn dictionary_short(bencher: Bencher, setup: MaskSetup) {
    run(
        bencher,
        dictionary_list(SHORT_LIST_WIDTH),
        selection_mask(setup),
    );
}

#[divan::bench(args = MASK_SETUPS)]
fn dictionary_wide(bencher: Bencher, setup: MaskSetup) {
    run(
        bencher,
        dictionary_list(WIDE_LIST_WIDTH),
        selection_mask(setup),
    );
}

#[divan::bench(args = MASK_SETUPS)]
fn string_short(bencher: Bencher, setup: MaskSetup) {
    run(
        bencher,
        string_list(SHORT_LIST_WIDTH),
        selection_mask(setup),
    );
}

#[divan::bench(args = MASK_SETUPS)]
fn string_wide(bencher: Bencher, setup: MaskSetup) {
    run(bencher, string_list(WIDE_LIST_WIDTH), selection_mask(setup));
}
