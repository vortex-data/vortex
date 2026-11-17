// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end chunked filtering, including execution of the selected child values.
//! Cases are (chunk count, selected rows, contiguous selection) over 65,536 rows.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_array::test_harness::mark_decoded;
use vortex_buffer::BitBuffer;
use vortex_fastlanes::BitPackedData;
use vortex_mask::Mask;
use vortex_pco::Pco;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    check_nullable_chunks();
    divan::main();
}

fn check_nullable_chunks() {
    let mut values = Vec::new();
    let chunks = (0..64i32).map(|chunk| {
        let child = if chunk % 7 == 0 {
            vec![]
        } else {
            vec![Some(chunk), None, Some(chunk + 100)]
        };
        values.extend_from_slice(&child);
        PrimitiveArray::from_option_iter(child).into_array()
    });
    let array = ChunkedArray::from_iter(chunks).into_array();
    let masks = [
        Mask::from_indices(values.len(), [0, 1, 37, values.len() - 1]),
        Mask::from_iter((0..values.len()).map(|index| index % 2 == 0)),
    ];
    for mask in masks {
        assert_arrays_eq!(
            array.filter(mask.clone()).unwrap(),
            PrimitiveArray::from_option_iter(values.clone())
                .filter(mask)
                .unwrap(),
            &mut SESSION.create_execution_ctx()
        );
    }
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>();
    vortex_fastlanes::initialize(&session);
    vortex_pco::initialize(&session);
    session
});

const LEN: usize = 65_536;
// Exercise each child's sparse filtering, dense filtering, and contiguous selection paths.
const CASES: &[(usize, usize, bool)] = &[(16, 64, false), (16, 32_768, false), (16, 64, true)];

// Chunk routing is independent of the child encoding, so cover it with primitive children.
const ROUTING_CASES: &[(usize, usize, bool)] = &[
    (16, 62_259, false),
    (16, 32_768, true),
    (1_024, 16, false),
    (1_024, 32_768, false),
];

fn mask_buffer(selected: usize, contiguous: bool) -> BitBuffer {
    let indices = if contiguous {
        let start = (LEN - selected) / 2;
        (start..start + selected).collect::<Vec<_>>()
    } else {
        let mut indices = (0..LEN).collect::<Vec<_>>();
        indices.shuffle(&mut StdRng::seed_from_u64(42));
        indices.truncate(selected);
        indices.sort_unstable();
        indices
    };
    BitBuffer::from_indices(LEN, indices)
}

fn primitive(len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len).map(|index| u32::try_from(index / 16).unwrap()))
}

fn bench_filter(
    bencher: Bencher,
    case: (usize, usize, bool),
    make_chunk: impl Fn(usize) -> ArrayRef,
) {
    let (chunks, selected, contiguous) = case;
    let array = ChunkedArray::from_iter((0..chunks).map(|_| make_chunk(LEN / chunks))).into_array();
    mark_decoded(&array);
    let buffer = mask_buffer(selected, contiguous);
    let mut ctx = SESSION.create_execution_ctx();
    let expected = array
        .clone()
        .execute::<RecursiveCanonical>(&mut ctx)
        .unwrap()
        .0
        .into_array()
        .filter(Mask::from_buffer(buffer.clone()))
        .unwrap();
    assert_arrays_eq!(
        array.filter(Mask::from_buffer(buffer.clone())).unwrap(),
        expected,
        &mut ctx
    );

    bencher
        .with_inputs(|| {
            (
                Mask::from_buffer(buffer.clone()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(mask, mut ctx)| {
            array
                .filter(mask)
                .unwrap()
                .execute::<RecursiveCanonical>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench(args = CASES)]
fn primitive_values(bencher: Bencher, case: (usize, usize, bool)) {
    bench_filter(bencher, case, |len| primitive(len).into_array());
}

#[divan::bench(args = ROUTING_CASES)]
fn chunk_routing(bencher: Bencher, case: (usize, usize, bool)) {
    bench_filter(bencher, case, |len| primitive(len).into_array());
}

#[divan::bench(args = CASES)]
fn string_views(bencher: Bencher, case: (usize, usize, bool)) {
    bench_filter(bencher, case, |len| {
        VarBinViewArray::from_iter_str((0..len).map(|index| format!("a long string value {index}")))
            .into_array()
    });
}

#[divan::bench(args = CASES)]
fn dictionary(bencher: Bencher, case: (usize, usize, bool)) {
    bench_filter(bencher, case, |len| {
        let codes =
            PrimitiveArray::from_iter((0..len).map(|index| u8::try_from(index % 16).unwrap()));
        DictArray::new(
            codes.into_array(),
            PrimitiveArray::from_iter(0..16u32).into_array(),
        )
        .into_array()
    });
}

#[divan::bench(args = CASES)]
fn run_end(bencher: Bencher, case: (usize, usize, bool)) {
    bench_filter(bencher, case, |len| {
        let ends =
            PrimitiveArray::from_iter((1..=len / 16).map(|run| u32::try_from(run * 16).unwrap()));
        let values =
            PrimitiveArray::from_iter((0..len / 16).map(|run| u32::try_from(run).unwrap()));
        RunEnd::try_new(
            ends.into_array(),
            values.into_array(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .into_array()
    });
}

#[divan::bench(args = CASES)]
fn bitpacked(bencher: Bencher, case: (usize, usize, bool)) {
    bench_filter(bencher, case, |len| {
        BitPackedData::encode(
            &primitive(len).into_array(),
            8,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .into_array()
    });
}

#[divan::bench(args = CASES)]
fn pco(bencher: Bencher, case: (usize, usize, bool)) {
    bench_filter(bencher, case, |len| {
        Pco::from_primitive(
            primitive(len).as_view(),
            8,
            1_024,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .into_array()
    });
}
