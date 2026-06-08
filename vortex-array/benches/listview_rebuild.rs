// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for ListView rebuild across different element types and scenarios.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::listview::ListViewRebuildMode;
use vortex_array::dtype::FieldNames;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

/// A shared session for the `ListView` rebuild benchmarks, used to create execution contexts.
static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn make_primitive_lv(num_lists: usize, list_size: usize, step: usize) -> ListViewArray {
    let element_count = step * num_lists + list_size;
    let elements = PrimitiveArray::from_iter(0..element_count as i32).into_array();
    let offsets: Buffer<u32> = (0..num_lists).map(|i| (i * step) as u32).collect();
    let sizes: Buffer<u32> = std::iter::repeat_n(list_size as u32, num_lists).collect();
    ListViewArray::new(
        elements,
        offsets.into_array(),
        sizes.into_array(),
        Validity::NonNullable,
    )
}

fn make_varbinview_lv(num_lists: usize, list_size: usize, step: usize) -> ListViewArray {
    let element_count = step * num_lists + list_size;
    let strings: Vec<String> = (0..element_count)
        .map(|i| {
            if i % 3 == 0 {
                format!("long-string-value-{i:06}")
            } else {
                format!("s{i}")
            }
        })
        .collect();
    let elements = VarBinViewArray::from_iter_str(strings.iter().map(|s| s.as_str())).into_array();
    let offsets: Buffer<u32> = (0..num_lists).map(|i| (i * step) as u32).collect();
    let sizes: Buffer<u32> = std::iter::repeat_n(list_size as u32, num_lists).collect();
    ListViewArray::new(
        elements,
        offsets.into_array(),
        sizes.into_array(),
        Validity::NonNullable,
    )
}

fn make_struct_lv(num_lists: usize, list_size: usize, step: usize) -> ListViewArray {
    let element_count = step * num_lists + list_size;
    let field_a = PrimitiveArray::from_iter(0..element_count as i32).into_array();
    let field_b = PrimitiveArray::from_iter((0..element_count).map(|i| i as f64)).into_array();
    let elements = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![field_a, field_b],
        element_count,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();

    let offsets: Buffer<u32> = (0..num_lists).map(|i| (i * step) as u32).collect();
    let sizes: Buffer<u32> = std::iter::repeat_n(list_size as u32, num_lists).collect();
    ListViewArray::new(
        elements,
        offsets.into_array(),
        sizes.into_array(),
        Validity::NonNullable,
    )
}

fn make_nested_list_lv(
    num_lists: usize,
    list_size: usize,
    inner_list_size: usize,
) -> ListViewArray {
    let elem_count = num_lists * list_size + list_size;
    let values = PrimitiveArray::from_iter(0..(elem_count * inner_list_size) as i32).into_array();
    let inner_offsets: Buffer<u32> = (0..=elem_count)
        .map(|i| (i * inner_list_size) as u32)
        .collect();
    let elements =
        ListArray::new(values, inner_offsets.into_array(), Validity::NonNullable).into_array();
    let offsets: Buffer<u32> = (0..num_lists).map(|i| (i * list_size) as u32).collect();
    let sizes: Buffer<u32> = std::iter::repeat_n(list_size as u32, num_lists).collect();
    ListViewArray::new(
        elements,
        offsets.into_array(),
        sizes.into_array(),
        Validity::NonNullable,
    )
}

#[divan::bench]
fn i32_small(bencher: Bencher) {
    let lv = make_primitive_lv(50, 32, 32);
    bencher
        .with_inputs(|| (&lv, SESSION.create_execution_ctx()))
        .bench_refs(|(lv, ctx)| {
            let rebuilt = lv
                .rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)
                .unwrap();
            rebuilt
                .elements()
                .clone()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn i32_small_overlapping(bencher: Bencher) {
    let lv = make_primitive_lv(50, 8, 1);
    bencher
        .with_inputs(|| (&lv, SESSION.create_execution_ctx()))
        .bench_refs(|(lv, ctx)| {
            let rebuilt = lv
                .rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)
                .unwrap();
            rebuilt
                .elements()
                .clone()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn varbinview_small(bencher: Bencher) {
    let lv = make_varbinview_lv(50, 32, 32);
    bencher
        .with_inputs(|| (&lv, SESSION.create_execution_ctx()))
        .bench_refs(|(lv, ctx)| {
            let rebuilt = lv
                .rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)
                .unwrap();
            rebuilt
                .elements()
                .clone()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn struct_small(bencher: Bencher) {
    let lv = make_struct_lv(50, 32, 32);
    bencher
        .with_inputs(|| (&lv, SESSION.create_execution_ctx()))
        .bench_refs(|(lv, ctx)| {
            let rebuilt = lv
                .rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)
                .unwrap();
            rebuilt
                .elements()
                .clone()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn i32_large(bencher: Bencher) {
    let lv = make_primitive_lv(50, 1_024, 1_024);
    bencher
        .with_inputs(|| (&lv, SESSION.create_execution_ctx()))
        .bench_refs(|(lv, ctx)| {
            let rebuilt = lv
                .rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)
                .unwrap();
            rebuilt
                .elements()
                .clone()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn varbinview_large(bencher: Bencher) {
    let lv = make_varbinview_lv(5, 1_024, 1_024);
    bencher
        .with_inputs(|| (&lv, SESSION.create_execution_ctx()))
        .bench_refs(|(lv, ctx)| {
            let rebuilt = lv
                .rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)
                .unwrap();
            rebuilt
                .elements()
                .clone()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn struct_large(bencher: Bencher) {
    let lv = make_struct_lv(25, 1_024, 1_024);
    bencher
        .with_inputs(|| (&lv, SESSION.create_execution_ctx()))
        .bench_refs(|(lv, ctx)| {
            let rebuilt = lv
                .rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)
                .unwrap();
            rebuilt
                .elements()
                .clone()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn fsl_large(bencher: Bencher) {
    let num_lists = 5;
    let list_size = 256;
    let fsl_count = num_lists * list_size + list_size;
    let inner = PrimitiveArray::from_iter((0..fsl_count * 64).map(|i| i as i32)).into_array();
    let elements =
        FixedSizeListArray::new(inner, 64, Validity::NonNullable, fsl_count).into_array();
    let offsets: Buffer<u32> = (0..num_lists).map(|i| (i * list_size) as u32).collect();
    let sizes: Buffer<u32> = std::iter::repeat_n(list_size as u32, num_lists).collect();
    let lv = ListViewArray::new(
        elements,
        offsets.into_array(),
        sizes.into_array(),
        Validity::NonNullable,
    );
    bencher
        .with_inputs(|| (&lv, SESSION.create_execution_ctx()))
        .bench_refs(|(lv, ctx)| {
            let rebuilt = lv
                .rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)
                .unwrap();
            rebuilt
                .elements()
                .clone()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn list_i32_large(bencher: Bencher) {
    let lv = make_nested_list_lv(2, 512, 2);
    bencher
        .with_inputs(|| (&lv, SESSION.create_execution_ctx()))
        .bench_refs(|(lv, ctx)| {
            let rebuilt = lv
                .rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)
                .unwrap();
            rebuilt
                .elements()
                .clone()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}
