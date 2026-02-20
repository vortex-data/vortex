// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for ListView rebuild across different element types and scenarios.

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::ListViewRebuildMode;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

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

fn make_i8_lv(num_lists: usize, list_size: usize, step: usize) -> ListViewArray {
    let element_count = step * num_lists + list_size;
    let elements = PrimitiveArray::from_iter((0..element_count).map(|i| i as i8)).into_array();
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

// ── i32 with varied list sizes ───────────────────────────────────────────────
const LIST_SIZES: &[usize] = &[512, 2048];

#[divan::bench(args = LIST_SIZES)]
fn i32_varied_list_sizes(bencher: Bencher, list_size: usize) {
    let lv = make_primitive_lv(100, list_size, list_size);
    bencher
        .with_inputs(|| &lv)
        .bench_refs(|lv| lv.rebuild(ListViewRebuildMode::MakeZeroCopyToList).unwrap());
}

// ── i8 with 65K-element lists ─────────────────────────────────────────────────
#[divan::bench]
fn i8_large_lists(bencher: Bencher) {
    let lv = make_i8_lv(10, 65_536, 65_536);
    bencher
        .with_inputs(|| &lv)
        .bench_refs(|lv| lv.rebuild(ListViewRebuildMode::MakeZeroCopyToList).unwrap());
}

// ── i32 with 8-element overlapping lists ──────────────────────────────────────
#[divan::bench]
fn i32_small_overlapping(bencher: Bencher) {
    let lv = make_primitive_lv(100, 8, 1);
    bencher
        .with_inputs(|| &lv)
        .bench_refs(|lv| lv.rebuild(ListViewRebuildMode::MakeZeroCopyToList).unwrap());
}

// ── VarBinView: variable-width elements ──────────────────────────────────────
#[divan::bench]
fn varbinview_rebuild(bencher: Bencher) {
    let lv = make_varbinview_lv(100, 1_024, 1_024);
    bencher
        .with_inputs(|| &lv)
        .bench_refs(|lv| lv.rebuild(ListViewRebuildMode::MakeZeroCopyToList).unwrap());
}

// ── Struct{i32, f64}: struct elements ─────────────────────────────────────────
#[divan::bench]
fn struct_rebuild(bencher: Bencher) {
    let lv = make_struct_lv(1_000, 1_024, 1_024);
    bencher
        .with_inputs(|| &lv)
        .bench_refs(|lv| lv.rebuild(ListViewRebuildMode::MakeZeroCopyToList).unwrap());
}

// ── FixedSizeList<i32, 64>: FSL elements ─────────────────────────────────────
#[divan::bench]
fn fsl_rebuild(bencher: Bencher) {
    let num_lists = 10;
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
        .with_inputs(|| &lv)
        .bench_refs(|lv| lv.rebuild(ListViewRebuildMode::MakeZeroCopyToList).unwrap());
}

// ── List<i32>: nested list elements ───────────────────────────────────────────
#[divan::bench]
fn list_i32_nested(bencher: Bencher) {
    let num_lists = 10;
    let list_size = 128;
    let elem_count = num_lists * list_size + list_size;
    let inner_list_size = 8;
    let values = PrimitiveArray::from_iter(0..(elem_count * inner_list_size) as i32).into_array();
    let inner_offsets: Buffer<u32> = (0..=elem_count)
        .map(|i| (i * inner_list_size) as u32)
        .collect();
    let elements =
        ListArray::new(values, inner_offsets.into_array(), Validity::NonNullable).into_array();
    let offsets: Buffer<u32> = (0..num_lists).map(|i| (i * list_size) as u32).collect();
    let sizes: Buffer<u32> = std::iter::repeat_n(list_size as u32, num_lists).collect();
    let lv = ListViewArray::new(
        elements,
        offsets.into_array(),
        sizes.into_array(),
        Validity::NonNullable,
    );
    bencher
        .with_inputs(|| &lv)
        .bench_refs(|lv| lv.rebuild(ListViewRebuildMode::MakeZeroCopyToList).unwrap());
}
