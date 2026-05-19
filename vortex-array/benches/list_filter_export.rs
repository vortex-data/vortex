// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end `ListArray` filter → export benchmarks targeting the statpopgen `GT` column
//! shape: a list of `Int8` genotypes per row.
//!
//! The `FilterReduce` impl for `List` returns a `ListViewArray` whose elements are the
//! original (uncompacted) buffer, sidestepping the `FilterKernel`'s `O(elements_len)` mask
//! scan + per-element copy. The export path then either does zero-copy primitive transfer
//! (DuckDB / Arrow for fixed-width canonical children) or prunes if the unreachable fraction
//! is large enough to pay back the rebuild. Without `FilterReduce`, the kernel eagerly
//! materialises a compact elements buffer even for destinations that wouldn't read the
//! unreferenced portion anyway.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use arrow_schema::DataType;
use arrow_schema::Field;
use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

/// 2048 rows × 4096 `Int8` genotypes per row ≈ 8 MiB elements — representative of one
/// statpopgen chunk.
const NUM_LISTS: usize = 2048;
const LIST_SIZE: usize = 4096;

fn make_gt_list() -> ListArray {
    let total = NUM_LISTS * LIST_SIZE;
    // Genotype values: 0, 1, or 2. We don't bother randomising; the structure is what matters.
    let elements_data: Vec<i8> = (0..total).map(|i| (i % 3) as i8).collect();
    let elements = PrimitiveArray::from_iter(elements_data).into_array();
    let offsets: Vec<u32> = (0..=NUM_LISTS).map(|i| (i * LIST_SIZE) as u32).collect();
    ListArray::try_new(
        elements,
        PrimitiveArray::from_iter(offsets).into_array(),
        Validity::NonNullable,
    )
    .unwrap()
}

/// Build a sparse selection mask: keep the first `kept` rows by setting their bits to true.
fn sparse_mask(kept: usize) -> Mask {
    let mut bits: Vec<bool> = vec![false; NUM_LISTS];
    for i in 0..kept.min(NUM_LISTS) {
        bits[i] = true;
    }
    Mask::from(BitBuffer::from(bits.as_slice()))
}

fn export_arrow_list(array: ArrayRef) {
    let inner = Field::new("item", DataType::Int8, false);
    let dt = DataType::List(inner.into());
    let field = Field::new("v", dt, false);
    LEGACY_SESSION
        .arrow()
        .execute_arrow(
            array,
            Some(&field),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
}

const KEPT_ROWS: [usize; 5] = [1, 5, 50, 500, 1024];

#[divan::bench(args = KEPT_ROWS)]
fn filter_then_export(bencher: Bencher, kept: usize) {
    let list = make_gt_list();
    let mask = sparse_mask(kept);
    bencher
        .with_inputs(|| (list.clone().into_array(), mask.clone()))
        .bench_values(|(a, m)| {
            let filtered = a.filter(m).unwrap();
            export_arrow_list(filtered);
        });
}
