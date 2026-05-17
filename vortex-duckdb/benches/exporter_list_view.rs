// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for exporting [`ListViewArray`] to DuckDB vectors.
//!
//! The listview exporter materialises the full `elements` child into a DuckDB vector on first
//! use and references it via per-row (offset, size) entries. After a selective filter/take it is
//! common for the `elements` child to contain large stretches that no view covers. These benches
//! exercise that path so we can quantify the cost of dragging the unreferenced elements through
//! the export and the savings from pruning them.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::validity::Validity;
use vortex_duckdb::SESSION;
use vortex_duckdb::duckdb::DataChunk;
use vortex_duckdb::duckdb::LogicalType;
use vortex_duckdb::exporter::ConversionCache;
use vortex_duckdb::exporter::new_array_exporter;

fn main() {
    divan::main();
}

const VECTOR_SIZE: usize = 2048;
/// Average list size across rows. Small enough to make per-list overhead matter.
const LIST_SIZE: usize = 8;

/// Build a primitive-element listview whose views cover only `referenced_fraction` of the
/// underlying elements buffer.
///
/// The unreferenced elements are interleaved with the referenced ranges to simulate the
/// post-filter case where each surviving row pulls its slice from the original elements buffer.
fn make_primitive_listview(referenced_fraction: f64) -> ListViewArray {
    let mut rng = StdRng::seed_from_u64(0);

    // Sum of all list sizes when fully dense.
    let referenced = VECTOR_SIZE * LIST_SIZE;
    let element_count = ((referenced as f64) / referenced_fraction).max(1.0) as usize;

    let elements = PrimitiveArray::from_iter(0i64..element_count as i64).into_array();

    let mut offsets: Vec<u32> = Vec::with_capacity(VECTOR_SIZE);
    let mut sizes: Vec<u32> = Vec::with_capacity(VECTOR_SIZE);
    let max_offset = element_count.saturating_sub(LIST_SIZE);
    for _ in 0..VECTOR_SIZE {
        let offset = rng.random_range(0..=max_offset.max(1));
        offsets.push(offset as u32);
        sizes.push(LIST_SIZE as u32);
    }

    ListViewArray::new(
        elements,
        PrimitiveArray::from_iter(offsets).into_array(),
        PrimitiveArray::from_iter(sizes).into_array(),
        Validity::NonNullable,
    )
}

/// Same as [`make_primitive_listview`] but with string elements.
fn make_varbinview_listview(referenced_fraction: f64) -> ListViewArray {
    let mut rng = StdRng::seed_from_u64(0);

    let referenced = VECTOR_SIZE * LIST_SIZE;
    let element_count = ((referenced as f64) / referenced_fraction).max(1.0) as usize;

    let strings: Vec<String> = (0..element_count)
        .map(|i| {
            if i % 3 == 0 {
                format!("a-longer-string-value-{i:08}")
            } else {
                format!("s{i}")
            }
        })
        .collect();
    let elements = VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();

    let mut offsets: Vec<u32> = Vec::with_capacity(VECTOR_SIZE);
    let mut sizes: Vec<u32> = Vec::with_capacity(VECTOR_SIZE);
    let max_offset = element_count.saturating_sub(LIST_SIZE);
    for _ in 0..VECTOR_SIZE {
        let offset = rng.random_range(0..=max_offset.max(1));
        offsets.push(offset as u32);
        sizes.push(LIST_SIZE as u32);
    }

    ListViewArray::new(
        elements,
        PrimitiveArray::from_iter(offsets).into_array(),
        PrimitiveArray::from_iter(sizes).into_array(),
        Validity::NonNullable,
    )
}

fn export_listview(lv: ListViewArray, element_type: LogicalType) {
    let list_type = LogicalType::list_type(element_type).unwrap();
    let mut chunk = DataChunk::new([list_type]);
    let cache = ConversionCache::default();
    let session = &*SESSION;
    let mut ctx = session.create_execution_ctx();
    let exporter = new_array_exporter(lv.into_array(), &cache, &mut ctx).unwrap();
    exporter
        .export(0, VECTOR_SIZE, chunk.get_vector_mut(0), &mut ctx)
        .unwrap();
    chunk.set_len(VECTOR_SIZE);
}

const REFERENCED_FRACTIONS: [f64; 4] = [0.01, 0.05, 0.25, 1.0];

#[divan::bench(args = REFERENCED_FRACTIONS)]
fn primitive(bencher: Bencher, referenced_fraction: f64) {
    bencher
        .with_inputs(|| make_primitive_listview(referenced_fraction))
        .bench_values(|lv| export_listview(lv, LogicalType::int64()));
}

#[divan::bench(args = REFERENCED_FRACTIONS)]
fn varbinview(bencher: Bencher, referenced_fraction: f64) {
    bencher
        .with_inputs(|| make_varbinview_listview(referenced_fraction))
        .bench_values(|lv| export_listview(lv, LogicalType::varchar()));
}
