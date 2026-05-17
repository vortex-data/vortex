// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for exporting [`ListArray`] to DuckDB vectors.
//!
//! Unlike [`ListViewArray`], a `ListArray`'s offsets are monotonic, so the only unreferenced
//! elements are the contiguous prefix before `offsets[0]` and suffix after `offsets[len]`. This
//! bench measures the impact of those leading/trailing unreferenced ranges (a common pattern
//! after slicing a chunked list array).

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use divan::Bencher;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex_duckdb::SESSION;
use vortex_duckdb::duckdb::DataChunk;
use vortex_duckdb::duckdb::LogicalType;
use vortex_duckdb::exporter::ConversionCache;
use vortex_duckdb::exporter::new_array_exporter;

fn main() {
    divan::main();
}

const VECTOR_SIZE: usize = 2048;
const LIST_SIZE: usize = 8;

/// Build a `ListArray` whose `offsets` reference only a window of the elements buffer.
///
/// `referenced_fraction` controls the ratio of `referenced_len / elements.len()`. The window is
/// placed in the middle of the elements buffer so there are unreferenced elements both before and
/// after the referenced range.
fn make_primitive_list(referenced_fraction: f64) -> ListArray {
    let referenced = VECTOR_SIZE * LIST_SIZE;
    let element_count = ((referenced as f64) / referenced_fraction).max(1.0) as usize;
    let leading = (element_count - referenced) / 2;

    let elements = PrimitiveArray::from_iter(0i64..element_count as i64).into_array();
    let offsets: Buffer<u32> = (0..=VECTOR_SIZE)
        .map(|i| (leading + i * LIST_SIZE) as u32)
        .collect();

    ListArray::try_new(elements, offsets.into_array(), Validity::NonNullable).unwrap()
}

fn make_varbinview_list(referenced_fraction: f64) -> ListArray {
    let referenced = VECTOR_SIZE * LIST_SIZE;
    let element_count = ((referenced as f64) / referenced_fraction).max(1.0) as usize;
    let leading = (element_count - referenced) / 2;

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
    let offsets: Buffer<u32> = (0..=VECTOR_SIZE)
        .map(|i| (leading + i * LIST_SIZE) as u32)
        .collect();

    ListArray::try_new(elements, offsets.into_array(), Validity::NonNullable).unwrap()
}

fn export_list(list: ListArray, element_type: LogicalType) {
    let list_type = LogicalType::list_type(element_type).unwrap();
    let mut chunk = DataChunk::new([list_type]);
    let cache = ConversionCache::default();
    let session = &*SESSION;
    let mut ctx = session.create_execution_ctx();
    let exporter = new_array_exporter(list.into_array(), &cache, &mut ctx).unwrap();
    exporter
        .export(0, VECTOR_SIZE, chunk.get_vector_mut(0), &mut ctx)
        .unwrap();
    chunk.set_len(VECTOR_SIZE);
}

const REFERENCED_FRACTIONS: [f64; 4] = [0.01, 0.05, 0.25, 1.0];

#[divan::bench(args = REFERENCED_FRACTIONS)]
fn primitive(bencher: Bencher, referenced_fraction: f64) {
    bencher
        .with_inputs(|| make_primitive_list(referenced_fraction))
        .bench_values(|list| export_list(list, LogicalType::int64()));
}

#[divan::bench(args = REFERENCED_FRACTIONS)]
fn varbinview(bencher: Bencher, referenced_fraction: f64) {
    bencher
        .with_inputs(|| make_varbinview_list(referenced_fraction))
        .bench_values(|list| export_list(list, LogicalType::varchar()));
}
