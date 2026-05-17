// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for exporting [`DictArray`] to DuckDB vectors.
//!
//! The dict exporter publishes the dictionary values once into a [`ReusableDict`] and then exposes
//! them through a selection vector built from the codes. When the codes only reference a small
//! subset of the values (e.g., after a selective filter pushed down beneath the dict layer), the
//! exporter still materialises the entire `values` array. These benches exercise that path so we
//! can quantify the cost of exporting unreferenced values and the savings from pruning them.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex_duckdb::SESSION;
use vortex_duckdb::duckdb::DataChunk;
use vortex_duckdb::duckdb::LogicalType;
use vortex_duckdb::exporter::ConversionCache;
use vortex_duckdb::exporter::new_array_exporter;

fn main() {
    divan::main();
}

/// Number of codes per export call. Matches DuckDB's `STANDARD_VECTOR_SIZE` (2048).
const VECTOR_SIZE: usize = 2048;

/// Build a primitive `DictArray` whose codes reference only `referenced_fraction` of the values.
fn make_primitive_dict(num_values: usize, referenced_fraction: f64) -> DictArray {
    let mut rng = StdRng::seed_from_u64(0);
    let referenced = ((num_values as f64) * referenced_fraction).max(1.0) as usize;

    let values = PrimitiveArray::from_iter(0i64..num_values as i64).into_array();
    let codes =
        PrimitiveArray::from_iter((0..VECTOR_SIZE).map(|_| rng.random_range(0..referenced) as u32))
            .into_array();
    DictArray::try_new(codes, values).unwrap()
}

/// Build a `VarBinViewArray`-backed `DictArray` whose codes reference `referenced_fraction` of
/// the values.
fn make_string_dict(num_values: usize, referenced_fraction: f64) -> DictArray {
    let mut rng = StdRng::seed_from_u64(0);
    let referenced = ((num_values as f64) * referenced_fraction).max(1.0) as usize;

    let strings: Vec<String> = (0..num_values)
        .map(|i| {
            if i % 3 == 0 {
                format!("a-longer-string-value-{i:08}")
            } else {
                format!("s{i}")
            }
        })
        .collect();
    let values = VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();
    let codes =
        PrimitiveArray::from_iter((0..VECTOR_SIZE).map(|_| rng.random_range(0..referenced) as u32))
            .into_array();
    DictArray::try_new(codes, values).unwrap()
}

fn export_dict(dict: DictArray, logical_type: LogicalType) {
    let mut chunk = DataChunk::new([logical_type]);
    let cache = ConversionCache::default();
    let session = &*SESSION;
    let mut ctx = session.create_execution_ctx();
    let exporter = new_array_exporter(dict.into_array(), &cache, &mut ctx).unwrap();
    exporter
        .export(0, VECTOR_SIZE, chunk.get_vector_mut(0), &mut ctx)
        .unwrap();
    chunk.set_len(VECTOR_SIZE);
}

const VALUES_SIZES: [usize; 3] = [1_024, 16_384, 131_072];
const REFERENCED_FRACTIONS: [f64; 4] = [0.01, 0.05, 0.25, 1.0];

#[divan::bench(
    args = [
        // (num_values, referenced_fraction)
        (VALUES_SIZES[0], REFERENCED_FRACTIONS[0]),
        (VALUES_SIZES[0], REFERENCED_FRACTIONS[3]),
        (VALUES_SIZES[1], REFERENCED_FRACTIONS[0]),
        (VALUES_SIZES[1], REFERENCED_FRACTIONS[1]),
        (VALUES_SIZES[1], REFERENCED_FRACTIONS[2]),
        (VALUES_SIZES[1], REFERENCED_FRACTIONS[3]),
        (VALUES_SIZES[2], REFERENCED_FRACTIONS[0]),
        (VALUES_SIZES[2], REFERENCED_FRACTIONS[1]),
    ]
)]
fn primitive(bencher: Bencher, (num_values, referenced_fraction): (usize, f64)) {
    bencher
        .with_inputs(|| make_primitive_dict(num_values, referenced_fraction))
        .bench_values(|dict| export_dict(dict, LogicalType::int64()));
}

#[divan::bench(
    args = [
        (VALUES_SIZES[0], REFERENCED_FRACTIONS[0]),
        (VALUES_SIZES[0], REFERENCED_FRACTIONS[3]),
        (VALUES_SIZES[1], REFERENCED_FRACTIONS[0]),
        (VALUES_SIZES[1], REFERENCED_FRACTIONS[1]),
        (VALUES_SIZES[1], REFERENCED_FRACTIONS[2]),
        (VALUES_SIZES[1], REFERENCED_FRACTIONS[3]),
        (VALUES_SIZES[2], REFERENCED_FRACTIONS[0]),
        (VALUES_SIZES[2], REFERENCED_FRACTIONS[1]),
    ]
)]
fn varbinview(bencher: Bencher, (num_values, referenced_fraction): (usize, f64)) {
    bencher
        .with_inputs(|| make_string_dict(num_values, referenced_fraction))
        .bench_values(|dict| export_dict(dict, LogicalType::varchar()));
}
