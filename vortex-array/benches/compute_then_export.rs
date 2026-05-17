// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end compute-then-export benchmarks across element types.
//!
//! These benches model the **realistic** Vortex workflow: a column is dictionary-encoded once
//! at ingest, then a query applies one of `filter` / `take` / `slice`, and the surviving rows
//! are exported (to Arrow here; the DuckDB exporter benches its own equivalent paths).
//!
//! The wins live in the `compute → export` boundary: a sparse code distribution after filter
//! leaves most of the dictionary unreferenced, so the export pays per-value work for entries
//! no surviving code touches. The `DensityHint` propagation through filter/take/slice gives
//! the export an O(1) signal that says "this dict is sparse now — prune before materialising"
//! without any extra scan.
//!
//! Element types covered:
//!  - `primitive (i64)` — zero-copy export, so the prune should *not* fire and we should see
//!    no overhead vs the dense baseline at the same row count.
//!  - `primitive (i32)` — same shape, half-width buffer.
//!  - `varbinview (utf8)` — heavy per-element conversion to Arrow, so the prune should
//!    dominate at low selectivities.
//!  - `varbinview (utf8) — long strings` — same path, but with backing-buffer dominated
//!    work so the savings are larger in absolute terms.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use arrow_schema::DataType;
use arrow_schema::Field;
use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::builders::dict::dict_encode;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

/// Realistic row count: matches DuckDB's `STANDARD_VECTOR_SIZE` × a few iterations of a scan.
const ROWS: usize = 16_384;

/// Cardinality of the source column (number of distinct values to dict-encode).
const CARDINALITY: usize = 8_192;

// ---- Source builders -------------------------------------------------------------------------

fn make_prim_i64(rng: &mut StdRng) -> ArrayRef {
    PrimitiveArray::from_iter((0..ROWS).map(|_| rng.random_range(0..CARDINALITY) as i64))
        .into_array()
}

fn make_prim_i32(rng: &mut StdRng) -> ArrayRef {
    PrimitiveArray::from_iter((0..ROWS).map(|_| rng.random_range(0..CARDINALITY) as i32))
        .into_array()
}

/// Short-string mix (avg ~12 bytes/value). Most strings fit inline in a `BinaryView`, so the
/// backing buffer stays small.
fn make_varbinview_short(rng: &mut StdRng) -> ArrayRef {
    let dict: Vec<String> = (0..CARDINALITY).map(|i| format!("s{i}")).collect();
    let strings: Vec<&str> = (0..ROWS)
        .map(|_| dict[rng.random_range(0..CARDINALITY)].as_str())
        .collect();
    VarBinViewArray::from_iter_str(strings).into_array()
}

/// Long-string mix (avg ~40 bytes/value). Every string exceeds the 12-byte inline limit, so the
/// backing buffer dominates and per-value Arrow conversion cost is high.
fn make_varbinview_long(rng: &mut StdRng) -> ArrayRef {
    let dict: Vec<String> = (0..CARDINALITY)
        .map(|i| format!("a-longer-string-value-padded-out-{i:08}"))
        .collect();
    let strings: Vec<&str> = (0..ROWS)
        .map(|_| dict[rng.random_range(0..CARDINALITY)].as_str())
        .collect();
    VarBinViewArray::from_iter_str(strings).into_array()
}

// ---- Compute primitives ----------------------------------------------------------------------

/// Build a Bernoulli mask of the requested selectivity, as a real `Mask`.
fn random_mask(rng: &mut StdRng, len: usize, selectivity: f64) -> Mask {
    let bits = (0..len)
        .map(|_| rng.random_bool(selectivity))
        .collect::<Vec<bool>>();
    Mask::from(BitBuffer::from(bits.as_slice()))
}

/// Build a random `take` indices array selecting `pick` rows out of `total`.
fn random_take_indices(rng: &mut StdRng, total: usize, pick: usize) -> ArrayRef {
    let mut idx: Vec<u32> = (0..pick)
        .map(|_| rng.random_range(0..total) as u32)
        .collect();
    idx.sort_unstable();
    PrimitiveArray::from_iter(idx).into_array()
}

// ---- Element-type matrix --------------------------------------------------------------------

fn arrow_type_for(name: &str) -> DataType {
    match name {
        "prim_i64" => DataType::Int64,
        "prim_i32" => DataType::Int32,
        "varbin_short" | "varbin_long" => DataType::Utf8View,
        _ => panic!("unknown element type"),
    }
}

fn make_source(name: &str, rng: &mut StdRng) -> ArrayRef {
    match name {
        "prim_i64" => make_prim_i64(rng),
        "prim_i32" => make_prim_i32(rng),
        "varbin_short" => make_varbinview_short(rng),
        "varbin_long" => make_varbinview_long(rng),
        _ => panic!("unknown element type"),
    }
}

fn export_arrow(array: ArrayRef, target_data_type: DataType) {
    let field = Field::new("v", target_data_type, false);
    LEGACY_SESSION
        .arrow()
        .execute_arrow(
            array,
            Some(&field),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
}

// ---- Bench cases -----------------------------------------------------------------------------
//
// We bench three workflow shapes — `filter`, `take`, `slice` — across three selectivities
// (`0.01`, `0.05`, `0.5`) and three element types. The matrix is large but each case is short,
// and the parameterisation is exactly what makes the impact of the density hint visible at the
// boundary.

const ELEM_TYPES: [&str; 4] = ["prim_i32", "prim_i64", "varbin_short", "varbin_long"];
const SELECTIVITIES: [f64; 3] = [0.01, 0.05, 0.5];

#[divan::bench(args = product(ELEM_TYPES, SELECTIVITIES))]
fn filter_then_export(bencher: Bencher, (elem, selectivity): (&str, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let source = make_source(elem, &mut rng);
    let dict = dict_encode(&source).unwrap();
    let mask = random_mask(&mut rng, dict.len(), selectivity);
    let dt = arrow_type_for(elem);

    bencher.bench(|| {
        let filtered = dict.clone().into_array().filter(mask.clone()).unwrap();
        export_arrow(filtered, dt.clone());
    });
}

#[divan::bench(args = product(ELEM_TYPES, SELECTIVITIES))]
fn take_then_export(bencher: Bencher, (elem, selectivity): (&str, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let source = make_source(elem, &mut rng);
    let dict = dict_encode(&source).unwrap();
    let pick = ((dict.len() as f64) * selectivity) as usize;
    let indices = random_take_indices(&mut rng, dict.len(), pick.max(1));
    let dt = arrow_type_for(elem);

    bencher.bench(|| {
        let taken = dict.clone().into_array().take(indices.clone()).unwrap();
        export_arrow(taken, dt.clone());
    });
}

#[divan::bench(args = product(ELEM_TYPES, SELECTIVITIES))]
fn slice_then_export(bencher: Bencher, (elem, selectivity): (&str, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let source = make_source(elem, &mut rng);
    let dict = dict_encode(&source).unwrap();
    let end = ((dict.len() as f64) * selectivity).max(1.0) as usize;
    let dt = arrow_type_for(elem);

    bencher.bench(|| {
        let sliced = dict.clone().into_array().slice(0..end).unwrap();
        export_arrow(sliced, dt.clone());
    });
}

#[divan::bench(args = ELEM_TYPES)]
fn export_dense(bencher: Bencher, elem: &str) {
    let mut rng = StdRng::seed_from_u64(0);
    let source = make_source(elem, &mut rng);
    let dict = dict_encode(&source).unwrap().into_array();
    let dt = arrow_type_for(elem);

    bencher.bench(|| {
        export_arrow(dict.clone(), dt.clone());
    });
}

/// Form the cartesian product of two const arrays so a single `args = …` covers the matrix.
fn product<A: Copy + 'static, B: Copy + 'static>(
    a: impl IntoIterator<Item = A>,
    b: impl IntoIterator<Item = B>,
) -> Vec<(A, B)> {
    let bs: Vec<B> = b.into_iter().collect();
    a.into_iter()
        .flat_map(|x| bs.iter().map(move |y| (x, *y)))
        .collect()
}
