// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end ListView compute → export benchmarks.
//!
//! Each case starts from a [`ListViewArray`] sized like a real Vortex scan chunk (2048 rows,
//! 8-element-wide lists, varbinview or primitive children). We apply one or two compute ops
//! (`take`, `slice`, or a chain) and then convert the result to Arrow ListView.
//!
//! The point is to make the `reachable_elements_bound` propagation visible end-to-end:
//!
//! - `take` and `slice` on `ListView` now sum the kept `sizes` and stamp the result onto
//!   the new array's `reachable_elements_bound` (cheap — they already iterate `sizes`).
//! - The export-time prune helper (`maybe_prune_unreferenced_elements`) reads the bound as
//!   an O(1) signal instead of recomputing `sum(sizes)`.
//! - For chained ops (`take` → `slice` → export) the bound tightens at each step without
//!   the consumer having to walk `sizes` again.

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
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::validity::Validity;

fn main() {
    divan::main();
}

const NUM_LISTS: usize = 2048;
const LIST_SIZE: usize = 8;

/// Build a ListView with random-offset views over a `density`-fraction-referenced elements
/// buffer (i.e. `density = 0.01` => elements buffer is ~100× larger than `sum(sizes)`).
fn make_lv_primitive(density: f64, rng: &mut StdRng) -> ListViewArray {
    let referenced = NUM_LISTS * LIST_SIZE;
    let element_count = ((referenced as f64) / density).max(1.0) as usize;
    let elements = PrimitiveArray::from_iter(0i64..(element_count as i64)).into_array();
    let max_offset = element_count.saturating_sub(LIST_SIZE);
    let offsets: Vec<u32> = (0..NUM_LISTS)
        .map(|_| rng.random_range(0..=max_offset.max(1)) as u32)
        .collect();
    let sizes = vec![LIST_SIZE as u32; NUM_LISTS];
    ListViewArray::new(
        elements,
        PrimitiveArray::from_iter(offsets).into_array(),
        PrimitiveArray::from_iter(sizes).into_array(),
        Validity::NonNullable,
    )
}

fn make_lv_varbin_short(density: f64, rng: &mut StdRng) -> ListViewArray {
    let referenced = NUM_LISTS * LIST_SIZE;
    let element_count = ((referenced as f64) / density).max(1.0) as usize;
    let strings: Vec<String> = (0..element_count).map(|i| format!("s{i}")).collect();
    let elements = VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();
    let max_offset = element_count.saturating_sub(LIST_SIZE);
    let offsets: Vec<u32> = (0..NUM_LISTS)
        .map(|_| rng.random_range(0..=max_offset.max(1)) as u32)
        .collect();
    let sizes = vec![LIST_SIZE as u32; NUM_LISTS];
    ListViewArray::new(
        elements,
        PrimitiveArray::from_iter(offsets).into_array(),
        PrimitiveArray::from_iter(sizes).into_array(),
        Validity::NonNullable,
    )
}

fn make_lv_varbin_long(density: f64, rng: &mut StdRng) -> ListViewArray {
    let referenced = NUM_LISTS * LIST_SIZE;
    let element_count = ((referenced as f64) / density).max(1.0) as usize;
    let strings: Vec<String> = (0..element_count)
        .map(|i| format!("a-longer-string-value-padded-out-{i:08}"))
        .collect();
    let elements = VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();
    let max_offset = element_count.saturating_sub(LIST_SIZE);
    let offsets: Vec<u32> = (0..NUM_LISTS)
        .map(|_| rng.random_range(0..=max_offset.max(1)) as u32)
        .collect();
    let sizes = vec![LIST_SIZE as u32; NUM_LISTS];
    ListViewArray::new(
        elements,
        PrimitiveArray::from_iter(offsets).into_array(),
        PrimitiveArray::from_iter(sizes).into_array(),
        Validity::NonNullable,
    )
}

fn arrow_type_for(elem: &str) -> DataType {
    let item = match elem {
        "prim" => DataType::Int64,
        "varbin_short" | "varbin_long" => DataType::Utf8View,
        _ => panic!("unknown elem"),
    };
    DataType::ListView(Field::new("item", item, false).into())
}

fn make_source(elem: &str, density: f64, rng: &mut StdRng) -> ListViewArray {
    match elem {
        "prim" => make_lv_primitive(density, rng),
        "varbin_short" => make_lv_varbin_short(density, rng),
        "varbin_long" => make_lv_varbin_long(density, rng),
        _ => panic!("unknown elem"),
    }
}

fn random_indices(rng: &mut StdRng, total: usize, pick: usize) -> ArrayRef {
    let mut idx: Vec<u32> = (0..pick)
        .map(|_| rng.random_range(0..total) as u32)
        .collect();
    idx.sort_unstable();
    PrimitiveArray::from_iter(idx).into_array()
}

fn export(array: ArrayRef, dt: &DataType) {
    let field = Field::new("v", dt.clone(), false);
    LEGACY_SESSION
        .arrow()
        .execute_arrow(
            array,
            Some(&field),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
}

// ── Element-type × source-density × op-selectivity matrix ─────────────────────────────────────
//
// Source density controls how sparse the *starting* elements buffer is. Op selectivity
// controls how many rows survive the compute op. The product gives us:
// - dense source + selective op  = "filter pushed below dict" shape
// - sparse source + selective op = "doubly-sparse" — should compound

const ELEM_TYPES: [&str; 3] = ["prim", "varbin_short", "varbin_long"];
/// Initial fraction of `elements` reachable through the views in the source array.
const SOURCE_DENSITIES: [f64; 2] = [0.05, 1.0];
/// Fraction of rows kept by the compute op.
const SELECTIVITIES: [f64; 3] = [0.01, 0.1, 0.5];

fn matrix() -> Vec<(&'static str, f64, f64)> {
    let mut out = Vec::new();
    for &elem in &ELEM_TYPES {
        for &d in &SOURCE_DENSITIES {
            for &s in &SELECTIVITIES {
                out.push((elem, d, s));
            }
        }
    }
    out
}

// ── Each op + export, with optional second op ─────────────────────────────────────────────────

#[divan::bench(args = matrix())]
fn export_only(bencher: Bencher, (elem, density, _sel): (&str, f64, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let lv = make_source(elem, density, &mut rng);
    let dt = arrow_type_for(elem);
    bencher
        .with_inputs(|| lv.clone().into_array())
        .bench_values(|a| export(a, &dt));
}

#[divan::bench(args = matrix())]
fn take_then_export(bencher: Bencher, (elem, density, sel): (&str, f64, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let lv = make_source(elem, density, &mut rng).into_array();
    let pick = ((NUM_LISTS as f64) * sel).max(1.0) as usize;
    let indices = random_indices(&mut rng, NUM_LISTS, pick);
    let dt = arrow_type_for(elem);
    bencher
        .with_inputs(|| (lv.clone(), indices.clone()))
        .bench_values(|(a, idx)| {
            let taken = a.take(idx).unwrap();
            export(taken, &dt);
        });
}

#[divan::bench(args = matrix())]
fn slice_then_export(bencher: Bencher, (elem, density, sel): (&str, f64, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let lv = make_source(elem, density, &mut rng).into_array();
    let end = ((NUM_LISTS as f64) * sel).max(1.0) as usize;
    let dt = arrow_type_for(elem);
    bencher
        .with_inputs(|| lv.clone())
        .bench_values(|a| {
            let sliced = a.slice(0..end).unwrap();
            export(sliced, &dt);
        });
}

#[divan::bench(args = matrix())]
fn filter_then_export(bencher: Bencher, (elem, density, sel): (&str, f64, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let lv = make_source(elem, density, &mut rng).into_array();
    let bits: Vec<bool> = (0..NUM_LISTS).map(|_| rng.random_bool(sel)).collect();
    let mask = vortex_mask::Mask::from(vortex_buffer::BitBuffer::from(bits.as_slice()));
    let dt = arrow_type_for(elem);
    bencher
        .with_inputs(|| (lv.clone(), mask.clone()))
        .bench_values(|(a, m)| {
            let filtered = a.filter(m).unwrap();
            export(filtered, &dt);
        });
}

// ── Chains: two compute ops then export ───────────────────────────────────────────────────────

#[divan::bench(args = matrix())]
fn take_slice_export(bencher: Bencher, (elem, density, sel): (&str, f64, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let lv = make_source(elem, density, &mut rng).into_array();
    let pick = ((NUM_LISTS as f64) * sel).max(1.0) as usize;
    let indices = random_indices(&mut rng, NUM_LISTS, pick);
    // After take we have `pick` rows. Slice the first half of those.
    let slice_end = (pick / 2).max(1);
    let dt = arrow_type_for(elem);
    bencher
        .with_inputs(|| (lv.clone(), indices.clone()))
        .bench_values(|(a, idx)| {
            let taken = a.take(idx).unwrap();
            let sliced = taken.slice(0..slice_end).unwrap();
            export(sliced, &dt);
        });
}

#[divan::bench(args = matrix())]
fn slice_take_export(bencher: Bencher, (elem, density, sel): (&str, f64, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let lv = make_source(elem, density, &mut rng).into_array();
    let half = NUM_LISTS / 2;
    let pick = ((half as f64) * sel).max(1.0) as usize;
    let indices = random_indices(&mut rng, half, pick);
    let dt = arrow_type_for(elem);
    bencher
        .with_inputs(|| (lv.clone(), indices.clone()))
        .bench_values(|(a, idx)| {
            let sliced = a.slice(0..half).unwrap();
            let taken = sliced.take(idx).unwrap();
            export(taken, &dt);
        });
}

#[divan::bench(args = matrix())]
fn filter_take_export(bencher: Bencher, (elem, density, sel): (&str, f64, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let lv = make_source(elem, density, &mut rng).into_array();
    let bits: Vec<bool> = (0..NUM_LISTS).map(|_| rng.random_bool(0.5)).collect();
    let mask = vortex_mask::Mask::from(vortex_buffer::BitBuffer::from(bits.as_slice()));
    // After 0.5-selectivity filter we have ~half the rows. Take a `sel` fraction of those.
    let post_filter_rows = bits.iter().filter(|&&b| b).count().max(1);
    let pick = ((post_filter_rows as f64) * sel).max(1.0) as usize;
    let indices = random_indices(&mut rng, post_filter_rows, pick);
    let dt = arrow_type_for(elem);
    bencher
        .with_inputs(|| (lv.clone(), mask.clone(), indices.clone()))
        .bench_values(|(a, m, idx)| {
            let filtered = a.filter(m).unwrap();
            let taken = filtered.take(idx).unwrap();
            export(taken, &dt);
        });
}
