// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end `ListArray` slice → export benchmarks.
//!
//! `ListArray`'s monotonic-offsets invariant means `take` / `filter` already produce compact
//! elements (they materialise a new contiguous elements buffer). The only ops that leave
//! prefix/suffix garbage in `elements` are `slice` (keeps offsets/validity, slices nothing) and
//! file loads that retained surrounding data. The export-time
//! `maybe_trim_unreferenced_elements` helper trims that garbage before downstream conversion.
//!
//! This bench measures the win from encoding-aware trim thresholds: a `slice` that keeps the
//! middle 20% of a dict-encoded varbin column would, under the old 97% savings threshold, fail
//! to trim — the export then decompresses the entire elements buffer. Compressed children
//! justify a much more aggressive trim because per-position export work is decompression, not
//! a `memcpy`.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#![expect(clippy::panic)]

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
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::builders::dict::dict_encode;
use vortex_array::validity::Validity;

fn main() {
    divan::main();
}

/// 16k lists × 8 elements = 128 KiB elements — comfortably above the encoding-aware MIN
/// threshold so the trim fires for compressed children. Smaller buffers fall below the MIN
/// and skip the trim regardless of savings.
const NUM_LISTS: usize = 16_384;
const LIST_SIZE: usize = 8;

fn make_list_primitive(rng: &mut StdRng) -> ListArray {
    let element_count = NUM_LISTS * LIST_SIZE;
    let elements = PrimitiveArray::from_iter(0i64..(element_count as i64)).into_array();
    let _ = rng;
    let offsets: Vec<u32> = (0..=NUM_LISTS).map(|i| (i * LIST_SIZE) as u32).collect();
    ListArray::try_new(
        elements,
        PrimitiveArray::from_iter(offsets).into_array(),
        Validity::NonNullable,
    )
    .unwrap()
}

fn make_list_varbin_long(rng: &mut StdRng) -> ListArray {
    let element_count = NUM_LISTS * LIST_SIZE;
    let strings: Vec<String> = (0..element_count)
        .map(|i| format!("a-longer-string-value-padded-out-{i:08}"))
        .collect();
    let elements = VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();
    let _ = rng;
    let offsets: Vec<u32> = (0..=NUM_LISTS).map(|i| (i * LIST_SIZE) as u32).collect();
    ListArray::try_new(
        elements,
        PrimitiveArray::from_iter(offsets).into_array(),
        Validity::NonNullable,
    )
    .unwrap()
}

/// `ListArray` whose elements are `DictArray<VarBinView>` — i.e. compressed.
///
/// Per-position export is dominated by dict decompression, so leaving prefix/suffix garbage in
/// the elements buffer is expensive even at moderate savings ratios. This is exactly the case
/// where the encoding-aware trim threshold (50% for compressed vs 97% for canonical) is meant
/// to fire.
fn make_list_dict_varbin(rng: &mut StdRng) -> ListArray {
    let element_count = NUM_LISTS * LIST_SIZE;
    let vocab: Vec<String> = (0..256)
        .map(|i| format!("vocab-entry-{i:04}-padded-out"))
        .collect();
    let strings: Vec<&str> = (0..element_count)
        .map(|_| vocab[rng.random_range(0..vocab.len())].as_str())
        .collect();
    let canonical = VarBinViewArray::from_iter_str(strings).into_array();
    let elements = dict_encode(&canonical).unwrap().into_array();
    let offsets: Vec<u32> = (0..=NUM_LISTS).map(|i| (i * LIST_SIZE) as u32).collect();
    ListArray::try_new(
        elements,
        PrimitiveArray::from_iter(offsets).into_array(),
        Validity::NonNullable,
    )
    .unwrap()
}

fn arrow_type_for(elem: &str) -> DataType {
    let item = match elem {
        "prim" => DataType::Int64,
        "varbin_long" | "dict_varbin" => DataType::Utf8View,
        _ => panic!("unknown elem"),
    };
    DataType::List(Field::new("item", item, false).into())
}

fn make_source(elem: &str, rng: &mut StdRng) -> ListArray {
    match elem {
        "prim" => make_list_primitive(rng),
        "varbin_long" => make_list_varbin_long(rng),
        "dict_varbin" => make_list_dict_varbin(rng),
        _ => panic!("unknown elem"),
    }
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

const ELEM_TYPES: [&str; 3] = ["prim", "varbin_long", "dict_varbin"];
/// Fraction of rows kept by `slice` (middle window). 0.20 means keep middle 20% — exactly the
/// case where compressed-element trimming should fire and canonical-element trimming should
/// not.
const KEEP_FRACTIONS: [f64; 4] = [0.01, 0.05, 0.20, 0.50];

fn matrix() -> Vec<(&'static str, f64)> {
    let mut out = Vec::new();
    for &elem in &ELEM_TYPES {
        for &k in &KEEP_FRACTIONS {
            out.push((elem, k));
        }
    }
    out
}

#[divan::bench(args = matrix())]
fn slice_middle_export(bencher: Bencher, (elem, keep): (&str, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let list = make_source(elem, &mut rng);
    let keep_rows = ((NUM_LISTS as f64) * keep).max(1.0) as usize;
    let start = (NUM_LISTS - keep_rows) / 2;
    let end = start + keep_rows;
    let dt = arrow_type_for(elem);
    bencher
        .with_inputs(|| list.clone().into_array())
        .bench_values(|a| {
            let sliced = a.slice(start..end).unwrap();
            export(sliced, &dt);
        });
}
