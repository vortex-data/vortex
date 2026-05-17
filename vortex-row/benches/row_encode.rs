// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(
    clippy::unwrap_used,
    clippy::clone_on_ref_ptr,
    clippy::cloned_ref_to_slice_refs,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::redundant_clone
)]

//! Row-encode throughput benchmarks comparing `arrow-row` against vortex's `convert_columns`
//! for the canonical scenarios shipped in PR 1: a primitive i64 column, a Utf8 column,
//! and a mixed-field struct. Per-encoding fast paths (Constant, Dict, Patched, BitPacked,
//! FoR, Delta) gain their own triplets in PR 3.

use std::sync::Arc;

use arrow_array::DictionaryArray;
use arrow_array::Int32Array;
use arrow_array::Int64Array;
use arrow_array::PrimitiveArray as ArrowPrimitiveArray;
use arrow_array::StringArray;
use arrow_array::StructArray as ArrowStructArray;
use arrow_array::types::Int32Type;
use arrow_row::RowConverter;
use arrow_row::SortField as ArrowSortField;
use arrow_schema::DataType;
use arrow_schema::Field;
use divan::counter::BytesCount;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Alphanumeric;
use rand::rngs::StdRng;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builders::dict::dict_encode;
use vortex_array::patches::Patches;
use vortex_fastlanes::BitPackedData;
use vortex_row::SortField;
use vortex_row::convert_columns;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const N: usize = 100_000;

fn main() {
    divan::main();
}

fn gen_i64(n: usize, seed: u64) -> Vec<i64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| rng.random_range(i64::MIN..i64::MAX))
        .collect()
}

fn gen_words(n: usize, mean_len: usize, seed: u64) -> Vec<String> {
    let rng = &mut StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let len = rng.random_range(mean_len.saturating_sub(4)..=mean_len + 4);
            rng.sample_iter(&Alphanumeric)
                .take(len)
                .map(char::from)
                .collect::<String>()
        })
        .collect()
}

// ---------- primitive_i64 ----------

#[divan::bench]
fn primitive_i64_arrow_row(bencher: divan::Bencher) {
    let v = gen_i64(N, 0);
    let arr = Arc::new(Int64Array::from(v.clone())) as arrow_array::ArrayRef;
    let conv = RowConverter::new(vec![ArrowSortField::new(DataType::Int64)]).unwrap();
    let bytes = (N * (1 + 8)) as u64;
    bencher
        .counter(BytesCount::new(bytes))
        .bench_local(|| conv.convert_columns(&[arr.clone()]).unwrap())
}

#[divan::bench]
fn primitive_i64_vortex(bencher: divan::Bencher) {
    let v = gen_i64(N, 0);
    let col = PrimitiveArray::from_iter(v.clone()).into_array();
    let bytes = (N * (1 + 8)) as u64;
    bencher.counter(BytesCount::new(bytes)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        convert_columns(&[col.clone()], &[SortField::default()], &mut ctx).unwrap()
    })
}

// ---------- utf8 ----------

#[divan::bench]
fn utf8_arrow_row(bencher: divan::Bencher) {
    let words = gen_words(N, 16, 7);
    let total: u64 = words
        .iter()
        .map(|w| 1 + (w.len().div_ceil(32) * 33) as u64)
        .sum();
    let arr = Arc::new(StringArray::from(words.clone())) as arrow_array::ArrayRef;
    let conv = RowConverter::new(vec![ArrowSortField::new(DataType::Utf8)]).unwrap();
    bencher
        .counter(BytesCount::new(total))
        .bench_local(|| conv.convert_columns(&[arr.clone()]).unwrap())
}

#[divan::bench]
fn utf8_vortex(bencher: divan::Bencher) {
    let words = gen_words(N, 16, 7);
    let total: u64 = words
        .iter()
        .map(|w| 1 + (w.len().div_ceil(32) * 33) as u64)
        .sum();
    let col = VarBinViewArray::from_iter_str(words.iter().map(String::as_str)).into_array();
    bencher.counter(BytesCount::new(total)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        convert_columns(&[col.clone()], &[SortField::default()], &mut ctx).unwrap()
    })
}

// ---------- struct_mixed ----------

fn struct_mixed_inputs() -> (Vec<i64>, Vec<String>, u64) {
    let ids = gen_i64(N, 1);
    let names = gen_words(N, 16, 2);
    // sentinel (1) + i64 (1+8=9) + utf8-name (1 + ceil(len/32)*33)
    let total: u64 = (0..N)
        .map(|i| {
            let name_bytes = 1 + (names[i].len().div_ceil(32) * 33) as u64;
            1u64 + 9u64 + name_bytes
        })
        .sum();
    (ids, names, total)
}

#[divan::bench]
fn struct_mixed_arrow_row(bencher: divan::Bencher) {
    let (ids, names, total) = struct_mixed_inputs();
    let id_arr = Arc::new(Int64Array::from(ids)) as arrow_array::ArrayRef;
    let name_arr = Arc::new(StringArray::from(names)) as arrow_array::ArrayRef;
    let arrow_struct = Arc::new(ArrowStructArray::from(vec![
        (Arc::new(Field::new("id", DataType::Int64, false)), id_arr),
        (
            Arc::new(Field::new("name", DataType::Utf8, false)),
            name_arr,
        ),
    ])) as arrow_array::ArrayRef;
    let struct_fields = vec![
        Arc::new(Field::new("id", DataType::Int64, false)),
        Arc::new(Field::new("name", DataType::Utf8, false)),
    ];
    let conv = RowConverter::new(vec![ArrowSortField::new(DataType::Struct(
        struct_fields.into(),
    ))])
    .unwrap();
    bencher
        .counter(BytesCount::new(total))
        .bench_local(|| conv.convert_columns(&[arrow_struct.clone()]).unwrap())
}

#[divan::bench]
fn struct_mixed_vortex(bencher: divan::Bencher) {
    let (ids, names, total) = struct_mixed_inputs();
    let id_arr = PrimitiveArray::from_iter(ids).into_array();
    let name_arr = VarBinViewArray::from_iter_str(names.iter().map(String::as_str)).into_array();
    let struct_arr = StructArray::from_fields(&[("id", id_arr), ("name", name_arr)])
        .unwrap()
        .into_array();
    bencher.counter(BytesCount::new(total)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        convert_columns(&[struct_arr.clone()], &[SortField::default()], &mut ctx).unwrap()
    })
}

// ---------- constant_i64 ----------

#[divan::bench]
fn constant_i64_arrow_row(bencher: divan::Bencher) {
    let arr = Arc::new(Int64Array::from(vec![42i64; N])) as arrow_array::ArrayRef;
    let conv = RowConverter::new(vec![ArrowSortField::new(DataType::Int64)]).unwrap();
    let total = (N * (1 + 8)) as u64;
    bencher
        .counter(BytesCount::new(total))
        .bench_local(|| conv.convert_columns(&[arr.clone()]).unwrap())
}

#[divan::bench]
fn constant_i64_vortex_with_kernel(bencher: divan::Bencher) {
    let arr = ConstantArray::new(42i64, N).into_array();
    let total = (N * (1 + 8)) as u64;
    bencher.counter(BytesCount::new(total)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        convert_columns(&[arr.clone()], &[SortField::default()], &mut ctx).unwrap()
    })
}

#[divan::bench]
fn constant_i64_vortex_without_kernel(bencher: divan::Bencher) {
    let arr = ConstantArray::new(42i64, N).into_array();
    let total = (N * (1 + 8)) as u64;
    bencher.counter(BytesCount::new(total)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let canonical = arr
            .clone()
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_array();
        convert_columns(&[canonical], &[SortField::default()], &mut ctx).unwrap()
    })
}

// ---------- dict_utf8 ----------

fn dict_utf8_inputs() -> (Vec<String>, Vec<String>, Vec<i32>, u64) {
    let n_unique = 1024usize;
    let unique = gen_words(n_unique, 16, 13);
    let mut rng = StdRng::seed_from_u64(17);
    let codes: Vec<i32> = (0..N)
        .map(|_| rng.random_range(0..n_unique) as i32)
        .collect();
    let strings: Vec<String> = codes.iter().map(|&c| unique[c as usize].clone()).collect();
    let bytes: u64 = strings
        .iter()
        .map(|w| 1 + (w.len().div_ceil(32) * 33) as u64)
        .sum();
    (unique, strings, codes, bytes)
}

#[divan::bench]
fn dict_utf8_arrow_dict(bencher: divan::Bencher) {
    let (unique, _, codes, total) = dict_utf8_inputs();
    let values: Arc<dyn arrow_array::Array> = Arc::new(StringArray::from(unique.clone()));
    let dict_arr: DictionaryArray<Int32Type> =
        DictionaryArray::new(ArrowPrimitiveArray::from(codes), values);
    let arr = Arc::new(dict_arr) as arrow_array::ArrayRef;
    let conv = RowConverter::new(vec![ArrowSortField::new(DataType::Dictionary(
        Box::new(DataType::Int32),
        Box::new(DataType::Utf8),
    ))])
    .unwrap();
    bencher
        .counter(BytesCount::new(total))
        .bench_local(|| conv.convert_columns(&[arr.clone()]).unwrap())
}

#[divan::bench]
fn dict_utf8_arrow_canonical(bencher: divan::Bencher) {
    let (_, strings, _, total) = dict_utf8_inputs();
    let arr = Arc::new(StringArray::from(strings.clone())) as arrow_array::ArrayRef;
    let conv = RowConverter::new(vec![ArrowSortField::new(DataType::Utf8)]).unwrap();
    bencher
        .counter(BytesCount::new(total))
        .bench_local(|| conv.convert_columns(&[arr.clone()]).unwrap())
}

#[divan::bench]
fn dict_utf8_vortex_with_kernel(bencher: divan::Bencher) {
    let (_, strings, _, total) = dict_utf8_inputs();
    let raw = VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();
    let dict = dict_encode(&raw).unwrap().into_array();
    bencher.counter(BytesCount::new(total)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        convert_columns(&[dict.clone()], &[SortField::default()], &mut ctx).unwrap()
    })
}

#[divan::bench]
fn dict_utf8_vortex_without_kernel(bencher: divan::Bencher) {
    let (_, strings, _, total) = dict_utf8_inputs();
    let raw = VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();
    let dict = dict_encode(&raw).unwrap().into_array();
    bencher.counter(BytesCount::new(total)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let canonical = dict
            .clone()
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_array();
        convert_columns(&[canonical], &[SortField::default()], &mut ctx).unwrap()
    })
}

// ---------- patched_i32 ----------

fn gen_patched_i32_inputs() -> (Vec<i32>, Vec<i32>, u64) {
    let mut rng = StdRng::seed_from_u64(400);
    // Inner is mostly zero, with random patches at ~5% of positions.
    let mut inner = vec![0i32; N];
    let mut values = Vec::new();
    for slot in inner.iter_mut().take(N) {
        if rng.random_range(0u32..100) < 5 {
            let v = rng.random_range(1i32..1_000_000);
            *slot = v;
            values.push(v);
        }
    }
    let bytes = (N * (1 + 4)) as u64;
    (inner, values, bytes)
}

#[divan::bench]
fn patched_i32_arrow_row(bencher: divan::Bencher) {
    let (inner, _, bytes) = gen_patched_i32_inputs();
    let arr = Arc::new(Int32Array::from(inner)) as arrow_array::ArrayRef;
    let conv = RowConverter::new(vec![ArrowSortField::new(DataType::Int32)]).unwrap();
    bencher
        .counter(BytesCount::new(bytes))
        .bench_local(|| conv.convert_columns(&[arr.clone()]).unwrap())
}

fn patched_i32_array() -> (vortex_array::ArrayRef, u64) {
    let mut rng = StdRng::seed_from_u64(400);
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<i32> = Vec::new();
    let mut inner = vec![0i32; N];
    for i in 0..N {
        if rng.random_range(0u32..100) < 5 {
            let v = rng.random_range(1i32..1_000_000);
            inner[i] = v;
            indices.push(i as u32);
            values.push(v);
        }
    }
    let inner_arr = PrimitiveArray::from_iter(vec![0i32; N]).into_array();
    let patches = Patches::new(
        N,
        0,
        PrimitiveArray::from_iter(indices).into_array(),
        PrimitiveArray::from_iter(values).into_array(),
        None,
    )
    .unwrap();
    let mut setup_ctx = LEGACY_SESSION.create_execution_ctx();
    let patched = Patched::from_array_and_patches(inner_arr, &patches, &mut setup_ctx)
        .unwrap()
        .into_array();
    drop(inner);
    let bytes = (N * (1 + 4)) as u64;
    (patched, bytes)
}

#[divan::bench]
fn patched_i32_with_kernel(bencher: divan::Bencher) {
    let (arr, bytes) = patched_i32_array();
    bencher.counter(BytesCount::new(bytes)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        convert_columns(&[arr.clone()], &[SortField::default()], &mut ctx).unwrap()
    })
}

#[divan::bench]
fn patched_i32_without_kernel(bencher: divan::Bencher) {
    let (arr, bytes) = patched_i32_array();
    bencher.counter(BytesCount::new(bytes)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let canonical = arr
            .clone()
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_array();
        convert_columns(&[canonical], &[SortField::default()], &mut ctx).unwrap()
    })
}

// ---------- bitpacked_i32 ----------

fn gen_bitpacked_i32_values(n: usize, seed: u64) -> Vec<i32> {
    // Small positive integers in the 0..255 range so they bit-pack to 8 bits without patches.
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(0i32..256)).collect()
}

#[divan::bench]
fn bitpacked_i32_arrow_row(bencher: divan::Bencher) {
    let v = gen_bitpacked_i32_values(N, 100);
    let arr = Arc::new(Int32Array::from(v.clone())) as arrow_array::ArrayRef;
    let conv = RowConverter::new(vec![ArrowSortField::new(DataType::Int32)]).unwrap();
    let bytes = (N * (1 + 4)) as u64;
    bencher
        .counter(BytesCount::new(bytes))
        .bench_local(|| conv.convert_columns(&[arr.clone()]).unwrap())
}

#[divan::bench]
fn bitpacked_i32_with_kernel(bencher: divan::Bencher) {
    let v = gen_bitpacked_i32_values(N, 100);
    let raw = PrimitiveArray::from_iter(v.clone()).into_array();
    let mut setup_ctx = LEGACY_SESSION.create_execution_ctx();
    let bp = BitPackedData::encode(&raw, 8, &mut setup_ctx)
        .unwrap()
        .into_array();
    let bytes = (N * (1 + 4)) as u64;
    bencher.counter(BytesCount::new(bytes)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        convert_columns(&[bp.clone()], &[SortField::default()], &mut ctx).unwrap()
    })
}

#[divan::bench]
fn bitpacked_i32_without_kernel(bencher: divan::Bencher) {
    let v = gen_bitpacked_i32_values(N, 100);
    let raw = PrimitiveArray::from_iter(v.clone()).into_array();
    let mut setup_ctx = LEGACY_SESSION.create_execution_ctx();
    let bp = BitPackedData::encode(&raw, 8, &mut setup_ctx)
        .unwrap()
        .into_array();
    let bytes = (N * (1 + 4)) as u64;
    bencher.counter(BytesCount::new(bytes)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let canonical = bp
            .clone()
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_array();
        convert_columns(&[canonical], &[SortField::default()], &mut ctx).unwrap()
    })
}
