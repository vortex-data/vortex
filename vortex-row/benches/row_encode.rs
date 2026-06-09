// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(
    clippy::unwrap_used,
    clippy::clone_on_ref_ptr,
    clippy::cloned_ref_to_slice_refs,
    clippy::redundant_clone
)]

//! Row-encode throughput benchmarks comparing `arrow-row` against Vortex's [`RowEncoder`]
//! for the core canonical scenarios: a primitive i64 column, a Utf8 column, and a
//! mixed-field struct.

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::Int64Array;
use arrow_array::StringArray;
use arrow_array::StructArray as ArrowStructArray;
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
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::session::ArraySession;
use vortex_row::RowEncoder;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const N: usize = 100_000;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

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
    let encoder = RowEncoder::default();
    bencher
        .counter(BytesCount::new(bytes))
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| encoder.encode(&[col.clone()], &mut ctx).unwrap())
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
    let encoder = RowEncoder::default();
    bencher
        .counter(BytesCount::new(total))
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| encoder.encode(&[col.clone()], &mut ctx).unwrap())
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
    let encoder = RowEncoder::default();
    bencher
        .counter(BytesCount::new(total))
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| encoder.encode(&[struct_arr.clone()], &mut ctx).unwrap())
}
