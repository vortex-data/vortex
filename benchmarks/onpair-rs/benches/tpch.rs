// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! End-to-end OnPair benchmark over a slice of TPC-H string columns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::unwrap_used
)]
//
// End-to-end OnPair benchmark across a representative slice of TPC-H string
// columns (free text, small enums, patterned IDs, multi-word names). Each
// column is generated in-memory at startup via `tpchgen`/`tpchgen-arrow`,
// cached behind a `Mutex<HashMap>`, and re-used across bench iterations.
//
// Env:
//   * `ONPAIR_BENCH_MAX_BYTES`    — per-column corpus cap (default 64 MiB)
//   * `ONPAIR_BENCH_SCALE_FACTOR` — TPC-H scale factor (default 1.0)
//
// Run with: cargo bench --bench tpch
//
// Targets the slim public API (`compress` + `ColumnView::decompress_into`).

use std::collections::HashMap;
use std::env;
use std::sync::Mutex;
use std::sync::OnceLock;

use std::mem::MaybeUninit;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::Schema;
use divan::Bencher;
use onpair::Column;
use onpair::Config;
use onpair::DECODE_PADDING;
use onpair::MaxDictBits;
use onpair::Threshold;
use onpair::compress;
use tpchgen::generators::CustomerGenerator;
use tpchgen::generators::LineItemGenerator;
use tpchgen::generators::OrderGenerator;
use tpchgen::generators::PartGenerator;
use tpchgen::generators::SupplierGenerator;
use tpchgen_arrow::CustomerArrow;
use tpchgen_arrow::LineItemArrow;
use tpchgen_arrow::OrderArrow;
use tpchgen_arrow::PartArrow;
use tpchgen_arrow::RecordBatchIterator;
use tpchgen_arrow::SupplierArrow;

// ─────────────────────────────────────────────────────────────────────────────
// Bench parameter matrix: (column_name, bits).
// Columns chosen to span character types: free text, small enums, patterned
// IDs, multi-word names. Add to / trim this list to change coverage.
// ─────────────────────────────────────────────────────────────────────────────

// Columns OnPair handles well: long free text with heavy word reuse, plus
// multi-word names with shared vocabulary. Excluded: low-cardinality enums
// (`l_shipmode`, `l_shipinstruct`) and all-unique address-shaped columns
// (`c_address`) which expand under OnPair; patterned IDs (`o_clerk`,
// `c_name`) and small corpora (`p_comment`) which compress mediocrely.
const PARAMS: &[(&str, u8)] = &[
    ("o_comment", 12),
    ("o_comment", 16),
    ("p_name", 12),
    ("p_name", 16),
    ("l_comment", 12),
    ("l_comment", 16),
];

const SCALE_FACTOR_DEFAULT: f64 = 1.0;
const MAX_BYTES_DEFAULT: usize = 64 << 20;
const BATCH_SIZE: usize = 8192 * 8;

// ─────────────────────────────────────────────────────────────────────────────
// Corpus generation + cache.
// ─────────────────────────────────────────────────────────────────────────────

struct Corpus {
    bytes: Vec<u8>,
    offsets: Vec<u64>,
    total_bytes: usize,
}

fn scale_factor() -> f64 {
    env::var("ONPAIR_BENCH_SCALE_FACTOR")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(SCALE_FACTOR_DEFAULT)
}

fn max_bytes() -> usize {
    env::var("ONPAIR_BENCH_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(MAX_BYTES_DEFAULT)
}

/// Cached per-column corpus. `Box::leak`'d so we can hand out `&'static`
/// references usable across bench closures; this is bench-only code.
fn corpus_for(col: &'static str) -> &'static Corpus {
    static CACHE: OnceLock<Mutex<HashMap<&'static str, &'static Corpus>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().expect("cache poisoned");
    if let Some(&c) = map.get(col) {
        return c;
    }
    let sf = scale_factor();
    let cap = max_bytes();
    let (bytes, offsets) = generate_column(col, sf, cap);
    let total_bytes = bytes.len();
    eprintln!(
        "[onpair tpch bench] {col}: sf={sf}, {} rows, {:.2} MiB",
        offsets.len() - 1,
        total_bytes as f64 / (1024.0 * 1024.0)
    );
    let c: &'static Corpus = Box::leak(Box::new(Corpus {
        bytes,
        offsets,
        total_bytes,
    }));
    map.insert(col, c);
    c
}

/// Dispatch column → table generator. Stops once concatenated bytes reach
/// `max_bytes`. Each `<Table>Arrow` iterator yields `RecordBatch`es and
/// exposes a `schema()` accessor via `RecordBatchIterator`.
fn generate_column(col: &str, sf: f64, max: usize) -> (Vec<u8>, Vec<u64>) {
    match col {
        // LineItem
        "l_returnflag" | "l_linestatus" | "l_shipinstruct" | "l_shipmode" | "l_comment" => {
            let it =
                LineItemArrow::new(LineItemGenerator::new(sf, 1, 1)).with_batch_size(BATCH_SIZE);
            let schema = it.schema().clone();
            collect_string_view(it, &schema, col, max)
        }
        // Order
        "o_orderstatus" | "o_orderpriority" | "o_clerk" | "o_comment" => {
            let it = OrderArrow::new(OrderGenerator::new(sf, 1, 1)).with_batch_size(BATCH_SIZE);
            let schema = it.schema().clone();
            collect_string_view(it, &schema, col, max)
        }
        // Customer
        "c_name" | "c_address" | "c_phone" | "c_mktsegment" | "c_comment" => {
            let it =
                CustomerArrow::new(CustomerGenerator::new(sf, 1, 1)).with_batch_size(BATCH_SIZE);
            let schema = it.schema().clone();
            collect_string_view(it, &schema, col, max)
        }
        // Part
        "p_name" | "p_mfgr" | "p_brand" | "p_type" | "p_container" | "p_comment" => {
            let it = PartArrow::new(PartGenerator::new(sf, 1, 1)).with_batch_size(BATCH_SIZE);
            let schema = it.schema().clone();
            collect_string_view(it, &schema, col, max)
        }
        // Supplier
        "s_name" | "s_address" | "s_phone" | "s_comment" => {
            let it =
                SupplierArrow::new(SupplierGenerator::new(sf, 1, 1)).with_batch_size(BATCH_SIZE);
            let schema = it.schema().clone();
            collect_string_view(it, &schema, col, max)
        }
        other => panic!("unknown TPC-H string column: {other}"),
    }
}

fn collect_string_view<I>(
    batches: I,
    schema: &Schema,
    col: &str,
    max_bytes: usize,
) -> (Vec<u8>, Vec<u64>)
where
    I: Iterator<Item = RecordBatch>,
{
    let idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == col)
        .unwrap_or_else(|| panic!("column `{col}` not found"));

    let mut bytes = Vec::with_capacity(max_bytes.min(1 << 28));
    let mut offsets: Vec<u64> = vec![0];
    'outer: for batch in batches {
        let arr = batch.column(idx).as_string_view();
        for v in arr.iter() {
            let s = v.unwrap_or("").as_bytes();
            bytes.extend_from_slice(s);
            offsets.push(bytes.len() as u64);
            if bytes.len() >= max_bytes {
                break 'outer;
            }
        }
    }
    (bytes, offsets)
}

fn build_column(col: &'static str, bits: u8) -> Column<u64> {
    let c = corpus_for(col);
    let cfg = Config {
        max_dict_bits: MaxDictBits::new(bits).unwrap(),
        threshold: Threshold::new(0.2).unwrap(),
        seed: Some(42),
    };
    compress(&c.bytes, &c.offsets, cfg).unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────
// Benches.
// ─────────────────────────────────────────────────────────────────────────────

#[divan::bench(args = PARAMS)]
fn train_and_compress(bencher: Bencher, param: (&'static str, u8)) {
    let (col, bits) = param;
    let c = corpus_for(col);
    let cfg = Config {
        max_dict_bits: MaxDictBits::new(bits).unwrap(),
        threshold: Threshold::new(0.2).unwrap(),
        seed: Some(42),
    };
    bencher
        .counter(divan::counter::BytesCount::new(c.total_bytes))
        .bench(|| {
            compress(
                divan::black_box(&c.bytes),
                divan::black_box(&c.offsets),
                cfg,
            )
            .unwrap()
        });
}

#[divan::bench(args = PARAMS)]
fn decompress_all(bencher: Bencher, param: (&'static str, u8)) {
    let (col, bits) = param;
    let c = corpus_for(col);
    let column = build_column(col, bits);
    let cap = column.view().decoded_len() + DECODE_PADDING;
    bencher
        .counter(divan::counter::BytesCount::new(c.total_bytes))
        .bench(|| {
            let mut buf = vec![MaybeUninit::uninit(); cap];
            // SAFETY: trusted column; `buf` is sized to decoded_len() + DECODE_PADDING.
            let n = unsafe { column.view().decompress_into(&mut buf) };
            divan::black_box(&buf[..n]);
        });
}

fn main() {
    // Pre-warm every (col, bits) combo so the source + compression-ratio lines
    // print before divan starts emitting per-bench output.
    eprintln!("\n[onpair tpch bench] === corpora + compression ratios ===");
    for &(col, bits) in PARAMS {
        let c = corpus_for(col);
        let column = build_column(col, bits);
        let dict_bytes = column.dict.bytes().len();
        let dict_offsets = column.dict.offsets().len() * 4;
        let codes = column.codes.len() * 2;
        let row_offsets = std::mem::size_of_val(column.row_offsets.as_slice());
        let compressed = dict_bytes + dict_offsets + codes + row_offsets;
        eprintln!(
            "  {col:<16} bits={bits}: ratio = {:.3}x  (raw {:.2} MiB → {:.2} MiB)",
            c.total_bytes as f64 / compressed as f64,
            c.total_bytes as f64 / (1024.0 * 1024.0),
            compressed as f64 / (1024.0 * 1024.0),
        );
    }
    eprintln!();
    divan::main();
}
