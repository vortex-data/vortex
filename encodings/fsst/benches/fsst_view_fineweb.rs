// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST vs FSSTView on **real FineWeb columns** (not synthetic data).
//!
//! The HuggingFace FineWeb `10BT` sample is ~2 GB, so this bench does not download it. Instead it
//! reads two length-prefixed binary dumps of real columns, produced once with DuckDB:
//!
//! ```text
//! python3 - <<'PY'
//! import duckdb, struct
//! con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
//! url = "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/v1.4.0/sample/10BT/001_00000.parquet"
//! con.execute(f"COPY (SELECT url, text FROM read_parquet('{url}') LIMIT 200000) TO '/tmp/s.parquet' (FORMAT PARQUET)")
//! def dump(col, path, limit):
//!     rows = con.execute(f"SELECT {col} FROM read_parquet('/tmp/s.parquet') WHERE {col} IS NOT NULL LIMIT {limit}").fetchall()
//!     with open(path, "wb") as f:
//!         f.write(struct.pack("<Q", len(rows)))
//!         for (v,) in rows:
//!             b = v.encode(); f.write(struct.pack("<I", len(b))); f.write(b)
//! dump("url", "/tmp/fineweb_url.bin", 200000)
//! dump("text", "/tmp/fineweb_text.bin", 40000)
//! PY
//! ```
//!
//! Then point the bench at them:
//!
//! ```text
//! FINEWEB_URL=/tmp/fineweb_url.bin FINEWEB_TEXT=/tmp/fineweb_text.bin \
//!   cargo bench -p vortex-fsst --bench fsst_view_fineweb
//! ```
//!
//! If the env vars are unset (or the files are missing), every bench no-ops, so CI stays green.
//!
//! File format: `u64` little-endian row count, then for each row a `u32` little-endian byte length
//! followed by that many UTF-8 bytes.
//!
//! Two real columns, two very different shapes:
//! - `url`  — short strings, ~72 B average (a realistic "short string" column).
//! - `text` — long  strings, ~3 KB average (a realistic "long string" column).
//!
//! Workloads compared, fsst (rewrite heap per op) vs fsstview (metadata-only per op):
//! - `single_filter`: one filter, then canonicalize to a `VarBinViewArray`.
//! - `chain`: convert once, then 5 alternating filter/take ops, then canonicalize once — the case
//!   the view encoding is actually designed for.

#![expect(clippy::unwrap_used)]

use std::path::PathBuf;

use divan::Bencher;
use divan::black_box;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_fsst::FSST;
use vortex_fsst::FSSTArray;
use vortex_fsst::FSSTView;
use vortex_fsst::FsstViewCompaction;
use vortex_fsst::canonicalize_fsstview_with;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_fsst::fsstview_from_fsst;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

/// A real FineWeb column to benchmark, selected by env var.
#[derive(Clone, Copy, Debug)]
enum Column {
    /// `url` — short strings, ~72 B average.
    Url,
    /// `text` — long strings, ~3 KB average.
    Text,
}

impl std::fmt::Display for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Column::Url => "url",
            Column::Text => "text",
        })
    }
}

impl Column {
    fn env_var(self) -> &'static str {
        match self {
            Column::Url => "FINEWEB_URL",
            Column::Text => "FINEWEB_TEXT",
        }
    }

    fn path(self) -> Option<PathBuf> {
        std::env::var_os(self.env_var())
            .map(PathBuf::from)
            .filter(|p| p.exists())
    }
}

const COLUMNS: &[Column] = &[Column::Url, Column::Text];

/// Read a length-prefixed dump into a `VarBinArray`. Returns `None` if the column isn't configured
/// (so the bench no-ops cleanly when the data isn't present).
fn load_column(col: Column) -> Option<VarBinArray> {
    let bytes = std::fs::read(col.path()?).unwrap();
    let mut pos = 0usize;
    #[expect(clippy::cast_possible_truncation)]
    let row_count = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    pos += 8;
    let mut values: Vec<Option<Vec<u8>>> = Vec::with_capacity(row_count);
    for _ in 0..row_count {
        let len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        values.push(Some(bytes[pos..pos + len].to_vec()));
        pos += len;
    }
    Some(VarBinArray::from_iter(
        values.into_iter().map(|v| v.map(Vec::into_boxed_slice)),
        DType::Utf8(Nullability::NonNullable),
    ))
}

fn compress(varbin: &VarBinArray, ctx: &mut ExecutionCtx) -> FSSTArray {
    let compressor = fsst_train_compressor(varbin);
    fsst_compress(varbin, varbin.len(), varbin.dtype(), &compressor, ctx)
}

/// Clustered selection (32 bursts, ~`keep` fraction) — a realistic correlated predicate, the shape
/// where survivors form runs rather than scattering uniformly.
fn clustered_mask(len: usize, keep: f64) -> Mask {
    let mut rng = StdRng::seed_from_u64(9);
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let total = (len as f64 * keep) as usize;
    let bursts = 32usize;
    let burst_len = (total / bursts).max(1);
    let mut keep_set = vec![false; len];
    for _ in 0..bursts {
        let start = rng.random_range(0..len.saturating_sub(burst_len).max(1));
        for j in start..(start + burst_len).min(len) {
            keep_set[j] = true;
        }
    }
    Mask::from_iter(keep_set)
}

/// Sorted-index take (~`keep` fraction) — an index lookup / RID-list join; preserves heap order.
fn sorted_take(len: usize, keep: f64) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(13);
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n = (len as f64 * keep) as usize;
    let mut idx: Vec<u64> = (0..n).map(|_| rng.random_range(0..len as u64)).collect();
    idx.sort_unstable();
    PrimitiveArray::from_iter(idx).into_array()
}

fn fsst_filter(array: &FSSTArray, mask: &Mask, ctx: &mut ExecutionCtx) -> FSSTArray {
    <FSST as FilterKernel>::filter(array.as_view(), mask, ctx)
        .unwrap()
        .unwrap()
        .try_downcast::<FSST>()
        .ok()
        .unwrap()
}

fn fsst_take(array: &FSSTArray, indices: &ArrayRef, ctx: &mut ExecutionCtx) -> FSSTArray {
    <FSST as TakeExecute>::take(array.as_view(), indices, ctx)
        .unwrap()
        .unwrap()
        .try_downcast::<FSST>()
        .ok()
        .unwrap()
}

fn fsst_to_vbv(array: &FSSTArray, ctx: &mut ExecutionCtx) -> ArrayRef {
    array
        .clone()
        .into_array()
        .execute::<VarBinViewArray>(ctx)
        .unwrap()
        .into_array()
}

// =============================== SINGLE FILTER -> VarBinView ===================================

#[divan::bench(args = COLUMNS)]
fn single_filter_fsst(bencher: Bencher, col: Column) {
    let Some(varbin) = load_column(col) else {
        return;
    };
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let mask = clustered_mask(fsst.len(), 0.10);
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| {
            let filtered = fsst_filter(fsst, mask, ctx);
            black_box(fsst_to_vbv(&filtered, ctx))
        });
}

#[divan::bench(args = COLUMNS)]
fn single_filter_view(bencher: Bencher, col: Column) {
    let Some(varbin) = load_column(col) else {
        return;
    };
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    let mask = clustered_mask(fsst.len(), 0.10);
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| {
            let view = fsstview_from_fsst(fsst, ctx).unwrap();
            let filtered = <FSSTView as FilterKernel>::filter(view.as_view(), mask, ctx)
                .unwrap()
                .unwrap()
                .try_downcast::<FSSTView>()
                .ok()
                .unwrap();
            black_box(
                canonicalize_fsstview_with(filtered.as_view(), FsstViewCompaction::Auto, ctx)
                    .unwrap(),
            )
        });
}

// =============================== CHAIN (convert once, N ops, export once) ======================

const CHAIN_LEN: usize = 5;

#[divan::bench(args = COLUMNS)]
fn chain_fsst(bencher: Bencher, col: Column) {
    let Some(varbin) = load_column(col) else {
        return;
    };
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    bencher
        .with_inputs(|| (&fsst, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, ctx)| {
            let mut cur = (*fsst).clone();
            for op in 0..CHAIN_LEN {
                if op % 2 == 0 {
                    let mask = clustered_mask(cur.len(), 0.80);
                    cur = fsst_filter(&cur, &mask, ctx);
                } else {
                    let indices = sorted_take(cur.len(), 0.80);
                    cur = fsst_take(&cur, &indices, ctx);
                }
            }
            black_box(fsst_to_vbv(&cur, ctx))
        });
}

#[divan::bench(args = COLUMNS)]
fn chain_view(bencher: Bencher, col: Column) {
    let Some(varbin) = load_column(col) else {
        return;
    };
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    bencher
        .with_inputs(|| (&fsst, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, ctx)| {
            // Convert once, then chain metadata-only ops, canonicalize once.
            let mut cur = fsstview_from_fsst(fsst, ctx).unwrap();
            for op in 0..CHAIN_LEN {
                let next = if op % 2 == 0 {
                    let mask = clustered_mask(cur.len(), 0.80);
                    <FSSTView as FilterKernel>::filter(cur.as_view(), &mask, ctx)
                        .unwrap()
                        .unwrap()
                } else {
                    let indices = sorted_take(cur.len(), 0.80);
                    <FSSTView as TakeExecute>::take(cur.as_view(), &indices, ctx)
                        .unwrap()
                        .unwrap()
                };
                cur = next.try_downcast::<FSSTView>().ok().unwrap();
            }
            black_box(
                canonicalize_fsstview_with(cur.as_view(), FsstViewCompaction::Auto, ctx).unwrap(),
            )
        });
}
