// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST vs FSSTView materializing a string column under the **real FineWeb benchmark predicates**.
//!
//! The FineWeb queries in `vortex-bench` are `SELECT * FROM fineweb WHERE <predicate>`: each one
//! evaluates a predicate to a row selection, then materializes the surviving rows. This bench does
//! exactly the materialization half — apply a real predicate's selection mask to an FSST-compressed
//! string column and decode it to a `VarBinViewArray` — comparing fsst (rewrites the code heap)
//! vs fsstview (metadata-only filter, decode once).
//!
//! The predicate masks and the string columns are produced once with DuckDB against the real
//! HuggingFace FineWeb 10BT sample (the same file `vortex-bench` uses). The ~2 GB sample is not
//! downloaded by the bench; the recipe is in `fineweb_queries_extract.py` next to this file, and
//! the resulting files are pointed at via env vars:
//!
//! ```text
//! FINEWEB_DIR=/tmp cargo bench -p vortex-fsst --bench fsst_view_fineweb_queries
//! ```
//!
//! `FINEWEB_DIR` must contain `fw_url.bin`, `fw_text.bin` (length-prefixed: `u64` count, then per
//! row `u32` len + bytes) and `fw_mask_<query>.bin` (`u64` count, then one byte per row, 1 = kept).
//! If `FINEWEB_DIR` is unset or files are missing, every bench no-ops so CI stays green.
//!
//! The real predicates span the spectrum the view's `Auto` export was built for: clustered
//! selections (`dump = ...`, `date LIKE '2020-10-%'`) where survivors form long runs, and scattered
//! `LIKE '%...%'` containment filters where they don't.

#![expect(clippy::unwrap_used)]

use std::path::PathBuf;

use divan::Bencher;
use divan::black_box;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
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

/// Real FineWeb benchmark predicates that select rows (the `WHERE` clauses of the `SELECT *`
/// queries in `vortex-bench/src/fineweb`). `filepath` matches zero rows so it is omitted.
const QUERIES: &[&str] = &[
    "dump_eq",     // dump = 'CC-MAIN-2016-30'           — clustered, ~7%
    "date_prefix", // date LIKE '2020-10-%'              — clustered, ~12%
    "google_and",  // url LIKE '%google%' AND text LIKE  — very selective, scattered
    "google_or",   // url/text LIKE '%google%'           — scattered, ~2%
    "vortex",      // text LIKE '% vortex %'             — tiny
    "espn_and",    // url LIKE '%espn%' AND lang/score   — tiny
    "espn_or",     // url LIKE '%espn%' OR ...           — tiny
];

/// The materialized string column. `url` is short (~72 B), `text` is long (~3 KB).
const COLUMNS: &[&str] = &["url", "text"];

fn dir() -> Option<PathBuf> {
    std::env::var_os("FINEWEB_DIR").map(PathBuf::from)
}

/// Read a length-prefixed column dump into a `VarBinArray`.
fn load_column(name: &str) -> Option<VarBinArray> {
    let path = dir()?.join(format!("fw_{name}.bin"));
    if !path.exists() {
        return None;
    }
    let bytes = std::fs::read(path).unwrap();
    let mut pos = 0usize;
    #[expect(clippy::cast_possible_truncation)]
    let rows = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    pos += 8;
    let mut values: Vec<Option<Vec<u8>>> = Vec::with_capacity(rows);
    for _ in 0..rows {
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

/// Read a one-byte-per-row predicate mask.
fn load_mask(query: &str) -> Option<Mask> {
    let path = dir()?.join(format!("fw_mask_{query}.bin"));
    if !path.exists() {
        return None;
    }
    let bytes = std::fs::read(path).unwrap();
    #[expect(clippy::cast_possible_truncation)]
    let rows = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    Some(Mask::from_iter((0..rows).map(|i| bytes[8 + i] != 0)))
}

fn compress(varbin: &VarBinArray, ctx: &mut ExecutionCtx) -> FSSTArray {
    let compressor = fsst_train_compressor(varbin);
    fsst_compress(varbin, varbin.len(), varbin.dtype(), &compressor, ctx)
}

#[derive(Clone, Copy)]
struct Case {
    column: &'static str,
    query: &'static str,
}

impl std::fmt::Display for Case {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.column, self.query)
    }
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    for &column in COLUMNS {
        for &query in QUERIES {
            v.push(Case { column, query });
        }
    }
    v
}

#[divan::bench(args = cases())]
fn fsst(bencher: Bencher, case: Case) {
    let (Some(varbin), Some(mask)) = (load_column(case.column), load_mask(case.query)) else {
        return;
    };
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
    bencher
        .with_inputs(|| (&fsst, &mask, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(fsst, mask, ctx)| {
            let filtered: ArrayRef = <FSST as FilterKernel>::filter(fsst.as_view(), mask, ctx)
                .unwrap()
                .unwrap();
            black_box(
                filtered
                    .execute::<VarBinViewArray>(ctx)
                    .unwrap()
                    .into_array(),
            )
        });
}

#[divan::bench(args = cases())]
fn view(bencher: Bencher, case: Case) {
    let (Some(varbin), Some(mask)) = (load_column(case.column), load_mask(case.query)) else {
        return;
    };
    let fsst = compress(&varbin, &mut LEGACY_SESSION.create_execution_ctx());
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
