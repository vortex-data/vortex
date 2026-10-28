// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Experiment-only knobs for sizing V2 chunks, read from the environment.
//!
//! - `VX_FILTER_SPLIT`: `rows:N` cuts filtered chunks of `N` rows; `bytes:B` cuts chunks holding
//!   about `B` uncompressed bytes of the filter's columns.
//! - `VX_PROJ_SPLIT`: `rows:N` projects a chunk's survivors in windows of `N` rows; `bytes:B` in
//!   windows whose survivors hold about `B` uncompressed bytes of the projection's columns.
//! - `VX_COL_BYTES`: `name=bytes,...`, each column's uncompressed bytes per row. Columns not
//!   listed count 8 bytes.

use std::sync::LazyLock;

use rustc_hash::FxHashMap;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::select::Select;

use crate::scan::execution::ROW_BLOCK_SIZE;

/// How a chunk is cut.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Split {
    Rows(u64),
    Bytes(u64),
}

fn parse(var: &str) -> Option<Split> {
    let value = std::env::var(var).ok()?;
    let (kind, n) = value.split_once(':')?;
    let n: u64 = n.parse().ok()?;
    match kind {
        "rows" => Some(Split::Rows(n)),
        "bytes" => Some(Split::Bytes(n)),
        _ => None,
    }
}

static FILTER_SPLIT: LazyLock<Option<Split>> = LazyLock::new(|| parse("VX_FILTER_SPLIT"));
static PROJ_SPLIT: LazyLock<Option<Split>> = LazyLock::new(|| parse("VX_PROJ_SPLIT"));
static COL_BYTES: LazyLock<FxHashMap<String, u64>> = LazyLock::new(|| {
    std::env::var("VX_COL_BYTES")
        .unwrap_or_default()
        .split(',')
        .filter_map(|pair| {
            let (name, bytes) = pair.split_once('=')?;
            Some((name.trim().to_owned(), bytes.trim().parse().ok()?))
        })
        .collect()
});

pub(crate) fn filter_split() -> Option<Split> {
    *FILTER_SPLIT
}

pub(crate) fn proj_split() -> Option<Split> {
    *PROJ_SPLIT
}

/// The uncompressed bytes per row of the top-level columns `expr` reads.
pub(crate) fn bytes_per_row(expr: &BoundExpression, root: &DType) -> u64 {
    let mut fields = Vec::new();
    collect_fields(expr, root, &mut fields);
    fields.sort_unstable();
    fields.dedup();
    fields
        .iter()
        .map(|name| COL_BYTES.get(name).copied().unwrap_or(8))
        .sum::<u64>()
        .max(1)
}

fn struct_names(dtype: &DType, out: &mut Vec<String>) {
    if let Some(fields) = dtype.as_struct_fields_opt() {
        out.extend(fields.names().iter().map(|name| name.to_string()));
    }
}

fn collect_fields(expr: &BoundExpression, root: &DType, out: &mut Vec<String>) {
    if expr.is_root() {
        struct_names(root, out);
        return;
    }
    if expr.children().len() == 1 && expr.child(0).is_root() {
        if let Some(name) = expr.as_opt::<GetItem>() {
            out.push(name.to_string());
            return;
        }
        if expr.is::<Select>() {
            struct_names(expr.dtype(), out);
            return;
        }
    }
    for child in expr.children() {
        collect_fields(child, root, out);
    }
}

/// Rows per filtered chunk for a filter reading `filter_bpr` bytes per row, when set.
pub(crate) fn filtered_chunk_rows(filter_bpr: u64) -> Option<usize> {
    let rows = match filter_split()? {
        Split::Rows(rows) => rows,
        Split::Bytes(bytes) => bytes / filter_bpr,
    };
    let block = ROW_BLOCK_SIZE as u64;
    let rows = (rows / block).clamp(1, 512) * block;
    usize::try_from(rows).ok()
}

/// Rows per projection window of a chunk of `len` rows with `selected` survivors, when set.
pub(crate) fn projection_window(proj_bpr: u64, len: usize, selected: usize) -> Option<usize> {
    let rows = match proj_split()? {
        Split::Rows(rows) => rows,
        Split::Bytes(bytes) => {
            let survivor_bytes = (selected as u64).max(1) * proj_bpr;
            bytes.saturating_mul(len as u64) / survivor_bytes.max(1)
        }
    };
    let rows = (rows / 1024).max(1) * 1024;
    usize::try_from(rows).ok()
}
