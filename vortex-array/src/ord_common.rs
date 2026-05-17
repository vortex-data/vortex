// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared minimal encoded column shapes used by the OVC SMJ design modules.
//!
//! These are stand-ins for Vortex's real `Array` types — direct slice
//! references so each design module can be self-contained and benched
//! without depending on the full Vortex ArrayRef plumbing.
//!
//! Three design modules build on these:
//!   * `ord_iter`    — the converged `OrdIter` trait + chunked dispatch
//!   * `ord_direct`  — hand-specialised per-encoding merge functions
//!   * `ord_memcmp`  — materialize to ord-bytes once, then byte-cmp merge
//!
//! See `docs/developer-guide/internals/smj-ovc-design.md`.

#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names
)]

/// Plain primitive column.
pub(crate) struct PrimI64<'a> {
    pub data: &'a [i64],
}

/// Dict-encoded column with a sorted, rank-aligned dictionary.
/// Code order == value order, so the merge driver can compare codes.
pub(crate) struct DictI64<'a> {
    pub codes: &'a [u32],
    pub dict: &'a [i64],
}

/// Run-end-encoded column: `run_ends[k]` is the first row NOT in run k;
/// `values[k]` is run k's value. Sorted runs (monotone values).
pub(crate) struct RunEndI64<'a> {
    pub run_ends: &'a [u32],
    pub values: &'a [i64],
    pub len: usize,
}

impl<'a> RunEndI64<'a> {
    /// Locate the run index containing logical row `row` (binary search).
    #[inline]
    pub(crate) fn run_of(&self, row: usize) -> usize {
        self.run_ends.partition_point(|&e| (e as usize) <= row)
    }
    /// Find the run index containing `row`, starting from a monotone hint.
    #[inline]
    pub(crate) fn run_of_hint(&self, row: usize, start_run: usize) -> usize {
        let mut r = start_run;
        while r < self.run_ends.len() && (self.run_ends[r] as usize) <= row {
            r += 1;
        }
        r
    }
}

/// Single-value-everywhere column.
pub(crate) struct ConstantI64 {
    pub value: i64,
    pub len: usize,
}

/// Variable-length binary column (offsets + data buffer).
pub(crate) struct VarBin<'a> {
    pub offsets: &'a [u32],
    pub data: &'a [u8],
}

impl<'a> VarBin<'a> {
    #[inline]
    pub(crate) fn bytes_at(&self, row: usize) -> &'a [u8] {
        let s = self.offsets[row] as usize;
        let e = self.offsets[row + 1] as usize;
        &self.data[s..e]
    }
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Test data builders. Shared across modules so each design exercises the
// same shape of data; the only difference is how it's consumed.
// ───────────────────────────────────────────────────────────────────────────

/// Sorted primitive data starting at `start`.
pub(crate) fn build_prim(n: usize, start: i64) -> Vec<i64> {
    (0..n as i64).map(|i| start + i).collect()
}

/// Sorted dict: `distinct` ascending dictionary entries, monotone codes.
pub(crate) fn build_dict(n: usize, distinct: usize, start: i64) -> (Vec<u32>, Vec<i64>) {
    let dict: Vec<i64> = (0..distinct as i64).map(|i| start + i * 17).collect();
    let codes: Vec<u32> = (0..n).map(|i| (i * distinct / n) as u32).collect();
    (codes, dict)
}

/// Sorted run-end: `runs` runs, each `run_len` rows long, monotone values.
pub(crate) fn build_runend(runs: usize, run_len: usize, start: i64) -> (Vec<u32>, Vec<i64>, usize) {
    let mut ends = Vec::with_capacity(runs);
    let mut values = Vec::with_capacity(runs);
    for r in 0..runs {
        ends.push(((r + 1) * run_len) as u32);
        values.push(start + r as i64 * 13);
    }
    (ends, values, runs * run_len)
}

/// VarBin with fixed-stride `value_len` per row and the row's leading u64
/// derived from `start + row` (so sides are disjoint when seeded differently).
pub(crate) fn build_varbin(n: usize, value_len: usize, start: u64) -> (Vec<u32>, Vec<u8>) {
    let mut data = vec![0u8; n * value_len];
    for row in 0..n {
        let key = (start + row as u64).to_be_bytes();
        data[row * value_len..row * value_len + 8].copy_from_slice(&key);
    }
    let offsets: Vec<u32> = (0..=n).map(|i| (i * value_len) as u32).collect();
    (offsets, data)
}
