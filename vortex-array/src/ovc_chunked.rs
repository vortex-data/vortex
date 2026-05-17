// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chunked-export OVC: each encoding implements `OrdScan` which writes a
//! chunk of ord-bytes (u64s) into caller-supplied scratch. The merge driver
//! reads from typed `&[u64]` scratch buffers — no dyn dispatch in the inner
//! loop. The dyn call is one-per-chunk and amortises to negligible.
//!
//! This validates the design from the chat: open extensibility (new
//! encodings impl OrdScan, get a canonicalize-default for free) plus
//! typed-inner-loop performance.

#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names
)]

use crate::ovc_encoded::{ChunkedPrim, ConstantI64, DictI64, PrimI64, RunEndI64, VarBin};

/// Order-preserving u64 scan trait. Encodings opt into a typed fast path
/// by overriding `scan_ord`; the default would canonicalize and recurse
/// (omitted here — we're benchmarking only the fast paths).
pub trait OrdScan {
    fn ord_len(&self) -> usize;
    /// Write rows [start, start+len) as order-preserving u64 into `out[..len]`.
    fn scan_ord(&self, start: usize, len: usize, out: &mut [u64]);
}

// ───────────────────────────────────────────────────────────────────────────
// OrdScan impls. Each is a tight typed loop — no dyn in the inner body.
// ───────────────────────────────────────────────────────────────────────────

impl OrdScan for PrimI64<'_> {
    fn ord_len(&self) -> usize {
        self.data.len()
    }
    fn scan_ord(&self, start: usize, len: usize, out: &mut [u64]) {
        let src = &self.data[start..start + len];
        for (i, &v) in src.iter().enumerate() {
            out[i] = (v as u64) ^ (1u64 << 63);
        }
    }
}

impl OrdScan for DictI64<'_> {
    fn ord_len(&self) -> usize {
        self.codes.len()
    }
    fn scan_ord(&self, start: usize, len: usize, out: &mut [u64]) {
        let codes = &self.codes[start..start + len];
        for (i, &c) in codes.iter().enumerate() {
            // Rank-aligned codes pack into high bits so order is preserved.
            out[i] = u64::from(c) << 32;
        }
    }
}

impl OrdScan for RunEndI64<'_> {
    fn ord_len(&self) -> usize {
        self.len
    }
    fn scan_ord(&self, start: usize, len: usize, out: &mut [u64]) {
        let end = start + len;
        let mut row = start;
        let mut run = self.run_of(start);
        while row < end {
            let run_end = (self.run_ends[run] as usize).min(end);
            let u = (self.values[run] as u64) ^ (1u64 << 63);
            for i in (row - start)..(run_end - start) {
                out[i] = u;
            }
            row = run_end;
            run += 1;
        }
    }
}

impl OrdScan for VarBin<'_> {
    fn ord_len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    fn scan_ord(&self, start: usize, len: usize, out: &mut [u64]) {
        for i in 0..len {
            let row = start + i;
            let s = self.offsets[row] as usize;
            let e = self.offsets[row + 1] as usize;
            let bytes = &self.data[s..e];
            let mut buf = [0u8; 8];
            let n = bytes.len().min(8);
            buf[..n].copy_from_slice(&bytes[..n]);
            out[i] = u64::from_be_bytes(buf);
        }
    }
}

impl OrdScan for ChunkedPrim<'_> {
    fn ord_len(&self) -> usize {
        self.len
    }
    fn scan_ord(&self, start: usize, len: usize, out: &mut [u64]) {
        // Find starting chunk and walk forward.
        let mut k = 0;
        while k < self.chunks.len() && self.chunk_offsets[k + 1] <= start {
            k += 1;
        }
        let mut row = start;
        let end = start + len;
        while row < end {
            let chunk = self.chunks[k];
            let chunk_start = self.chunk_offsets[k];
            let local_start = row - chunk_start;
            let local_end = (end - chunk_start).min(chunk.len());
            for i in local_start..local_end {
                out[row - start + i - local_start] =
                    (chunk[i] as u64) ^ (1u64 << 63);
            }
            row = chunk_start + local_end;
            k += 1;
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// STRUCTURED CHUNKS
//
// Instead of always flattening to a u64 per row, a chunk can carry the
// encoding's natural shape. The merge driver then uses a fast path per
// variant:
//   Constant : emit `len` rows of one value, no per-row work
//   RunEnd   : walk run headers, emit run_len rows per boundary
//   Dense    : universal fallback — one u64 per row, scan as before
//
// Each chunk borrows from caller-owned scratch. Scratch is sized once at
// the start of the merge; chunks point into different parts of it
// depending on their kind.
// ───────────────────────────────────────────────────────────────────────────

/// Caller-owned scratch backing a single side's current chunk.
pub struct Scratch {
    dense: Vec<u64>,
    run_ends: Vec<u32>,
    run_values: Vec<u64>,
}

impl Scratch {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            dense: vec![0u64; chunk_size],
            run_ends: vec![0u32; chunk_size], // at most chunk_size runs in a chunk
            run_values: vec![0u64; chunk_size],
        }
    }
}

/// A chunk's structural shape. Borrows from caller-owned `Scratch`.
pub enum OrdChunk<'a> {
    /// `len` consecutive rows all equal to `value`.
    Constant { value: u64, len: usize },
    /// `n_runs` runs starting at chunk-local row 0. `run_ends[k]` is the
    /// first chunk-local row index NOT in run k. `values[k]` is run k's
    /// ord value.
    RunEnd { run_ends: &'a [u32], values: &'a [u64], len: usize },
    /// One u64 per row.
    Dense(&'a [u64]),
}

impl OrdChunk<'_> {
    pub fn len(&self) -> usize {
        match self {
            OrdChunk::Constant { len, .. } => *len,
            OrdChunk::RunEnd { len, .. } => *len,
            OrdChunk::Dense(v) => v.len(),
        }
    }
}

/// Structured chunked scan: the encoding produces the BEST chunk variant
/// it can. New encodings get `Dense` for free (via the default impl).
pub trait OrdScanStructured {
    fn ord_len(&self) -> usize;

    /// Fill at most `max_len` rows starting at `start`, returning the chunk.
    /// Default: produce a Dense chunk via `scan_ord`. Override to expose
    /// structural shape (Constant / RunEnd).
    fn scan_chunk<'s>(
        &self,
        start: usize,
        max_len: usize,
        scratch: &'s mut Scratch,
    ) -> OrdChunk<'s>;
}

impl OrdScanStructured for PrimI64<'_> {
    fn ord_len(&self) -> usize {
        self.data.len()
    }
    fn scan_chunk<'s>(
        &self,
        start: usize,
        max_len: usize,
        scratch: &'s mut Scratch,
    ) -> OrdChunk<'s> {
        let take = max_len.min(self.data.len() - start);
        for (i, &v) in self.data[start..start + take].iter().enumerate() {
            scratch.dense[i] = (v as u64) ^ (1u64 << 63);
        }
        OrdChunk::Dense(&scratch.dense[..take])
    }
}

impl OrdScanStructured for DictI64<'_> {
    fn ord_len(&self) -> usize {
        self.codes.len()
    }
    fn scan_chunk<'s>(
        &self,
        start: usize,
        max_len: usize,
        scratch: &'s mut Scratch,
    ) -> OrdChunk<'s> {
        let take = max_len.min(self.codes.len() - start);
        for (i, &c) in self.codes[start..start + take].iter().enumerate() {
            scratch.dense[i] = u64::from(c) << 32;
        }
        OrdChunk::Dense(&scratch.dense[..take])
    }
}

impl OrdScanStructured for ConstantI64 {
    fn ord_len(&self) -> usize {
        self.len
    }
    fn scan_chunk<'s>(
        &self,
        start: usize,
        max_len: usize,
        _scratch: &'s mut Scratch,
    ) -> OrdChunk<'s> {
        let take = max_len.min(self.len - start);
        OrdChunk::Constant {
            value: (self.value as u64) ^ (1u64 << 63),
            len: take,
        }
    }
}

impl OrdScanStructured for RunEndI64<'_> {
    fn ord_len(&self) -> usize {
        self.len
    }
    fn scan_chunk<'s>(
        &self,
        start: usize,
        max_len: usize,
        scratch: &'s mut Scratch,
    ) -> OrdChunk<'s> {
        let end = (start + max_len).min(self.len);
        let take = end - start;
        // Find runs overlapping [start, end). Re-encode local run_ends so
        // the first run starts at chunk-local 0.
        let mut run = self.run_of(start);
        let mut n_runs = 0;
        let mut row = start;
        while row < end {
            let run_end_global = (self.run_ends[run] as usize).min(end);
            let local_end = (run_end_global - start) as u32;
            scratch.run_ends[n_runs] = local_end;
            scratch.run_values[n_runs] = (self.values[run] as u64) ^ (1u64 << 63);
            n_runs += 1;
            row = run_end_global;
            run += 1;
        }
        OrdChunk::RunEnd {
            run_ends: &scratch.run_ends[..n_runs],
            values: &scratch.run_values[..n_runs],
            len: take,
        }
    }
}

impl OrdScanStructured for VarBin<'_> {
    fn ord_len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    fn scan_chunk<'s>(
        &self,
        start: usize,
        max_len: usize,
        scratch: &'s mut Scratch,
    ) -> OrdChunk<'s> {
        let n_rows = self.offsets.len().saturating_sub(1);
        let take = max_len.min(n_rows - start);
        for i in 0..take {
            let row = start + i;
            let s = self.offsets[row] as usize;
            let e = self.offsets[row + 1] as usize;
            let bytes = &self.data[s..e];
            let mut buf = [0u8; 8];
            let n = bytes.len().min(8);
            buf[..n].copy_from_slice(&bytes[..n]);
            scratch.dense[i] = u64::from_be_bytes(buf);
        }
        OrdChunk::Dense(&scratch.dense[..take])
    }
}

/// Structured n-way merge. Per-side cursor advances the active chunk's
/// shape; refill on exhaustion. Constant and RunEnd chunks let the inner
/// loop emit multiple rows per iteration without per-row compares against
/// the same side.
pub fn merge_n_way_structured(sides: &[&dyn OrdScanStructured], chunk_size: usize) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut scratches: Vec<Scratch> = (0..n).map(|_| Scratch::new(chunk_size)).collect();
    let mut chunk_base = vec![0usize; n];
    let mut chunk_len = vec![0usize; n]; // rows the current chunk covers
    // Per-side cursor — meaning varies by chunk variant.
    let mut chunk_pos = vec![0usize; n];
    // Per-side current chunk kind + cached fast-path state.
    enum SideKind {
        Constant { value: u64, remaining: usize },
        RunEnd { run_idx: usize, n_runs: usize, run_pos_within: usize, run_len_within: usize },
        Dense, // chunk_pos indexes into scratch.dense[..chunk_len]
        Empty,
    }
    let mut kinds: Vec<SideKind> = (0..n).map(|_| SideKind::Empty).collect();

    let refill = |i: usize,
                  scratch: &mut Scratch,
                  chunk_base: &mut [usize],
                  chunk_len: &mut [usize],
                  chunk_pos: &mut [usize],
                  kinds: &mut [SideKind]| {
        let next_start = chunk_base[i] + chunk_len[i];
        let rem = sides[i].ord_len().saturating_sub(next_start);
        if rem == 0 {
            kinds[i] = SideKind::Empty;
            return;
        }
        let take = rem.min(chunk_size);
        let chunk = sides[i].scan_chunk(next_start, take, scratch);
        chunk_base[i] = next_start;
        chunk_len[i] = chunk.len();
        chunk_pos[i] = 0;
        kinds[i] = match chunk {
            OrdChunk::Constant { value, len } => SideKind::Constant { value, remaining: len },
            OrdChunk::RunEnd { run_ends, values: _, len: _ } => {
                let n_runs = run_ends.len();
                let first_run_len = if n_runs > 0 { run_ends[0] as usize } else { 0 };
                SideKind::RunEnd {
                    run_idx: 0,
                    n_runs,
                    run_pos_within: 0,
                    run_len_within: first_run_len,
                }
            }
            OrdChunk::Dense(_) => SideKind::Dense,
        };
    };

    // Prime all sides.
    for i in 0..n {
        refill(i, &mut scratches[i], &mut chunk_base, &mut chunk_len, &mut chunk_pos, &mut kinds);
    }

    let mut count = 0usize;
    loop {
        // Determine each side's current head value (without consuming).
        let mut min_v = u64::MAX;
        let mut min_side = usize::MAX;
        for i in 0..n {
            let cur = match &kinds[i] {
                SideKind::Empty => continue,
                SideKind::Constant { value, .. } => *value,
                SideKind::RunEnd { run_idx, .. } => scratches[i].run_values[*run_idx],
                SideKind::Dense => scratches[i].dense[chunk_pos[i]],
            };
            if cur < min_v {
                min_v = cur;
                min_side = i;
            }
        }
        if min_side == usize::MAX {
            break;
        }
        count += 1;
        // Advance the winner.
        match &mut kinds[min_side] {
            SideKind::Constant { remaining, .. } => {
                *remaining -= 1;
                chunk_pos[min_side] += 1;
                if *remaining == 0 {
                    refill(
                        min_side,
                        &mut scratches[min_side],
                        &mut chunk_base,
                        &mut chunk_len,
                        &mut chunk_pos,
                        &mut kinds,
                    );
                }
            }
            SideKind::RunEnd {
                run_idx,
                n_runs,
                run_pos_within,
                run_len_within,
            } => {
                *run_pos_within += 1;
                chunk_pos[min_side] += 1;
                if *run_pos_within >= *run_len_within {
                    *run_idx += 1;
                    if *run_idx >= *n_runs {
                        refill(
                            min_side,
                            &mut scratches[min_side],
                            &mut chunk_base,
                            &mut chunk_len,
                            &mut chunk_pos,
                            &mut kinds,
                        );
                    } else {
                        let prev_end = scratches[min_side].run_ends[*run_idx - 1] as usize;
                        let cur_end = scratches[min_side].run_ends[*run_idx] as usize;
                        *run_pos_within = 0;
                        *run_len_within = cur_end - prev_end;
                    }
                }
            }
            SideKind::Dense => {
                chunk_pos[min_side] += 1;
                if chunk_pos[min_side] >= chunk_len[min_side] {
                    refill(
                        min_side,
                        &mut scratches[min_side],
                        &mut chunk_base,
                        &mut chunk_len,
                        &mut chunk_pos,
                        &mut kinds,
                    );
                }
            }
            SideKind::Empty => unreachable!(),
        }
    }
    count
}

// ───────────────────────────────────────────────────────────────────────────
// Chunked n-way merge driver. Single dyn call per chunk (refill); typed
// inner loop over &[u64] scratch buffers. For single-column, "OVC" is just
// the ord-value itself; min wins, ties advance one side at a time.
// ───────────────────────────────────────────────────────────────────────────

pub fn merge_n_way_chunked(sides: &[&dyn OrdScan], chunk_size: usize) -> usize {
    let n = sides.len();
    if n == 0 {
        return 0;
    }
    let mut scratch: Vec<Vec<u64>> = (0..n).map(|_| vec![0u64; chunk_size]).collect();
    let mut chunk_base = vec![0usize; n]; // absolute row where the current chunk starts
    let mut chunk_fill = vec![0usize; n]; // valid rows in scratch
    let mut chunk_pos = vec![0usize; n]; // cursor within the chunk

    // Prime each side.
    for i in 0..n {
        let take = sides[i].ord_len().min(chunk_size);
        if take > 0 {
            sides[i].scan_ord(0, take, &mut scratch[i][..take]);
            chunk_fill[i] = take;
        }
    }

    let mut count = 0usize;
    loop {
        // Refill any side whose chunk is exhausted.
        for i in 0..n {
            if chunk_pos[i] >= chunk_fill[i] {
                let next_start = chunk_base[i] + chunk_fill[i];
                let rem = sides[i].ord_len().saturating_sub(next_start);
                if rem == 0 {
                    chunk_pos[i] = chunk_fill[i]; // permanently exhausted
                    continue;
                }
                let take = rem.min(chunk_size);
                sides[i].scan_ord(next_start, take, &mut scratch[i][..take]);
                chunk_base[i] = next_start;
                chunk_fill[i] = take;
                chunk_pos[i] = 0;
            }
        }

        // Typed inner loop: scan n scratch[*][chunk_pos[*]] u64s for the min.
        let mut min_v = u64::MAX;
        let mut min_side = usize::MAX;
        for i in 0..n {
            if chunk_pos[i] < chunk_fill[i] {
                let v = scratch[i][chunk_pos[i]];
                if v < min_v {
                    min_v = v;
                    min_side = i;
                }
            }
        }
        if min_side == usize::MAX {
            break;
        }
        count += 1;
        chunk_pos[min_side] += 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::ovc_encoded::{
        DictI64, PrimI64, RunEndI64, VarBin, materialize_dict, materialize_prim,
        materialize_runend, merge_n_way_memcmp, merge_n_way_ovc_dict, merge_n_way_ovc_prim,
        merge_n_way_ovc_runend, merge_n_way_ovc_varbin,
    };

    use super::*;

    #[test]
    fn agreement_chunked_prim() {
        let s0: Vec<i64> = (0..50).collect();
        let s1: Vec<i64> = (50..100).collect();
        let prim_sides = vec![PrimI64 { data: &s0 }, PrimI64 { data: &s1 }];
        let dyn_sides: Vec<&dyn OrdScan> = vec![&prim_sides[0], &prim_sides[1]];
        assert_eq!(merge_n_way_chunked(&dyn_sides, 16), 100);
        assert_eq!(merge_n_way_ovc_prim(&prim_sides), 100);
    }

    #[test]
    fn agreement_chunked_dict() {
        let codes0: Vec<u32> = (0..50).map(|i| (i / 5) as u32).collect();
        let dict0: Vec<i64> = (0..10).map(|i| i * 17).collect();
        let codes1: Vec<u32> = (0..50).map(|i| (i / 5) as u32).collect();
        let dict1: Vec<i64> = (10..20).map(|i| i * 17).collect();
        let dict_sides = vec![
            DictI64 { codes: &codes0, dict: &dict0 },
            DictI64 { codes: &codes1, dict: &dict1 },
        ];
        let dyn_sides: Vec<&dyn OrdScan> = vec![&dict_sides[0], &dict_sides[1]];
        let chunked = merge_n_way_chunked(&dyn_sides, 16);
        let direct = merge_n_way_ovc_dict(&dict_sides);
        assert_eq!(chunked, 100);
        assert_eq!(direct, 100);
    }

    #[test]
    fn agreement_structured_prim() {
        let s0: Vec<i64> = (0..50).collect();
        let s1: Vec<i64> = (50..100).collect();
        let p = [PrimI64 { data: &s0 }, PrimI64 { data: &s1 }];
        let sides: Vec<&dyn OrdScanStructured> = vec![&p[0], &p[1]];
        assert_eq!(merge_n_way_structured(&sides, 16), 100);
    }

    #[test]
    fn agreement_structured_constant() {
        use crate::ovc_encoded::ConstantI64;
        let sides_owned =
            [ConstantI64 { value: 1, len: 10 }, ConstantI64 { value: 5, len: 10 }];
        let sides: Vec<&dyn OrdScanStructured> = vec![&sides_owned[0], &sides_owned[1]];
        assert_eq!(merge_n_way_structured(&sides, 4), 20);
    }

    #[test]
    fn agreement_structured_runend() {
        let e0: Vec<u32> = (1u32..=10).map(|i| i * 5).collect();
        let v0: Vec<i64> = (0..10).map(|i| i * 2).collect();
        let e1: Vec<u32> = (1u32..=5).map(|i| i * 13).collect();
        let v1: Vec<i64> = (0..5).map(|i| i * 2 + 1).collect();
        let re = [
            RunEndI64 { run_ends: &e0, values: &v0, len: 50 },
            RunEndI64 { run_ends: &e1, values: &v1, len: 65 },
        ];
        let sides: Vec<&dyn OrdScanStructured> = vec![&re[0], &re[1]];
        assert_eq!(merge_n_way_structured(&sides, 16), 115);
    }

    #[test]
    fn agreement_chunked_runend() {
        let e0: Vec<u32> = (1u32..=10).map(|i| i * 5).collect();
        let v0: Vec<i64> = (0..10).map(|i| i * 2).collect();
        let e1: Vec<u32> = (1u32..=5).map(|i| i * 13).collect();
        let v1: Vec<i64> = (0..5).map(|i| i * 2 + 1).collect();
        let runend_sides = vec![
            RunEndI64 { run_ends: &e0, values: &v0, len: 50 },
            RunEndI64 { run_ends: &e1, values: &v1, len: 65 },
        ];
        let dyn_sides: Vec<&dyn OrdScan> = vec![&runend_sides[0], &runend_sides[1]];
        assert_eq!(merge_n_way_chunked(&dyn_sides, 16), 115);
        assert_eq!(merge_n_way_ovc_runend(&runend_sides), 115);
    }

    /// Compare chunked-dispatch vs direct-OVC vs materialize+memcmp across
    /// encodings and chunk sizes. The hypothesis: at chunk_size ≥ 64, the
    /// dyn dispatch cost vanishes and chunked-dispatch matches direct OVC
    /// (within encoding-specific overhead).
    ///
    /// Run: cargo test --release -p vortex-array ovc_chunked::tests::bench \
    ///     -- --ignored --nocapture --test-threads=1
    #[test]
    #[ignore = "benchmark, run explicitly"]
    #[allow(clippy::cast_precision_loss)]
    fn bench_chunked_vs_direct() {
        const N: usize = 50_000;
        const N_SIDES: usize = 8;
        const ITERS: u32 = 10;
        const CHUNK_SIZES: &[usize] = &[1, 8, 64, 256, 1024, 4096];

        // Build data once per encoding.
        let prim_data: Vec<Vec<i64>> = (0..N_SIDES)
            .map(|i| (0..N as i64).map(|j| (i * N) as i64 + j).collect())
            .collect();
        let prim_sides: Vec<PrimI64> = prim_data.iter().map(|d| PrimI64 { data: d }).collect();

        let dict_data: Vec<(Vec<u32>, Vec<i64>)> = (0..N_SIDES)
            .map(|i| {
                let dict: Vec<i64> = (0..256).map(|x| (i * N) as i64 * 17 + x * 17).collect();
                let codes: Vec<u32> = (0..N).map(|j| (j * 256 / N) as u32).collect();
                (codes, dict)
            })
            .collect();
        let dict_sides: Vec<DictI64> = dict_data
            .iter()
            .map(|(c, d)| DictI64 { codes: c, dict: d })
            .collect();

        let runend_data: Vec<(Vec<u32>, Vec<i64>, usize)> = (0..N_SIDES)
            .map(|i| {
                let runs = 500;
                let run_len = N / runs;
                let ends: Vec<u32> = (1..=runs).map(|r| (r * run_len) as u32).collect();
                let values: Vec<i64> =
                    (0..runs).map(|r| (i * N) as i64 * 13 + r as i64 * 13).collect();
                (ends, values, runs * run_len)
            })
            .collect();
        let runend_sides: Vec<RunEndI64> = runend_data
            .iter()
            .map(|(e, v, n)| RunEndI64 { run_ends: e, values: v, len: *n })
            .collect();

        let vb_offsets: Vec<Vec<u32>> =
            (0..N_SIDES).map(|_| (0..=N).map(|i| (i * 50) as u32).collect()).collect();
        let vb_data: Vec<Vec<u8>> = (0..N_SIDES)
            .map(|side| {
                let mut buf = vec![0u8; N * 50];
                for row in 0..N {
                    let key = ((side * N + row) as u64).to_be_bytes();
                    buf[row * 50..row * 50 + 8].copy_from_slice(&key);
                }
                buf
            })
            .collect();
        let vb_sides: Vec<VarBin> = vb_offsets
            .iter()
            .zip(vb_data.iter())
            .map(|(o, d)| VarBin { offsets: o, data: d })
            .collect();

        let total_rows = (N * N_SIDES) as u64;

        let bench_chunks = |label: &str, dyn_sides: Vec<&dyn OrdScan>| {
            println!("\n  -- {label} -- chunked Dense-only (dyn per chunk, u64 scratch)");
            for &cs in CHUNK_SIZES {
                let _ = merge_n_way_chunked(&dyn_sides, cs);
                let t = Instant::now();
                let mut acc = 0u64;
                for _ in 0..ITERS {
                    acc =
                        acc.wrapping_add(std::hint::black_box(
                            merge_n_way_chunked(&dyn_sides, cs) as u64,
                        ));
                }
                let ns = t.elapsed().as_nanos() as f64
                    / (u64::from(ITERS) * total_rows) as f64;
                println!("    chunk_size={cs:>5}  {ns:>8.2} ns/row   acc={acc}");
            }
        };

        let bench_structured =
            |label: &str, dyn_sides: Vec<&dyn OrdScanStructured>| {
                println!(
                    "\n  -- {label} -- structured chunked (Constant/RunEnd/Dense variants)"
                );
                for &cs in CHUNK_SIZES {
                    let _ = merge_n_way_structured(&dyn_sides, cs);
                    let t = Instant::now();
                    let mut acc = 0u64;
                    for _ in 0..ITERS {
                        acc = acc.wrapping_add(std::hint::black_box(
                            merge_n_way_structured(&dyn_sides, cs) as u64,
                        ));
                    }
                    let ns = t.elapsed().as_nanos() as f64
                        / (u64::from(ITERS) * total_rows) as f64;
                    println!("    chunk_size={cs:>5}  {ns:>8.2} ns/row   acc={acc}");
                }
            };

        fn run_one(
            label: &str,
            iters: u32,
            total_rows: u64,
            mut f: impl FnMut() -> u64,
        ) {
            let _ = f();
            let t = Instant::now();
            let mut acc = 0u64;
            for _ in 0..iters {
                acc = acc.wrapping_add(std::hint::black_box(f()));
            }
            let ns = t.elapsed().as_nanos() as f64 / (u64::from(iters) * total_rows) as f64;
            println!("    {:<36} {:>8.2} ns/row   acc={acc}", label, ns);
        }

        println!("\n== 8-way merge, {N} rows/side, disjoint, single i64 col ==");

        // PRIMITIVE
        println!("\n-- PRIMITIVE --");
        run_one("direct OVC over primitive", ITERS, total_rows, || {
            merge_n_way_ovc_prim(&prim_sides) as u64
        });
        {
            let mats: Vec<Vec<u8>> = prim_sides.iter().map(materialize_prim).collect();
            let refs: Vec<&[u8]> = mats.iter().map(Vec::as_slice).collect();
            run_one("materialize + memcmp", ITERS, total_rows, || {
                merge_n_way_memcmp(&refs) as u64
            });
        }
        let dyn_prim: Vec<&dyn OrdScan> = prim_sides.iter().map(|s| s as &dyn OrdScan).collect();
        bench_chunks("PRIMITIVE", dyn_prim);
        let dyn_prim_s: Vec<&dyn OrdScanStructured> =
            prim_sides.iter().map(|s| s as &dyn OrdScanStructured).collect();
        bench_structured("PRIMITIVE", dyn_prim_s);

        // DICT
        println!("\n-- DICT (256 distinct vals) --");
        run_one("direct OVC over dict (codes)", ITERS, total_rows, || {
            merge_n_way_ovc_dict(&dict_sides) as u64
        });
        {
            let mats: Vec<Vec<u8>> = dict_sides.iter().map(materialize_dict).collect();
            let refs: Vec<&[u8]> = mats.iter().map(Vec::as_slice).collect();
            run_one("materialize + memcmp", ITERS, total_rows, || {
                merge_n_way_memcmp(&refs) as u64
            });
        }
        let dyn_dict: Vec<&dyn OrdScan> = dict_sides.iter().map(|s| s as &dyn OrdScan).collect();
        bench_chunks("DICT", dyn_dict);
        let dyn_dict_s: Vec<&dyn OrdScanStructured> =
            dict_sides.iter().map(|s| s as &dyn OrdScanStructured).collect();
        bench_structured("DICT", dyn_dict_s);

        // RUN-END (long runs)
        println!("\n-- RUN-END (avg 100 rows/run) --");
        run_one("direct OVC over runend", ITERS, total_rows, || {
            merge_n_way_ovc_runend(&runend_sides) as u64
        });
        {
            let mats: Vec<Vec<u8>> = runend_sides.iter().map(materialize_runend).collect();
            let refs: Vec<&[u8]> = mats.iter().map(Vec::as_slice).collect();
            run_one("materialize + memcmp", ITERS, total_rows, || {
                merge_n_way_memcmp(&refs) as u64
            });
        }
        let dyn_re: Vec<&dyn OrdScan> = runend_sides.iter().map(|s| s as &dyn OrdScan).collect();
        bench_chunks("RUN-END", dyn_re);
        let dyn_re_s: Vec<&dyn OrdScanStructured> =
            runend_sides.iter().map(|s| s as &dyn OrdScanStructured).collect();
        bench_structured("RUN-END", dyn_re_s);

        // VARBIN
        println!("\n-- VARBIN (50B values, leading key) --");
        run_one("direct OVC over varbin (first 8B)", ITERS, total_rows, || {
            merge_n_way_ovc_varbin(&vb_sides) as u64
        });
        let dyn_vb: Vec<&dyn OrdScan> = vb_sides.iter().map(|s| s as &dyn OrdScan).collect();
        bench_chunks("VARBIN", dyn_vb);
    }
}
