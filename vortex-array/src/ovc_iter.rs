// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `OrdIter` — the consolidated transformed-iterator design for OVC.
//!
//! Replaces the parallel-array bookkeeping in `ovc_chunked::merge_n_way_structured`
//! with iters that own their own cursor. The chunk shape is the same enum
//! (Constant / RunEnd / Dense). The merge driver becomes a single function
//! generic over `&mut [Box<dyn OrdIter + '_>]`.
//!
//! Exploratory; see `docs/developer-guide/internals/smj-ovc-design.md`.

#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names
)]

use crate::ovc_encoded::{ConstantI64, DictI64, PrimI64, RunEndI64, VarBin};

/// Caller-owned scratch backing one iter's chunk output.
pub struct Scratch {
    dense: Vec<u64>,
    run_ends: Vec<u32>,
    run_values: Vec<u64>,
}

impl Scratch {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            dense: vec![0u64; chunk_size],
            run_ends: vec![0u32; chunk_size],
            run_values: vec![0u64; chunk_size],
        }
    }
}

/// Structural chunk variant. Borrows from caller-supplied `Scratch`.
pub enum OrdChunk<'a> {
    Constant { value: u64, len: usize },
    RunEnd { run_ends: &'a [u32], values: &'a [u64], len: usize },
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

/// Transformed lending-iterator over ord-byte chunks. Each impl owns its
/// own cursor (no external bookkeeping). Default `next_chunk` produces
/// Dense via the encoding's per-row decode; encodings opt into Constant or
/// RunEnd shapes by overriding.
pub trait OrdIter {
    fn ord_len(&self) -> usize;

    /// Pull the next chunk into `scratch`. Returns `None` when exhausted.
    fn next_chunk<'s>(
        &mut self,
        max_rows: usize,
        scratch: &'s mut Scratch,
    ) -> Option<OrdChunk<'s>>;

    /// Advance `n` rows without producing values (duplicate-bypass shortcut).
    fn skip(&mut self, n: usize);
}

// ───────────────────────────────────────────────────────────────────────────
// Iters per encoding. Each owns its own cursor.
// ───────────────────────────────────────────────────────────────────────────

pub struct PrimIter<'a> {
    data: &'a [i64],
    pos: usize,
}

impl<'a> PrimIter<'a> {
    pub fn new(data: &'a [i64]) -> Self {
        Self { data, pos: 0 }
    }
}

impl OrdIter for PrimIter<'_> {
    fn ord_len(&self) -> usize {
        self.data.len()
    }
    fn next_chunk<'s>(&mut self, max_rows: usize, scratch: &'s mut Scratch) -> Option<OrdChunk<'s>> {
        if self.pos >= self.data.len() {
            return None;
        }
        let take = max_rows.min(self.data.len() - self.pos);
        let src = &self.data[self.pos..self.pos + take];
        for (i, &v) in src.iter().enumerate() {
            scratch.dense[i] = (v as u64) ^ (1u64 << 63);
        }
        self.pos += take;
        Some(OrdChunk::Dense(&scratch.dense[..take]))
    }
    fn skip(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.data.len());
    }
}

pub struct DictIter<'a> {
    codes: &'a [u32],
    pos: usize,
}

impl<'a> DictIter<'a> {
    pub fn new(codes: &'a [u32]) -> Self {
        Self { codes, pos: 0 }
    }
}

impl OrdIter for DictIter<'_> {
    fn ord_len(&self) -> usize {
        self.codes.len()
    }
    fn next_chunk<'s>(&mut self, max_rows: usize, scratch: &'s mut Scratch) -> Option<OrdChunk<'s>> {
        if self.pos >= self.codes.len() {
            return None;
        }
        let take = max_rows.min(self.codes.len() - self.pos);
        let src = &self.codes[self.pos..self.pos + take];
        for (i, &c) in src.iter().enumerate() {
            scratch.dense[i] = u64::from(c) << 32;
        }
        self.pos += take;
        Some(OrdChunk::Dense(&scratch.dense[..take]))
    }
    fn skip(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.codes.len());
    }
}

pub struct ConstantIter {
    value: u64,
    total: usize,
    pos: usize,
}

impl ConstantIter {
    pub fn new(value: i64, len: usize) -> Self {
        Self {
            value: (value as u64) ^ (1u64 << 63),
            total: len,
            pos: 0,
        }
    }
}

impl OrdIter for ConstantIter {
    fn ord_len(&self) -> usize {
        self.total
    }
    fn next_chunk<'s>(
        &mut self,
        max_rows: usize,
        _scratch: &'s mut Scratch,
    ) -> Option<OrdChunk<'s>> {
        if self.pos >= self.total {
            return None;
        }
        let take = max_rows.min(self.total - self.pos);
        self.pos += take;
        Some(OrdChunk::Constant { value: self.value, len: take })
    }
    fn skip(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.total);
    }
}

pub struct RunEndIter<'a> {
    run_ends: &'a [u32],
    values: &'a [i64],
    total: usize,
    pos: usize,
    run_idx: usize,
}

impl<'a> RunEndIter<'a> {
    pub fn new(run_ends: &'a [u32], values: &'a [i64], total: usize) -> Self {
        Self { run_ends, values, total, pos: 0, run_idx: 0 }
    }
}

impl OrdIter for RunEndIter<'_> {
    fn ord_len(&self) -> usize {
        self.total
    }
    fn next_chunk<'s>(&mut self, max_rows: usize, scratch: &'s mut Scratch) -> Option<OrdChunk<'s>> {
        if self.pos >= self.total {
            return None;
        }
        let end = (self.pos + max_rows).min(self.total);
        let take = end - self.pos;
        // Find the run containing self.pos (run_idx is a monotone hint).
        while self.run_idx < self.run_ends.len()
            && (self.run_ends[self.run_idx] as usize) <= self.pos
        {
            self.run_idx += 1;
        }
        let mut local_run = 0;
        let mut row = self.pos;
        let mut run = self.run_idx;
        while row < end {
            let run_end_global = (self.run_ends[run] as usize).min(end);
            scratch.run_ends[local_run] = (run_end_global - self.pos) as u32;
            scratch.run_values[local_run] = (self.values[run] as u64) ^ (1u64 << 63);
            local_run += 1;
            row = run_end_global;
            if run_end_global == self.run_ends[run] as usize {
                run += 1;
            } else {
                break;
            }
        }
        self.pos = end;
        Some(OrdChunk::RunEnd {
            run_ends: &scratch.run_ends[..local_run],
            values: &scratch.run_values[..local_run],
            len: take,
        })
    }
    fn skip(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.total);
    }
}

pub struct VarBinIter<'a> {
    offsets: &'a [u32],
    data: &'a [u8],
    pos: usize,
}

impl<'a> VarBinIter<'a> {
    pub fn new(offsets: &'a [u32], data: &'a [u8]) -> Self {
        Self { offsets, data, pos: 0 }
    }
}

/// Multi-column iter: composes K inner iters whose values pack together
/// into one u64 of OVC head value. For 2 i32 columns we put col0 in the
/// high 32 bits, col1 in the low 32 bits — a single u64 compare resolves
/// the full key. Real multi-col OVC would split offset bits as well and
/// fall back to cmp_full on tie; this is the simplest composition that
/// captures the merge-driver shape.
pub struct MultiColI32Iter<'a> {
    col0: &'a [i32],
    col1: &'a [i32],
    pos: usize,
}

impl<'a> MultiColI32Iter<'a> {
    pub fn new(col0: &'a [i32], col1: &'a [i32]) -> Self {
        assert_eq!(col0.len(), col1.len());
        Self { col0, col1, pos: 0 }
    }
}

impl OrdIter for MultiColI32Iter<'_> {
    fn ord_len(&self) -> usize {
        self.col0.len()
    }
    fn next_chunk<'s>(&mut self, max_rows: usize, scratch: &'s mut Scratch) -> Option<OrdChunk<'s>> {
        if self.pos >= self.col0.len() {
            return None;
        }
        let take = max_rows.min(self.col0.len() - self.pos);
        let c0 = &self.col0[self.pos..self.pos + take];
        let c1 = &self.col1[self.pos..self.pos + take];
        for i in 0..take {
            let u0 = (c0[i] as u32) ^ (1u32 << 31);
            let u1 = (c1[i] as u32) ^ (1u32 << 31);
            scratch.dense[i] = (u64::from(u0) << 32) | u64::from(u1);
        }
        self.pos += take;
        Some(OrdChunk::Dense(&scratch.dense[..take]))
    }
    fn skip(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.col0.len());
    }
}

impl OrdIter for VarBinIter<'_> {
    fn ord_len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    fn next_chunk<'s>(&mut self, max_rows: usize, scratch: &'s mut Scratch) -> Option<OrdChunk<'s>> {
        let total = self.offsets.len().saturating_sub(1);
        if self.pos >= total {
            return None;
        }
        let take = max_rows.min(total - self.pos);
        for i in 0..take {
            let row = self.pos + i;
            let s = self.offsets[row] as usize;
            let e = self.offsets[row + 1] as usize;
            let bytes = &self.data[s..e];
            let mut buf = [0u8; 8];
            let n = bytes.len().min(8);
            buf[..n].copy_from_slice(&bytes[..n]);
            scratch.dense[i] = u64::from_be_bytes(buf);
        }
        self.pos += take;
        Some(OrdChunk::Dense(&scratch.dense[..take]))
    }
    fn skip(&mut self, n: usize) {
        let total = self.offsets.len().saturating_sub(1);
        self.pos = (self.pos + n).min(total);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Convenience constructors from the source structs in ovc_encoded.
// ───────────────────────────────────────────────────────────────────────────

impl<'a> From<&PrimI64<'a>> for PrimIter<'a> {
    fn from(p: &PrimI64<'a>) -> Self {
        Self::new(p.data)
    }
}
impl<'a> From<&DictI64<'a>> for DictIter<'a> {
    fn from(d: &DictI64<'a>) -> Self {
        Self::new(d.codes)
    }
}
impl<'a> From<&RunEndI64<'a>> for RunEndIter<'a> {
    fn from(r: &RunEndI64<'a>) -> Self {
        Self::new(r.run_ends, r.values, r.len)
    }
}
impl From<&ConstantI64> for ConstantIter {
    fn from(c: &ConstantI64) -> Self {
        Self::new(c.value, c.len)
    }
}
impl<'a> From<&VarBin<'a>> for VarBinIter<'a> {
    fn from(v: &VarBin<'a>) -> Self {
        Self::new(v.offsets, v.data)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// n-way merge driver over &mut [Box<dyn OrdIter + '_>] with structured chunks.
//
// State per side captures the current chunk's shape + cursor. When a side's
// chunk is exhausted we pull a new one via the trait method.
// ───────────────────────────────────────────────────────────────────────────

enum SideState {
    Empty,
    Constant { remaining: usize },
    RunEnd { run_idx: usize, n_runs: usize, within: usize, run_len: usize },
    Dense { pos: usize, fill: usize },
}

/// n-way merge driver with two optimisations beyond the basic version:
///   1. `heads: Vec<u64>` keeps the current ord value per side in a flat
///      array, so the min-search is a tight u64 loop — no per-iteration
///      enum match.
///   2. **Bulk emit on the winner** when its current head stays constant
///      for a known span (Constant chunk: entire remaining; RunEnd: rest of
///      the current run). All those rows are emitted in one operation —
///      no per-row min recheck, no per-row state update. Safe because the
///      other sides' heads are unchanged while only the winner advances.
pub fn merge_n_way_iter(iters: &mut [Box<dyn OrdIter + '_>], chunk_size: usize) -> usize {
    let n = iters.len();
    if n == 0 {
        return 0;
    }
    let mut scratches: Vec<Scratch> = (0..n).map(|_| Scratch::new(chunk_size)).collect();
    let mut state: Vec<SideState> = (0..n).map(|_| SideState::Empty).collect();
    let mut heads: Vec<u64> = vec![u64::MAX; n];

    fn refill(
        i: usize,
        iters: &mut [Box<dyn OrdIter + '_>],
        scratches: &mut [Scratch],
        state: &mut [SideState],
        heads: &mut [u64],
        chunk_size: usize,
    ) {
        let chunk = iters[i].next_chunk(chunk_size, &mut scratches[i]);
        match chunk {
            None => {
                state[i] = SideState::Empty;
                heads[i] = u64::MAX;
            }
            Some(OrdChunk::Constant { value, len }) => {
                state[i] = SideState::Constant { remaining: len };
                heads[i] = value;
            }
            Some(OrdChunk::RunEnd { run_ends, values: _, len: _ }) => {
                let n_runs = run_ends.len();
                let first_run_len = if n_runs > 0 { run_ends[0] as usize } else { 0 };
                state[i] = SideState::RunEnd { run_idx: 0, n_runs, within: 0, run_len: first_run_len };
                heads[i] = scratches[i].run_values[0];
            }
            Some(OrdChunk::Dense(buf)) => {
                let fill = buf.len();
                state[i] = SideState::Dense { pos: 0, fill };
                heads[i] = scratches[i].dense[0];
            }
        }
    }

    for i in 0..n {
        refill(i, iters, &mut scratches, &mut state, &mut heads, chunk_size);
    }

    let mut count = 0usize;
    loop {
        // Tight typed min-search across heads.
        let mut min_v = u64::MAX;
        let mut min_side = usize::MAX;
        for i in 0..n {
            if heads[i] < min_v {
                min_v = heads[i];
                min_side = i;
            }
        }
        if min_side == usize::MAX {
            break;
        }

        // Determine how many consecutive rows the winner can emit while
        // its value stays at min_v. For Constant / RunEnd this can be the
        // whole remaining span; for Dense it's 1 (consecutive Dense values
        // may differ).
        let runway = match &state[min_side] {
            SideState::Constant { remaining } => *remaining,
            SideState::RunEnd { within, run_len, .. } => *run_len - *within,
            SideState::Dense { .. } => 1,
            SideState::Empty => unreachable!(),
        };
        count += runway;

        // Advance the winner's state by `runway` rows. May exhaust the
        // chunk, triggering a refill (which updates heads[min_side]).
        let mut needs_refill = false;
        match &mut state[min_side] {
            SideState::Constant { remaining } => {
                *remaining = 0; // we just emitted all of it
                needs_refill = true;
            }
            SideState::RunEnd { run_idx, n_runs, within, run_len } => {
                *within = *run_len; // finished current run
                *run_idx += 1;
                if *run_idx >= *n_runs {
                    needs_refill = true;
                } else {
                    let prev = scratches[min_side].run_ends[*run_idx - 1] as usize;
                    let cur = scratches[min_side].run_ends[*run_idx] as usize;
                    *within = 0;
                    *run_len = cur - prev;
                    heads[min_side] = scratches[min_side].run_values[*run_idx];
                }
            }
            SideState::Dense { pos, fill } => {
                *pos += 1;
                if *pos >= *fill {
                    needs_refill = true;
                } else {
                    heads[min_side] = scratches[min_side].dense[*pos];
                }
            }
            SideState::Empty => unreachable!(),
        }
        if needs_refill {
            refill(min_side, iters, &mut scratches, &mut state, &mut heads, chunk_size);
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::ovc_encoded::{
        DictI64, PrimI64, RunEndI64, VarBin, merge_n_way_ovc_dict, merge_n_way_ovc_prim,
        merge_n_way_ovc_runend, merge_n_way_ovc_varbin,
    };

    use super::*;

    #[test]
    fn agreement_iter_prim() {
        let s0: Vec<i64> = (0..50).collect();
        let s1: Vec<i64> = (50..100).collect();
        let mut iters: Vec<Box<dyn OrdIter + '_>> = vec![
            Box::new(PrimIter::new(&s0)),
            Box::new(PrimIter::new(&s1)),
        ];
        assert_eq!(merge_n_way_iter(&mut iters, 16), 100);
    }

    #[test]
    fn agreement_iter_constant() {
        let mut iters: Vec<Box<dyn OrdIter + '_>> = vec![
            Box::new(ConstantIter::new(1, 10)),
            Box::new(ConstantIter::new(5, 10)),
        ];
        assert_eq!(merge_n_way_iter(&mut iters, 4), 20);
    }

    #[test]
    fn agreement_iter_runend() {
        let e0: Vec<u32> = (1u32..=10).map(|i| i * 5).collect();
        let v0: Vec<i64> = (0..10).map(|i| i * 2).collect();
        let e1: Vec<u32> = (1u32..=5).map(|i| i * 13).collect();
        let v1: Vec<i64> = (0..5).map(|i| i * 2 + 1).collect();
        let mut iters: Vec<Box<dyn OrdIter + '_>> = vec![
            Box::new(RunEndIter::new(&e0, &v0, 50)),
            Box::new(RunEndIter::new(&e1, &v1, 65)),
        ];
        assert_eq!(merge_n_way_iter(&mut iters, 16), 115);
    }

    /// Compare OrdIter (new) against direct OVC, structured chunked (existing),
    /// and materialize+memcmp across encodings.
    ///
    /// Run: cargo test --release -p vortex-array ovc_iter::tests::bench \
    ///     -- --ignored --nocapture --test-threads=1
    #[test]
    #[ignore = "benchmark, run explicitly"]
    #[allow(clippy::cast_precision_loss)]
    fn bench_iter_vs_others() {
        const N: usize = 50_000;
        const N_SIDES: usize = 8;
        const ITERS: u32 = 10;
        const CHUNK: usize = 1024;

        // Build identical data to ovc_chunked bench.
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

        fn run_one(label: &str, iters: u32, total_rows: u64, mut f: impl FnMut() -> u64) {
            let _ = f();
            let t = Instant::now();
            let mut acc = 0u64;
            for _ in 0..iters {
                acc = acc.wrapping_add(std::hint::black_box(f()));
            }
            let ns = t.elapsed().as_nanos() as f64 / (u64::from(iters) * total_rows) as f64;
            println!("    {:<40} {:>8.2} ns/row   acc={acc}", label, ns);
        }

        println!("\n== 8-way merge, {N} rows/side, disjoint, CHUNK={CHUNK} ==");

        // PRIMITIVE
        println!("\n-- PRIMITIVE --");
        run_one("direct OVC over primitive", ITERS, total_rows, || {
            merge_n_way_ovc_prim(&prim_sides) as u64
        });
        run_one("OrdIter (new)", ITERS, total_rows, || {
            let mut iters: Vec<Box<dyn OrdIter + '_>> = prim_sides
                .iter()
                .map(|p| Box::new(PrimIter::new(p.data)) as Box<dyn OrdIter + '_>)
                .collect();
            merge_n_way_iter(&mut iters, CHUNK) as u64
        });

        // DICT
        println!("\n-- DICT (256 distinct) --");
        run_one("direct OVC over dict", ITERS, total_rows, || {
            merge_n_way_ovc_dict(&dict_sides) as u64
        });
        run_one("OrdIter (new)", ITERS, total_rows, || {
            let mut iters: Vec<Box<dyn OrdIter + '_>> = dict_sides
                .iter()
                .map(|d| Box::new(DictIter::new(d.codes)) as Box<dyn OrdIter + '_>)
                .collect();
            merge_n_way_iter(&mut iters, CHUNK) as u64
        });

        // RUN-END
        println!("\n-- RUN-END (100/run) --");
        run_one("direct OVC over runend", ITERS, total_rows, || {
            merge_n_way_ovc_runend(&runend_sides) as u64
        });
        run_one("OrdIter (new)", ITERS, total_rows, || {
            let mut iters: Vec<Box<dyn OrdIter + '_>> = runend_sides
                .iter()
                .map(|r| Box::new(RunEndIter::new(r.run_ends, r.values, r.len)) as Box<dyn OrdIter + '_>)
                .collect();
            merge_n_way_iter(&mut iters, CHUNK) as u64
        });

        // VARBIN
        println!("\n-- VARBIN (50B) --");
        run_one("direct OVC over varbin", ITERS, total_rows, || {
            merge_n_way_ovc_varbin(&vb_sides) as u64
        });
        run_one("OrdIter (new)", ITERS, total_rows, || {
            let mut iters: Vec<Box<dyn OrdIter + '_>> = vb_sides
                .iter()
                .map(|v| Box::new(VarBinIter::new(v.offsets, v.data)) as Box<dyn OrdIter + '_>)
                .collect();
            merge_n_way_iter(&mut iters, CHUNK) as u64
        });
    }

    /// Workload variants beyond fully-disjoint:
    ///   * DENSE: every side has identical data → every emit is a tie
    ///     across all 8 sides. Tests how bulk-emit handles cross-side ties.
    ///   * LESS_DENSE: 50% overlap on the leading column across consecutive
    ///     sides → partial ties, mixed pattern.
    ///   * MULTI-COL: 2-column key packed into one OVC u64.
    #[test]
    #[ignore = "benchmark, run explicitly"]
    #[allow(clippy::cast_precision_loss)]
    fn bench_iter_dense_and_multicol() {
        const N: usize = 50_000;
        const N_SIDES: usize = 8;
        const ITERS: u32 = 10;
        const CHUNK: usize = 1024;

        fn run_one(label: &str, iters: u32, total_rows: u64, mut f: impl FnMut() -> u64) {
            let _ = f();
            let t = Instant::now();
            let mut acc = 0u64;
            for _ in 0..iters {
                acc = acc.wrapping_add(std::hint::black_box(f()));
            }
            let ns = t.elapsed().as_nanos() as f64 / (u64::from(iters) * total_rows) as f64;
            println!("    {:<40} {:>8.2} ns/row   acc={acc}", label, ns);
        }

        let total_rows = (N * N_SIDES) as u64;

        // ===== DENSE: every side has identical data =====
        println!("\n== DENSE (all 8 sides identical content) ==");

        // Primitive — same sorted values on every side
        let prim_dense: Vec<i64> = (0..N as i64).collect();
        println!("\n-- DENSE PRIMITIVE --");
        run_one("OrdIter dense prim", ITERS, total_rows, || {
            let mut iters: Vec<Box<dyn OrdIter + '_>> = (0..N_SIDES)
                .map(|_| Box::new(PrimIter::new(&prim_dense)) as Box<dyn OrdIter + '_>)
                .collect();
            merge_n_way_iter(&mut iters, CHUNK) as u64
        });

        // RunEnd — same runs on every side; bulk-emit should still fire per side
        let runs = 500;
        let run_len = N / runs;
        let re_ends: Vec<u32> = (1..=runs).map(|r| (r * run_len) as u32).collect();
        let re_values: Vec<i64> = (0..runs).map(|r| r as i64 * 13).collect();
        println!("\n-- DENSE RUN-END (100/run, all sides identical) --");
        run_one("OrdIter dense runend", ITERS, total_rows, || {
            let mut iters: Vec<Box<dyn OrdIter + '_>> = (0..N_SIDES)
                .map(|_| {
                    Box::new(RunEndIter::new(&re_ends, &re_values, runs * run_len))
                        as Box<dyn OrdIter + '_>
                })
                .collect();
            merge_n_way_iter(&mut iters, CHUNK) as u64
        });

        // Constant — every row on every side has the same value. Bulk-emit should
        // collapse this to N_SIDES iterations total.
        println!("\n-- DENSE CONSTANT (one value, all sides) --");
        run_one("OrdIter dense constant", ITERS, total_rows, || {
            let mut iters: Vec<Box<dyn OrdIter + '_>> = (0..N_SIDES)
                .map(|_| Box::new(ConstantIter::new(42, N)) as Box<dyn OrdIter + '_>)
                .collect();
            merge_n_way_iter(&mut iters, CHUNK) as u64
        });

        // ===== LESS DENSE: 50% overlap between adjacent sides =====
        println!("\n== LESS-DENSE (50% overlap between adjacent sides) ==");
        let prim_overlap: Vec<Vec<i64>> = (0..N_SIDES)
            .map(|i| {
                let base = (i * N / 2) as i64; // each side shifts by N/2
                (0..N as i64).map(|j| base + j).collect()
            })
            .collect();
        println!("\n-- LESS-DENSE PRIMITIVE --");
        run_one("OrdIter less-dense prim", ITERS, total_rows, || {
            let mut iters: Vec<Box<dyn OrdIter + '_>> = prim_overlap
                .iter()
                .map(|d| Box::new(PrimIter::new(d)) as Box<dyn OrdIter + '_>)
                .collect();
            merge_n_way_iter(&mut iters, CHUNK) as u64
        });

        // ===== MULTI-COLUMN: 2 i32 cols packed into one OVC u64 =====
        // Same data shape as the single-column bench but with two columns.
        // Disjoint on the leading column.
        println!("\n== MULTI-COLUMN (2 i32 cols packed into one OVC u64) ==");
        let mc_col0: Vec<Vec<i32>> = (0..N_SIDES)
            .map(|i| (0..N as i32).map(|j| (i as i32) * (N as i32) + j).collect())
            .collect();
        let mc_col1: Vec<Vec<i32>> = (0..N_SIDES)
            .map(|_| (0..N as i32).map(|j| j % 13).collect())
            .collect();
        println!("\n-- MULTI-COL (2 i32) DISJOINT --");
        run_one("OrdIter multi-col disjoint", ITERS, total_rows, || {
            let mut iters: Vec<Box<dyn OrdIter + '_>> = (0..N_SIDES)
                .map(|i| {
                    Box::new(MultiColI32Iter::new(&mc_col0[i], &mc_col1[i]))
                        as Box<dyn OrdIter + '_>
                })
                .collect();
            merge_n_way_iter(&mut iters, CHUNK) as u64
        });

        // Multi-col DENSE
        let mc_dense_col0: Vec<i32> = (0..N as i32).collect();
        let mc_dense_col1: Vec<i32> = (0..N as i32).map(|j| j % 13).collect();
        println!("\n-- MULTI-COL (2 i32) DENSE --");
        run_one("OrdIter multi-col dense", ITERS, total_rows, || {
            let mut iters: Vec<Box<dyn OrdIter + '_>> = (0..N_SIDES)
                .map(|_| {
                    Box::new(MultiColI32Iter::new(&mc_dense_col0, &mc_dense_col1))
                        as Box<dyn OrdIter + '_>
                })
                .collect();
            merge_n_way_iter(&mut iters, CHUNK) as u64
        });
    }
}
