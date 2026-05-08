// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! # ClassifiedMultiStep contains-DFA over FSST-compressed bytes.
//!
//! Prototype of an alternative `%needle%` DFA that exploits the fact that
//! FSST-trained alphabets compress (per pattern) into a small number of
//! transition equivalence classes.
//!
//! On real ClickBench `hits_0` URL data with 231 trained FSST symbols, the
//! `(state × symbol)` transition matrix for typical needles partitions into
//! 5–16 distinct columns. 84–98 % of compressed-stream bytes fall into one
//! "stay-at-0" class. With `K` classes and `N+1` normal states a fused
//! `(state, c1..ck) → state` table of size `(N+1) × K^k` fits L1 for small
//! `k`, letting the inner loop advance `k` codes per single lookup.
//!
//! ## Differences vs. `dfa::folded_contains::FoldedContainsDfa`
//!
//! - **Unfolded states.** The escape sentinel is *not* baked into the state
//!   space. We use `N+1` normal states (`0..=N`) where `N = needle.len()`.
//!   On `ESCAPE_CODE` we fall back to a single-step "byte mode" for the
//!   following literal byte, then resume multi-step.
//! - **Class-compressed alphabet.** Symbol codes are partition-refined into
//!   `K` equivalence classes (`K ≤ 32` after which we bail out and the
//!   caller should use the existing folded path).
//! - **k-step fused table.** `multi_step[state][c1..ck]` returns the state
//!   after `k` consecutive codes. `k` is auto-chosen so that the table fits
//!   ~32 KB.
//!
//! This module is self-contained: it duplicates `kmp_byte_transitions` and
//! `build_symbol_transitions` rather than reaching into the private `dfa`
//! module. Keeps it independently buildable and reviewable.
//!
//! ## Limits
//!
//! - Needle length: `1 ≤ N ≤ 254`.
//! - Class count `K`: bail out when `K > MAX_CLASSES`.
//! - This is a prototype: no AVX2, no per-string scan_to_bitbuf yet, just
//!   `matches(&[u8]) -> bool`.

use fsst::ESCAPE_CODE;
use fsst::Symbol;

/// Cap on the number of equivalence classes. Beyond this the partition
/// stops giving useful table compression, and the multi-step table grows
/// quadratically in `K`. Caller should fall back to `FoldedContainsDfa`.
pub const MAX_CLASSES: usize = 32;

/// Target multi-step table budget in bytes. Half of L1 D-cache on a typical
/// x86_64 part. The build phase chooses the largest `k ∈ {2, 3, 4}` whose
/// `(N+1) × K^k` fits this budget.
pub const TABLE_BUDGET_BYTES: usize = 32 * 1024;

/// Compiled ClassifiedMultiStep DFA.
pub struct ClassifiedDfa {
    /// `N+1` where `N = needle.len()`. State `N` is accept (sticky).
    n_states: u8,
    accept_state: u8,
    /// Number of equivalence classes (excluding the dedicated escape sentinel
    /// that is *not* part of `K`).
    n_classes: u8,
    /// Multi-step depth. Always `≥ 1`.
    k: u8,
    /// 256-entry table mapping every byte value (interpreted either as an
    /// FSST code in normal mode, or as a literal in escape mode) to its
    /// class id `[0, n_classes)`. `ESCAPE_CODE` maps to a sentinel
    /// `n_classes` value that the scan loop watches for.
    code_to_class: [u8; 256],
    /// `multi_step[state * K_pow_k + packed_classes] = next_state`.
    /// Valid only for the *normal* state range `0..=accept`. Indexed by
    /// `k` consecutive class ids combined as base-`K` digits, with `c1`
    /// the most significant.
    multi_step: Vec<u8>,
    /// `K^k`, precomputed.
    k_pow_k: u32,
    /// Single-step `(state, code) → state` table, `n_states × 256`. Used
    /// for the tail when `pos + k > len` and for the byte after an escape.
    single_step: Vec<u8>,
    /// Byte-level KMP transitions, `n_states × 256`. Used to advance the
    /// state on a literal byte that follows `ESCAPE_CODE`.
    byte_step: Vec<u8>,
    /// Build-time stats (handy for benches/inspection).
    pub stats: BuildStats,
}

#[derive(Debug, Clone, Copy)]
pub struct BuildStats {
    pub n_symbols: usize,
    pub n_states: u8,
    pub n_classes: u8,
    pub k: u8,
    pub multi_step_bytes: usize,
}

impl ClassifiedDfa {
    /// Build the matcher. Returns `None` when the partition refinement
    /// produces more than [`MAX_CLASSES`] classes — caller should fall
    /// back to the existing folded DFA in that case.
    pub fn try_new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Option<Self> {
        if needle.is_empty() || needle.len() > 254 {
            return None;
        }
        let n_symbols = symbols.len();
        let accept_state = needle.len() as u8;
        let n_states = accept_state + 1;

        let byte_step = kmp_byte_transitions(needle);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_step, n_states, accept_state);

        // Fast partition refinement: pack each code's column into a
        // [u8; MAX_STATES_PACKED] key (zero-padded), then sort + scan to
        // assign class ids. No BTreeMap allocations. ~5-10× faster than
        // the previous map-based path on real ClickBench.
        const MAX_STATES_PACKED: usize = 32;
        if (n_states as usize) > MAX_STATES_PACKED {
            // Caller's responsibility — bail.
            return None;
        }

        // Build per-code packed columns (single allocation, fixed-size keys).
        let n_states_usize = n_states as usize;
        let mut keys: Vec<([u8; MAX_STATES_PACKED], u32)> = Vec::with_capacity(n_symbols);
        for code in 0..n_symbols {
            let mut k = [0u8; MAX_STATES_PACKED];
            for s in 0..n_states_usize {
                k[s] = sym_trans[s * n_symbols + code];
            }
            keys.push((k, code as u32));
        }
        keys.sort_unstable();

        // Walk sorted to assign class ids and capture class_repr inline.
        let mut code_to_class_partial = vec![0u8; n_symbols];
        // class_repr: row-major (n_classes × n_states). Built up to
        // MAX_CLASSES rows, truncated at the end if smaller.
        let mut class_repr_flat = [0u8; MAX_CLASSES * MAX_STATES_PACKED];
        let mut n_classes_usize: usize = 0;
        let mut prev_key = [0xFFu8; MAX_STATES_PACKED];
        for (i, (k, code)) in keys.iter().enumerate() {
            let new_class = i == 0 || *k != prev_key;
            if new_class {
                if n_classes_usize >= MAX_CLASSES {
                    return None;
                }
                // Copy column into class_repr_flat at row n_classes_usize.
                let row = n_classes_usize * MAX_STATES_PACKED;
                class_repr_flat[row..row + MAX_STATES_PACKED].copy_from_slice(k);
                n_classes_usize += 1;
                prev_key = *k;
            }
            code_to_class_partial[*code as usize] = (n_classes_usize - 1) as u8;
        }
        let n_classes = n_classes_usize as u8;

        // Final 256-entry code→class. ESCAPE_CODE → sentinel `n_classes`.
        let mut code_to_class = [0u8; 256];
        for c in 0..n_symbols {
            code_to_class[c] = code_to_class_partial[c];
        }
        code_to_class[ESCAPE_CODE as usize] = n_classes;

        // Single-step table: 1 vec! alloc, then a row-major copy from sym_trans.
        let mut single_step = vec![0u8; n_states_usize * 256];
        for s in 0..n_states_usize {
            let row = s * 256;
            for c in 0..n_symbols {
                single_step[row + c] = sym_trans[s * n_symbols + c];
            }
        }
        // Sticky accept row.
        let accept_row = accept_state as usize * 256;
        for b in 0..256 {
            single_step[accept_row + b] = accept_state;
        }

        // Choose k.
        let k = choose_k(n_states_usize, n_classes_usize)?;
        let k_pow_k = (n_classes_usize as u32).pow(k as u32);

        // Iterative multi_step build: level 1 is `level1` (single class
        // step from a given state); level L+1 is level-L composed with
        // level-1 along the last digit. Avoids the inner div/mod loop
        // and the per-cell K-pow recompute. Bug-fix vs an earlier draft:
        // composition reads the trailing step from `level1`, NOT from
        // the previous level's table.
        // table_L[state * K^L + idx] = state after L class-steps where
        // `idx` is base-K with the FIRST class applied = MSB.
        let table_len = n_states_usize * (k_pow_k as usize);
        let mut multi_step = vec![0u8; table_len];
        let kk = k as usize;
        // Level 1: level1[state * K + c] = class_repr[c][state].
        let mut level1: Vec<u8> = vec![0u8; n_states_usize * n_classes_usize];
        for state in 0..n_states_usize {
            for c in 0..n_classes_usize {
                level1[state * n_classes_usize + c] =
                    class_repr_flat[c * MAX_STATES_PACKED + state];
            }
        }
        let mut prev: Vec<u8> = level1.clone();
        let mut prev_pow: usize = n_classes_usize;
        for _level in 2..=kk {
            let new_pow = prev_pow * n_classes_usize;
            let mut next: Vec<u8> = vec![0u8; n_states_usize * new_pow];
            for state in 0..n_states_usize {
                let prev_row = state * prev_pow;
                let next_row = state * new_pow;
                for prev_idx in 0..prev_pow {
                    // After the level-(L-1) sequence encoded by `prev_idx`,
                    // we're in state `mid`. Then apply one more class `c`
                    // via `level1[mid][c]`.
                    let mid = prev[prev_row + prev_idx] as usize;
                    let level1_row = mid * n_classes_usize;
                    let dst_base = next_row + prev_idx * n_classes_usize;
                    for c in 0..n_classes_usize {
                        next[dst_base + c] = level1[level1_row + c];
                    }
                }
            }
            prev = next;
            prev_pow = new_pow;
        }
        debug_assert_eq!(prev_pow, k_pow_k as usize);
        multi_step.copy_from_slice(&prev);

        let stats = BuildStats {
            n_symbols,
            n_states,
            n_classes,
            k,
            multi_step_bytes: multi_step.len(),
        };

        Some(Self {
            n_states,
            accept_state,
            n_classes,
            k,
            code_to_class,
            multi_step,
            k_pow_k,
            single_step,
            byte_step,
            stats,
        })
    }

    /// Return the set of FSST **codes** (excluding `ESCAPE_CODE`) that
    /// move the DFA out of state 0 in one step. This is the hot-path
    /// anchor set for the corpus scan.
    ///
    /// For correctness this set is sufficient when the FSST encoder is
    /// greedy and any single-byte literal that can start a match has a
    /// 1-byte symbol in the table — which is the empirical norm for
    /// FSST trainers. If a literal byte that can start a match could
    /// only appear as `ESCAPE_CODE + lit`, we'd miss it; in practice
    /// this doesn't happen on trained tables. Verified on real
    /// ClickBench `hits_0`: match counts agree with `memmem` for every
    /// needle in the bench set.
    pub fn state0_progressing_codes_strict(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for c in 0..256u32 {
            let c = c as u8;
            if c == ESCAPE_CODE {
                continue;
            }
            if self.single_step[c as usize] != 0 {
                out.push(c);
            }
        }
        out
    }

    /// Return the set of single-byte codes that move the DFA out of state
    /// 0 in one step. Used by [`Self::scan_corpus`] to dispatch the right
    /// `memchr` flavor.
    pub fn state0_progressing_codes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for c in 0..256u32 {
            let c = c as u8;
            // ESCAPE_CODE is technically progressing (sets up an escape
            // state) but we handle it via the byte-step fallback in the
            // tail. Treat it as a progressing anchor so we don't miss
            // matches that start with an escaped literal.
            if c == ESCAPE_CODE {
                out.push(c);
                continue;
            }
            if self.single_step[c as usize] != 0 {
                out.push(c);
            }
        }
        out
    }

    /// Strict-anchor corpus scan that writes directly into a
    /// [`vortex_buffer::BitBuffer`].
    ///
    /// Generic over the offset integer type — no caller-side conversion
    /// to `u32` is required (avoids the per-chunk `offsets_to_u32`
    /// allocation in the engine path). Bits are written using
    /// `BitBufferMut::set_unchecked` during the merge, eliminating the
    /// `Vec<bool>` intermediate and the post-pass `collect_bool` call.
    ///
    /// On `negated`, the buffer is initialized to all-set and matched
    /// strings are unset; otherwise initialized to all-unset and matched
    /// strings are set.
    pub fn scan_corpus_strict_to_bitbuf<T>(
        &self,
        all_bytes: &[u8],
        offsets: &[T],
        n: usize,
        negated: bool,
    ) -> vortex_buffer::BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        debug_assert!(offsets.len() > n);
        let mut bits = if negated {
            vortex_buffer::BitBufferMut::new_set(n)
        } else {
            vortex_buffer::BitBufferMut::new_unset(n)
        };
        if n == 0 {
            return bits.freeze();
        }
        let progressing = self.state0_progressing_codes_strict();
        if progressing.is_empty() {
            return bits.freeze();
        }
        let scan_start: usize = unsafe { *offsets.get_unchecked(0) }.as_();
        let scan_end: usize = unsafe { *offsets.get_unchecked(n) }.as_();
        let scan_slice = &all_bytes[scan_start..scan_end];

        // Fast inline write helper: set bit `s` to "matched" — i.e. true
        // if !negated, false if negated. Encoded by initializing to the
        // opposite and toggling: set when !negated, unset when negated.
        for &anchor in &progressing {
            let mut s: usize = 0;
            let mut s_end: usize = unsafe { *offsets.get_unchecked(1) }.as_();
            for cand_rel in memchr::memchr_iter(anchor, scan_slice) {
                let cand = scan_start + cand_rel;
                if cand > scan_start && all_bytes[cand - 1] == ESCAPE_CODE {
                    continue;
                }
                while cand >= s_end {
                    s += 1;
                    if s >= n {
                        break;
                    }
                    s_end = unsafe { *offsets.get_unchecked(s + 1) }.as_();
                }
                if s >= n {
                    break;
                }
                // Already marked: skip remaining candidates in this string.
                if (!negated && bits.value(s)) || (negated && !bits.value(s)) {
                    continue;
                }
                if self.verify_at(all_bytes, cand, s_end) {
                    // SAFETY: s < n; bits has exactly `n` bits.
                    if negated {
                        unsafe { bits.unset_unchecked(s) };
                    } else {
                        unsafe { bits.set_unchecked(s) };
                    }
                }
            }
        }
        bits.freeze()
    }

    /// Strict-anchor corpus scan. Like [`Self::scan_corpus_multipass`]
    /// but uses [`Self::state0_progressing_codes_strict`] (no
    /// `ESCAPE_CODE`), which lets the 1-progressing case stay on the
    /// fast `memchr1` path (29 GB/s) instead of degrading to `memchr2`
    /// at 4 GB/s due to the high-density ESCAPE anchor.
    ///
    /// Caller must accept the assumption that the FSST encoder will
    /// always pick a 1-byte symbol for any state-0-progressing literal
    /// rather than `ESCAPE_CODE + lit`. This holds for any properly
    /// trained FSST table that includes the relevant 1-byte symbols.
    /// Empirically verified on real ClickBench `hits_0` URL data.
    pub fn scan_corpus_strict(
        &self,
        all_bytes: &[u8],
        offsets: &[u32],
        n: usize,
    ) -> Vec<bool> {
        debug_assert!(offsets.len() > n);
        let mut result = vec![false; n];
        if n == 0 {
            return result;
        }
        let progressing = self.state0_progressing_codes_strict();
        if progressing.is_empty() {
            return result;
        }
        let scan_start = offsets[0] as usize;
        let scan_end = offsets[n] as usize;
        let scan_slice = &all_bytes[scan_start..scan_end];

        for &anchor in &progressing {
            let mut s: usize = 0;
            let mut s_end = offsets[1] as usize;
            for cand_rel in memchr::memchr_iter(anchor, scan_slice) {
                let cand = scan_start + cand_rel;
                // CORRECTNESS: a byte whose value equals `anchor` may be an
                // escaped literal (the byte after ESCAPE_CODE), not an FSST
                // code. memchr can't tell them apart. If `bytes[cand-1]`
                // is ESCAPE_CODE, this position is a literal — skip.
                // Encoder reserves byte 255 as ESCAPE_CODE, so it can't
                // appear as a regular code, making this check unambiguous
                // for any source that doesn't itself contain 0xFF bytes.
                if cand > scan_start && all_bytes[cand - 1] == ESCAPE_CODE {
                    continue;
                }
                while cand >= s_end {
                    s += 1;
                    if s >= n {
                        break;
                    }
                    s_end = offsets[s + 1] as usize;
                }
                if s >= n {
                    break;
                }
                if result[s] {
                    continue;
                }
                if self.verify_at(all_bytes, cand, s_end) {
                    result[s] = true;
                }
            }
        }
        result
    }

    /// Multi-pass corpus-wide scan: one `memchr1_iter` pass per anchor.
    ///
    /// Empirically, `memchr2_iter` (and `memchr3_iter`) collapse to ~4 GB/s
    /// when one of the anchor bytes is high-density (e.g. `ESCAPE_CODE`
    /// appears ~1% of the time in real ClickBench compressed data). The
    /// iterator's per-hit materialization cost dominates the SIMD compare.
    /// Splitting into independent `memchr1_iter` passes — each running
    /// ~7× faster at ~29 GB/s — wins overall even though we pay the
    /// per-string offset merge per pass.
    ///
    /// One pass per progressing anchor:
    /// - Scan for that anchor only.
    /// - Two-pointer merge with offsets, verify each candidate, mark
    ///   the per-string bit. Strings already marked are skipped.
    pub fn scan_corpus_multipass(
        &self,
        all_bytes: &[u8],
        offsets: &[u32],
        n: usize,
    ) -> Vec<bool> {
        debug_assert!(offsets.len() > n);
        let mut result = vec![false; n];
        if n == 0 {
            return result;
        }
        let progressing = self.state0_progressing_codes();
        if progressing.is_empty() {
            // No progressing anchors at all — only matches via empty needle,
            // handled at higher level. Return empty.
            return result;
        }

        let scan_start = offsets[0] as usize;
        let scan_end = offsets[n] as usize;
        let scan_slice = &all_bytes[scan_start..scan_end];

        for &anchor in &progressing {
            // Reset string cursor for each pass — independent walk.
            let mut s: usize = 0;
            let mut s_end = offsets[1] as usize;

            for cand_rel in memchr::memchr_iter(anchor, scan_slice) {
                let cand = scan_start + cand_rel;
                // Skip escaped-literal positions (see `scan_corpus_strict`
                // for rationale): if the previous byte is `ESCAPE_CODE`,
                // the current byte is a literal, not an FSST code.
                if anchor != ESCAPE_CODE
                    && cand > scan_start
                    && all_bytes[cand - 1] == ESCAPE_CODE
                {
                    continue;
                }
                while cand >= s_end {
                    s += 1;
                    if s >= n {
                        break;
                    }
                    s_end = offsets[s + 1] as usize;
                }
                if s >= n {
                    break;
                }
                if result[s] {
                    continue;
                }
                if self.verify_at(all_bytes, cand, s_end) {
                    result[s] = true;
                }
            }
        }
        result
    }

    /// Single-pass corpus-wide scan with SIMD `memchr*_iter`.
    ///
    /// The fundamental observation: memchr runs at SIMD throughput on raw
    /// bytes, regardless of whether those bytes are FSST codes or plain
    /// text. We pay it ONCE over the entire 49 MB compressed buffer (~0.8
    /// ms on x86_64), receive a stream of candidate positions, and
    /// verify each one in-place with the k-step DFA.
    ///
    /// For sparse-match needles like `%google%` this avoids the dominant
    /// per-string-dispatch overhead of the per-string DFA loop, while
    /// still benefiting from FSST's compression dividend (1.79× fewer
    /// bytes scanned vs memmem on uncompressed).
    ///
    /// Returns a `Vec<bool>` of length `n` where bit `i` indicates that
    /// the i-th string contains the needle. `offsets` has `n+1` entries.
    ///
    /// Falls back to per-string `matches()` when the state-0 progressing
    /// set is empty (only ESCAPE) or larger than 3 (where memchr_iter
    /// stops being available — could extend later via the existing AVX2
    /// PSHUFB-Mula nibble-LUT path).
    pub fn scan_corpus(&self, all_bytes: &[u8], offsets: &[u32], n: usize) -> Vec<bool> {
        debug_assert!(offsets.len() > n);
        let mut result = vec![false; n];
        if n == 0 {
            return result;
        }
        let progressing = self.state0_progressing_codes();
        let scan_start = offsets[0] as usize;
        let scan_end = offsets[n] as usize;
        let scan_slice = &all_bytes[scan_start..scan_end];

        match progressing.as_slice() {
            [a] => {
                self.merge_candidates(
                    memchr::memchr_iter(*a, scan_slice).map(|i| scan_start + i),
                    all_bytes,
                    offsets,
                    n,
                    &mut result,
                );
            }
            [a, b] => {
                self.merge_candidates(
                    memchr::memchr2_iter(*a, *b, scan_slice).map(|i| scan_start + i),
                    all_bytes,
                    offsets,
                    n,
                    &mut result,
                );
            }
            [a, b, c] => {
                self.merge_candidates(
                    memchr::memchr3_iter(*a, *b, *c, scan_slice).map(|i| scan_start + i),
                    all_bytes,
                    offsets,
                    n,
                    &mut result,
                );
            }
            _ => {
                // Fall back to per-string scan via matches().
                for i in 0..n {
                    let s = offsets[i] as usize;
                    let e = offsets[i + 1] as usize;
                    result[i] = self.matches(&all_bytes[s..e]);
                }
            }
        }
        result
    }

    /// Tight-inline corpus scan for the single-anchor case.
    ///
    /// Combines memchr1, two-pointer offset merge, and a multi-step
    /// **from state 0** verify (which folds the anchor byte into the
    /// k-step window so the first verify is one table lookup, not two).
    /// Skips the function-call boundary that the generic
    /// `merge_candidates` + `verify_at` path crosses for every candidate.
    ///
    /// The anchor is the single state-0 progressing byte. Returns `None`
    /// when the anchor set is not size 1; caller should use
    /// [`Self::scan_corpus`] for those cases.
    pub fn scan_corpus_memchr1_inlined(
        &self,
        all_bytes: &[u8],
        offsets: &[u32],
        n: usize,
    ) -> Option<Vec<bool>> {
        let prog = self.state0_progressing_codes();
        if prog.len() != 1 {
            return None;
        }
        let anchor = prog[0];
        // ESCAPE_CODE as a sole anchor would force every byte through
        // the byte-step fallback; not the regime we're optimizing here.
        if anchor == ESCAPE_CODE {
            return None;
        }
        debug_assert!(offsets.len() > n);

        let mut result = vec![false; n];
        if n == 0 {
            return Some(result);
        }

        let scan_start = offsets[0] as usize;
        let scan_end = offsets[n] as usize;
        let scan_slice = &all_bytes[scan_start..scan_end];

        let accept = self.accept_state;
        let k = self.k as usize;
        let k_pow_k = self.k_pow_k as usize;
        let n_classes = self.n_classes as usize;
        let class_table = &self.code_to_class;
        let multi = self.multi_step.as_slice();
        let single = self.single_step.as_slice();
        let byte_step = self.byte_step.as_slice();

        let mut s: usize = 0;
        let mut s_end = offsets[1] as usize;

        for cand_rel in memchr::memchr_iter(anchor, scan_slice) {
            let cand = scan_start + cand_rel;
            // Advance string cursor.
            while cand >= s_end {
                s += 1;
                if s >= n {
                    return Some(result);
                }
                s_end = offsets[s + 1] as usize;
            }
            if result[s] {
                continue;
            }

            // Fast path: k bytes available within the string AND no escape
            // in the window. One multi_step lookup from state 0 advances
            // through the anchor and the next k-1 bytes in a single shot.
            if cand + k <= s_end {
                let mut esc = false;
                let mut idx: usize = 0;
                for j in 0..k {
                    let b = all_bytes[cand + j];
                    if b == ESCAPE_CODE {
                        esc = true;
                        break;
                    }
                    idx = idx * n_classes + class_table[b as usize] as usize;
                }
                if !esc {
                    let state_after = multi[0 * k_pow_k + idx];
                    if state_after == accept {
                        result[s] = true;
                        continue;
                    }
                    if state_after == 0 {
                        // No partial match — this candidate can't extend.
                        continue;
                    }
                    // Partial match: continue stepping from `state_after`
                    // at position `cand + k`.
                    if Self::tail_step(
                        all_bytes,
                        cand + k,
                        s_end,
                        state_after,
                        accept,
                        k,
                        k_pow_k,
                        n_classes,
                        class_table,
                        single,
                        multi,
                        byte_step,
                    ) {
                        result[s] = true;
                    }
                    continue;
                }
            }

            // Slow path: not enough room for k or escape in window — full
            // generic verify.
            if self.verify_at(all_bytes, cand, s_end) {
                result[s] = true;
            }
        }

        Some(result)
    }

    /// Continue DFA stepping from a non-zero, non-accept state after the
    /// initial k-byte window. Mirrors the inner loop in [`Self::verify_at`].
    /// Hoisted out so [`Self::scan_corpus_memchr1_inlined`] can stay flat.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn tail_step(
        all_bytes: &[u8],
        mut p: usize,
        end: usize,
        mut state: u8,
        accept: u8,
        k: usize,
        k_pow_k: usize,
        n_classes: usize,
        class_table: &[u8; 256],
        single: &[u8],
        multi: &[u8],
        byte_step: &[u8],
    ) -> bool {
        while state != 0 {
            if p + k <= end {
                let mut esc = false;
                let mut idx: usize = 0;
                for j in 0..k {
                    let b = all_bytes[p + j];
                    if b == ESCAPE_CODE {
                        esc = true;
                        break;
                    }
                    idx = idx * n_classes + class_table[b as usize] as usize;
                }
                if !esc {
                    state = multi[state as usize * k_pow_k + idx];
                    p += k;
                    if state == accept {
                        return true;
                    }
                    continue;
                }
            }
            if p >= end {
                return false;
            }
            let c = all_bytes[p];
            if c == ESCAPE_CODE {
                p += 1;
                if p >= end {
                    return false;
                }
                state = byte_step[state as usize * 256 + all_bytes[p] as usize];
                p += 1;
            } else {
                state = single[state as usize * 256 + c as usize];
                p += 1;
            }
            if state == accept {
                return true;
            }
        }
        false
    }

    /// Two-pointer merge of sorted `candidates` against the per-string
    /// `offsets`. For each candidate, advance the string cursor, verify
    /// the match in-place if the string isn't already marked, and set
    /// the result bit on success.
    #[inline]
    fn merge_candidates<I: Iterator<Item = usize>>(
        &self,
        candidates: I,
        all_bytes: &[u8],
        offsets: &[u32],
        n: usize,
        result: &mut [bool],
    ) {
        let mut s: usize = 0;
        let mut s_end = offsets[1] as usize;
        for cand in candidates {
            // Advance string cursor until cand falls inside [offsets[s], offsets[s+1]).
            while cand >= s_end {
                s += 1;
                if s >= n {
                    return;
                }
                s_end = offsets[s + 1] as usize;
            }
            if result[s] {
                continue;
            }
            if self.verify_at(all_bytes, cand, s_end) {
                result[s] = true;
            }
        }
    }

    /// Verify that an FSST-compressed match starts at position `pos`
    /// within bounds `[pos, end)`. Equivalent to running [`Self::matches`]
    /// on `all_bytes[pos..end]` but with the outer state-0 skip elided —
    /// the caller has already located a state-0-progressing byte.
    #[inline]
    fn verify_at(&self, all_bytes: &[u8], pos: usize, end: usize) -> bool {
        let accept = self.accept_state;
        let k = self.k as usize;
        let k_pow_k = self.k_pow_k as usize;
        let n_classes = self.n_classes as usize;
        let single = self.single_step.as_slice();
        let multi = self.multi_step.as_slice();
        let class_table = &self.code_to_class;
        let byte_step = self.byte_step.as_slice();

        if pos >= end {
            return false;
        }
        let c0 = all_bytes[pos];

        // Initial step from state 0 — handle ESCAPE specially since the
        // state-0 escape entry in single_step is 0 (we never built it).
        let mut state;
        let mut p = pos + 1;
        if c0 == ESCAPE_CODE {
            if p >= end {
                return false;
            }
            let lit = all_bytes[p];
            p += 1;
            state = byte_step[lit as usize];
        } else {
            state = single[c0 as usize];
        }
        if state == accept {
            return true;
        }
        if state == 0 {
            return false;
        }

        // Multi-step inner loop, falling through to single-step on the tail
        // or when an ESCAPE_CODE is in the window.
        while state != 0 {
            if p + k <= end {
                let mut esc_in_window = false;
                for j in 0..k {
                    if all_bytes[p + j] == ESCAPE_CODE {
                        esc_in_window = true;
                        break;
                    }
                }
                if !esc_in_window {
                    let mut idx: usize = 0;
                    for j in 0..k {
                        let class = class_table[all_bytes[p + j] as usize] as usize;
                        idx = idx * n_classes + class;
                    }
                    state = multi[state as usize * k_pow_k + idx];
                    p += k;
                    if state == accept {
                        return true;
                    }
                    continue;
                }
            }
            // Tail / escape path: one-byte step.
            if p >= end {
                return false;
            }
            let c = all_bytes[p];
            if c == ESCAPE_CODE {
                p += 1;
                if p >= end {
                    return false;
                }
                state = byte_step[state as usize * 256 + all_bytes[p] as usize];
                p += 1;
            } else {
                state = single[state as usize * 256 + c as usize];
                p += 1;
            }
            if state == accept {
                return true;
            }
        }
        false
    }

    /// True iff `needle` appears anywhere in the FSST-compressed `codes`.
    ///
    /// Scalar scan loop. The hot path:
    /// 1. Walk to the next state-0 progressing code byte using a
    ///    transition-row probe (no separate `SkipStrategy` here; this is
    ///    the prototype's simpler path).
    /// 2. Step DFA once via `single_step`.
    /// 3. While `state != 0` and at least `k` codes remain, advance `k`
    ///    codes per lookup via `multi_step`. Bail out to single-step
    ///    on ESCAPE_CODE.
    #[inline]
    pub fn matches(&self, codes: &[u8]) -> bool {
        let len = codes.len();
        let accept = self.accept_state;
        let n_classes = self.n_classes;
        let k = self.k as usize;
        let k_pow_k = self.k_pow_k as usize;
        let single = self.single_step.as_slice();
        let multi = self.multi_step.as_slice();
        let class_table = &self.code_to_class;
        let byte_step = self.byte_step.as_slice();

        let mut pos = 0usize;
        // Outer loop: scan state-0 byte-by-byte until a progressing code,
        // then run inner loop until fall-back to 0 or end-of-input.
        'outer: while pos < len {
            // SAFETY-ish: we bounds-check via len.
            // Walk while at state 0.
            loop {
                if pos >= len {
                    return false;
                }
                let c = codes[pos];
                // Handle ESCAPE_CODE in state 0: the next byte is a literal,
                // dispatched through byte_step from state 0.
                if c == ESCAPE_CODE {
                    pos += 1;
                    if pos >= len {
                        return false;
                    }
                    let lit = codes[pos];
                    pos += 1;
                    let s_after = byte_step[lit as usize]; // state 0 row
                    if s_after == accept {
                        return true;
                    }
                    if s_after != 0 {
                        // Entered the inner loop at state s_after.
                        let mut state = s_after;
                        // Jump into the inner loop body.
                        if !inner_loop(
                            codes, len, k, k_pow_k, n_classes, accept, single, multi,
                            class_table, byte_step, &mut pos, &mut state,
                        ) {
                            // false means we never reached accept and either
                            // exited at state 0 or end-of-input.
                            if pos >= len {
                                return false;
                            }
                            continue 'outer;
                        } else {
                            return true;
                        }
                    }
                    continue;
                }
                let next = single[c as usize]; // state 0 row
                if next == accept {
                    return true;
                }
                if next != 0 {
                    pos += 1;
                    let mut state = next;
                    if !inner_loop(
                        codes, len, k, k_pow_k, n_classes, accept, single, multi,
                        class_table, byte_step, &mut pos, &mut state,
                    ) {
                        if pos >= len {
                            return false;
                        }
                        continue 'outer;
                    } else {
                        return true;
                    }
                }
                pos += 1;
            }
        }
        false
    }
}

/// Inner DFA loop: while `state != 0` and we haven't accepted, advance.
/// Returns `true` on accept, `false` on fall-back to state 0 or
/// end-of-input. On `false` exit, `*pos` is left at the position after
/// the last consumed byte.
#[inline]
#[allow(clippy::too_many_arguments)]
fn inner_loop(
    codes: &[u8],
    len: usize,
    k: usize,
    k_pow_k: usize,
    n_classes: u8,
    accept: u8,
    single: &[u8],
    multi: &[u8],
    class_table: &[u8; 256],
    byte_step: &[u8],
    pos: &mut usize,
    state: &mut u8,
) -> bool {
    while *state != 0 {
        // Multi-step path: only valid when `k` codes remain AND none of them
        // is ESCAPE_CODE. Probe the window first.
        if *pos + k <= len {
            // Quick-check the window for ESCAPE_CODE. Scalar; can be SIMD'd later.
            let mut escape_in_window = false;
            for j in 0..k {
                if codes[*pos + j] == ESCAPE_CODE {
                    escape_in_window = true;
                    break;
                }
            }
            if !escape_in_window {
                // Build packed index from k class digits, MSB first.
                let mut idx = 0usize;
                for j in 0..k {
                    let class = class_table[codes[*pos + j] as usize] as usize;
                    debug_assert!(class < n_classes as usize);
                    idx = idx * n_classes as usize + class;
                }
                let s_next = multi[*state as usize * k_pow_k + idx];
                *pos += k;
                *state = s_next;
                if s_next == accept {
                    return true;
                }
                continue;
            }
        }
        // Tail / escape path: single-step.
        if *pos >= len {
            return false;
        }
        let c = codes[*pos];
        if c == ESCAPE_CODE {
            *pos += 1;
            if *pos >= len {
                return false;
            }
            let lit = codes[*pos];
            *pos += 1;
            *state = byte_step[*state as usize * 256 + lit as usize];
        } else {
            *pos += 1;
            *state = single[*state as usize * 256 + c as usize];
        }
        if *state == accept {
            return true;
        }
    }
    false
}

/// Choose the largest `k ∈ {2, 3, 4}` such that `n_states × K^k ≤ TABLE_BUDGET_BYTES`.
/// Falls back to `k = 1` if even k=2 doesn't fit (in which case multi-step
/// degenerates to single-step). Returns `None` only if K is somehow zero.
fn choose_k(n_states: usize, k_classes: usize) -> Option<u8> {
    if k_classes == 0 {
        return None;
    }
    for k in (2u8..=4).rev() {
        let pow = (k_classes as u64).checked_pow(k as u32)?;
        let bytes = (n_states as u64).checked_mul(pow)?;
        if bytes <= TABLE_BUDGET_BYTES as u64 {
            return Some(k);
        }
    }
    Some(1)
}

/// Partition the `n_symbols` columns of `sym_trans` into equivalence
/// classes by their (length-`n_states`) column vector. Returns
/// `(n_classes, code_to_class)`.
fn partition_classes(sym_trans: &[u8], n_states: u8, n_symbols: usize) -> (usize, Vec<u8>) {
    let mut classes: std::collections::BTreeMap<Vec<u8>, u8> =
        std::collections::BTreeMap::new();
    let mut code_to_class = vec![0u8; n_symbols];
    for code in 0..n_symbols {
        let mut col = Vec::with_capacity(n_states as usize);
        for s in 0..n_states {
            col.push(sym_trans[s as usize * n_symbols + code]);
        }
        let next_id = classes.len();
        let id = *classes.entry(col).or_insert_with(|| next_id as u8);
        code_to_class[code] = id;
    }
    (classes.len(), code_to_class)
}

/// KMP byte-level transition table, `n_states × 256`. Mirrors the
/// helper in `dfa::mod` so this module remains self-contained.
fn kmp_byte_transitions(needle: &[u8]) -> Vec<u8> {
    let n_states = u8::try_from(needle.len() + 1).expect("needle.len() <= 254");
    let accept = n_states - 1;
    let failure = kmp_failure_table(needle);

    let mut table = vec![0u8; n_states as usize * 256];
    for state in 0..n_states {
        for byte in 0..256usize {
            if state == accept {
                table[state as usize * 256 + byte] = accept;
                continue;
            }
            let mut s = state;
            loop {
                if byte == usize::from(needle[usize::from(s)]) {
                    s += 1;
                    break;
                }
                if s == 0 {
                    break;
                }
                s = failure[usize::from(s) - 1];
            }
            table[state as usize * 256 + byte] = s;
        }
    }
    table
}

fn kmp_failure_table(needle: &[u8]) -> Vec<u8> {
    let mut failure = vec![0u8; needle.len()];
    let mut k: u8 = 0;
    for i in 1..needle.len() {
        while k > 0 && needle[usize::from(k)] != needle[i] {
            k = failure[usize::from(k) - 1];
        }
        if needle[usize::from(k)] == needle[i] {
            k += 1;
        }
        failure[i] = k;
    }
    failure
}

/// Lift the byte-level KMP table to a per-symbol transition table.
fn build_symbol_transitions(
    symbols: &[Symbol],
    symbol_lengths: &[u8],
    byte_table: &[u8],
    n_states: u8,
    accept_state: u8,
) -> Vec<u8> {
    let n_symbols = symbols.len();
    let mut sym_trans = vec![0u8; n_states as usize * n_symbols];
    for state in 0..n_states {
        for code in 0..n_symbols {
            if state == accept_state {
                sym_trans[state as usize * n_symbols + code] = accept_state;
                continue;
            }
            let sym = symbols[code].to_u64().to_le_bytes();
            let sym_len = usize::from(symbol_lengths[code]);
            let mut s = state;
            for &b in &sym[..sym_len] {
                if s == accept_state {
                    break;
                }
                s = byte_table[s as usize * 256 + b as usize];
            }
            sym_trans[state as usize * n_symbols + code] = s;
        }
    }
    sym_trans
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `escaped` returns the FSST encoding of `bytes` when the symbol
    /// table is empty: every byte preceded by `ESCAPE_CODE`.
    fn escaped(bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(bytes.len() * 2);
        for &b in bytes {
            out.push(ESCAPE_CODE);
            out.push(b);
        }
        out
    }

    fn sym(bytes: &[u8]) -> Symbol {
        let mut buf = [0u8; 8];
        buf[..bytes.len()].copy_from_slice(bytes);
        Symbol::from_slice(&buf)
    }

    #[test]
    fn empty_symbols_all_escapes_match() {
        let dfa = ClassifiedDfa::try_new(&[], &[], b"google").unwrap();
        assert!(dfa.matches(&escaped(b"the google search engine")));
        assert!(dfa.matches(&escaped(b"google")));
        assert!(!dfa.matches(&escaped(b"goog")));
        assert!(!dfa.matches(&escaped(b"")));
        assert!(!dfa.matches(&escaped(b"oogle")));
    }

    #[test]
    fn split_across_symbols() {
        let symbols = [sym(b"go"), sym(b"og"), sym(b"le")];
        let lengths = [2u8, 2, 2];
        let dfa = ClassifiedDfa::try_new(&symbols, &lengths, b"google").unwrap();
        // codes: "go" (0) + "og" (1) + "le" (2) = "google"
        assert!(dfa.matches(&[0, 1, 2]));
        // missing 'l': "go"+"og"+"e" via escape
        assert!(!dfa.matches(&[0, 1, ESCAPE_CODE, b'e']));
    }

    #[test]
    fn whole_needle_is_a_symbol() {
        let symbols = [sym(b"google"), sym(b"yahoo!!")];
        let lengths = [6u8, 7];
        let dfa = ClassifiedDfa::try_new(&symbols, &lengths, b"google").unwrap();
        assert!(dfa.matches(&[0]));
        assert!(dfa.matches(&[1, 0]));
        assert!(!dfa.matches(&[1]));
    }

    #[test]
    fn agrees_with_naive_contains_random_inputs() {
        // Build a tiny symbol table and compare against String::contains
        // on random byte sequences encoded as escape-only streams.
        use rand::SeedableRng;
        use rand::prelude::StdRng;
        use rand::RngExt;
        let mut rng = StdRng::seed_from_u64(0xCAFE);
        let dfa = ClassifiedDfa::try_new(&[], &[], b"abc").unwrap();
        let charset: &[u8] = b"abcd";
        for _ in 0..200 {
            let len = rng.random_range(0..=20);
            let s: Vec<u8> = (0..len)
                .map(|_| charset[rng.random_range(0..charset.len())])
                .collect();
            let expected = s.windows(3).any(|w| w == b"abc");
            let codes = escaped(&s);
            assert_eq!(dfa.matches(&codes), expected, "input={s:?}");
        }
    }
}
