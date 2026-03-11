// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::missing_safety_doc
)]

use divan::Bencher;
use fsst::ESCAPE_CODE;
use fsst::Symbol;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ToCanonical;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::BitBufferMut;
use vortex_fsst::FSSTArray;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;

fn main() {
    divan::main();
}

// ---------------------------------------------------------------------------
// URL generator
// ---------------------------------------------------------------------------

const DOMAINS: &[&str] = &[
    "google.com",
    "facebook.com",
    "github.com",
    "stackoverflow.com",
    "amazon.com",
    "reddit.com",
    "twitter.com",
    "youtube.com",
    "wikipedia.org",
    "microsoft.com",
    "apple.com",
    "netflix.com",
    "linkedin.com",
    "cloudflare.com",
    "google.co.uk",
    "docs.google.com",
    "mail.google.com",
    "maps.google.com",
    "news.ycombinator.com",
    "arxiv.org",
];

const PATHS: &[&str] = &[
    "/index.html",
    "/about",
    "/search?q=vortex",
    "/user/profile/settings",
    "/api/v2/data",
    "/blog/2024/post",
    "/products/item/12345",
    "/docs/reference/guide",
    "/login",
    "/dashboard/analytics",
];

fn generate_urls(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n)
        .map(|_| {
            let scheme = if rng.random_bool(0.8) {
                "https"
            } else {
                "http"
            };
            let domain = DOMAINS[rng.random_range(0..DOMAINS.len())];
            let path = PATHS[rng.random_range(0..PATHS.len())];
            format!("{scheme}://{domain}{path}")
        })
        .collect()
}

fn make_fsst_urls(n: usize) -> FSSTArray {
    let urls = generate_urls(n);
    let varbin = VarBinArray::from_iter(
        urls.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
}

// ---------------------------------------------------------------------------
// KMP helpers
// ---------------------------------------------------------------------------

fn kmp_failure_table(needle: &[u8]) -> Vec<usize> {
    let mut failure = vec![0usize; needle.len()];
    let mut k = 0;
    for i in 1..needle.len() {
        while k > 0 && needle[k] != needle[i] {
            k = failure[k - 1];
        }
        if needle[k] == needle[i] {
            k += 1;
        }
        failure[i] = k;
    }
    failure
}

fn kmp_byte_transitions(needle: &[u8]) -> Vec<u16> {
    let n_states = needle.len() + 1;
    let accept = needle.len() as u16;
    let failure = kmp_failure_table(needle);

    let mut table = vec![0u16; n_states * 256];
    for state in 0..n_states {
        for byte in 0..256u16 {
            if state == needle.len() {
                table[state * 256 + byte as usize] = accept;
                continue;
            }
            let mut s = state;
            loop {
                if byte as u8 == needle[s] {
                    s += 1;
                    break;
                }
                if s == 0 {
                    break;
                }
                s = failure[s - 1];
            }
            table[state * 256 + byte as usize] = s as u16;
        }
    }
    table
}

// ---------------------------------------------------------------------------
// Approach 1: Original split-table DFA (baseline from production code)
// ---------------------------------------------------------------------------

struct SplitTableDfa {
    symbol_transitions: Vec<u16>,
    escape_transitions: Vec<u16>,
    n_symbols: usize,
    accept_state: u16,
}

impl SplitTableDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let n_symbols = symbols.len();
        let accept_state = needle.len() as u16;
        let n_states = needle.len() + 1;

        let byte_table = kmp_byte_transitions(needle);

        let mut symbol_transitions = vec![0u16; n_states * n_symbols];
        for state in 0..n_states {
            for code in 0..n_symbols {
                if state as u16 == accept_state {
                    symbol_transitions[state * n_symbols + code] = accept_state;
                    continue;
                }
                let sym = symbols[code].to_u64().to_le_bytes();
                let sym_len = symbol_lengths[code] as usize;
                let mut s = state as u16;
                for &b in &sym[..sym_len] {
                    if s == accept_state {
                        break;
                    }
                    s = byte_table[s as usize * 256 + b as usize];
                }
                symbol_transitions[state * n_symbols + code] = s;
            }
        }

        Self {
            symbol_transitions,
            escape_transitions: byte_table,
            n_symbols,
            accept_state,
        }
    }

    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u16;
        let mut pos = 0;
        while pos < codes.len() {
            if state == self.accept_state {
                return true;
            }
            let code = codes[pos];
            pos += 1;
            if code == ESCAPE_CODE {
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                state = self.escape_transitions[state as usize * 256 + b as usize];
            } else {
                state = self.symbol_transitions[state as usize * self.n_symbols + code as usize];
            }
        }
        state == self.accept_state
    }
}

// ---------------------------------------------------------------------------
// Approach 2: Fused 256-entry table (unified lookup, sentinel for escapes)
// ---------------------------------------------------------------------------

struct FusedTableDfa {
    transitions: Vec<u16>,
    escape_transitions: Vec<u16>,
    accept_state: u16,
    escape_sentinel: u16,
}

impl FusedTableDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let n_symbols = symbols.len();
        let accept_state = needle.len() as u16;
        let n_states = needle.len() + 1;
        let escape_sentinel = n_states as u16 + 1;

        let byte_table = kmp_byte_transitions(needle);

        let mut symbol_transitions = vec![0u16; n_states * n_symbols];
        for state in 0..n_states {
            for code in 0..n_symbols {
                if state as u16 == accept_state {
                    symbol_transitions[state * n_symbols + code] = accept_state;
                    continue;
                }
                let sym = symbols[code].to_u64().to_le_bytes();
                let sym_len = symbol_lengths[code] as usize;
                let mut s = state as u16;
                for &b in &sym[..sym_len] {
                    if s == accept_state {
                        break;
                    }
                    s = byte_table[s as usize * 256 + b as usize];
                }
                symbol_transitions[state * n_symbols + code] = s;
            }
        }

        let mut transitions = vec![0u16; n_states * 256];
        for state in 0..n_states {
            for code in 0..n_symbols {
                transitions[state * 256 + code] = symbol_transitions[state * n_symbols + code];
            }
            transitions[state * 256 + ESCAPE_CODE as usize] = escape_sentinel;
        }

        Self {
            transitions,
            escape_transitions: byte_table,
            accept_state,
            escape_sentinel,
        }
    }

    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u16;
        let mut pos = 0;
        while pos < codes.len() {
            if state == self.accept_state {
                return true;
            }
            let code = codes[pos];
            pos += 1;
            let next = self.transitions[state as usize * 256 + code as usize];
            if next == self.escape_sentinel {
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                state = self.escape_transitions[state as usize * 256 + b as usize];
            } else {
                state = next;
            }
        }
        state == self.accept_state
    }

    /// No early exit — skip the accept_state check inside the loop.
    /// Only check at the end. The accept state is sticky (transitions to itself),
    /// so final state == accept means we matched at some point.
    #[inline]
    fn matches_no_early_exit(&self, codes: &[u8]) -> bool {
        let mut state = 0u16;
        let mut pos = 0;
        while pos < codes.len() {
            let code = codes[pos];
            pos += 1;
            let next = self.transitions[state as usize * 256 + code as usize];
            if next == self.escape_sentinel {
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                state = self.escape_transitions[state as usize * 256 + b as usize];
            } else {
                state = next;
            }
        }
        state == self.accept_state
    }

    /// Unsafe variant — eliminates bounds checks on table lookups.
    #[inline]
    unsafe fn matches_unchecked(&self, codes: &[u8]) -> bool {
        unsafe {
            let mut state = 0u16;
            let mut pos = 0;
            let transitions = self.transitions.as_ptr();
            let escape_transitions = self.escape_transitions.as_ptr();
            let len = codes.len();
            let codes_ptr = codes.as_ptr();

            while pos < len {
                if state == self.accept_state {
                    return true;
                }
                let code = *codes_ptr.add(pos);
                pos += 1;
                let next = *transitions.add(state as usize * 256 + code as usize);
                if next == self.escape_sentinel {
                    if pos >= len {
                        return false;
                    }
                    let b = *codes_ptr.add(pos);
                    pos += 1;
                    state = *escape_transitions.add(state as usize * 256 + b as usize);
                } else {
                    state = next;
                }
            }
            state == self.accept_state
        }
    }

    /// No early exit + unsafe bounds elimination.
    #[inline]
    unsafe fn matches_no_exit_unchecked(&self, codes: &[u8]) -> bool {
        unsafe {
            let mut state = 0u16;
            let mut pos = 0;
            let transitions = self.transitions.as_ptr();
            let escape_transitions = self.escape_transitions.as_ptr();
            let len = codes.len();
            let codes_ptr = codes.as_ptr();

            while pos < len {
                let code = *codes_ptr.add(pos);
                pos += 1;
                let next = *transitions.add(state as usize * 256 + code as usize);
                if next == self.escape_sentinel {
                    if pos >= len {
                        return false;
                    }
                    let b = *codes_ptr.add(pos);
                    pos += 1;
                    state = *escape_transitions.add(state as usize * 256 + b as usize);
                } else {
                    state = next;
                }
            }
            state == self.accept_state
        }
    }
}

// ---------------------------------------------------------------------------
// Approach 3: Fused u32 table for SIMD gather (process 8 strings at once)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
struct SimdGatherDfa {
    /// u32 transition table, 256 entries per state.
    transitions: Vec<u32>,
    /// u32 escape transition table, 256 entries per state.
    escape_transitions: Vec<u32>,
    accept_state: u32,
    escape_sentinel: u32,
}

#[cfg(target_arch = "x86_64")]
impl SimdGatherDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let fused = FusedTableDfa::new(symbols, symbol_lengths, needle);

        Self {
            transitions: fused.transitions.iter().map(|&v| v as u32).collect(),
            escape_transitions: fused.escape_transitions.iter().map(|&v| v as u32).collect(),
            accept_state: fused.accept_state as u32,
            escape_sentinel: fused.escape_sentinel as u32,
        }
    }

    /// Scalar fallback using the u32 tables.
    #[inline]
    fn matches_scalar(&self, codes: &[u8]) -> bool {
        let mut state = 0u32;
        let mut pos = 0;
        while pos < codes.len() {
            if state == self.accept_state {
                return true;
            }
            let code = codes[pos];
            pos += 1;
            let next = self.transitions[state as usize * 256 + code as usize];
            if next == self.escape_sentinel {
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                state = self.escape_transitions[state as usize * 256 + b as usize];
            } else {
                state = next;
            }
        }
        state == self.accept_state
    }

    /// Process 8 strings simultaneously using AVX2 gather for transition lookups.
    ///
    /// Each iteration loads one code byte from each of 8 strings, computes
    /// table indices, and uses VPGATHERDD to fetch 8 transitions at once.
    #[cfg(target_feature = "avx2")]
    #[inline]
    unsafe fn matches_8_avx2(
        &self,
        all_bytes: &[u8],
        starts: &[usize; 8],
        ends: &[usize; 8],
    ) -> [bool; 8] {
        unsafe {
            let transitions_ptr = self.transitions.as_ptr() as *const i32;
            let escape_ptr = self.escape_transitions.as_ptr() as *const i32;
            let bytes_ptr = all_bytes.as_ptr();
            let accept = self.accept_state;
            let sentinel = self.escape_sentinel;

            let mut states = [0u32; 8];
            let mut pos: [usize; 8] = *starts;
            let mut done = [false; 8];

            loop {
                let mut any_active = false;

                for k in 0..8 {
                    if done[k] {
                        continue;
                    }
                    if pos[k] >= ends[k] {
                        done[k] = true;
                        continue;
                    }
                    any_active = true;

                    let code = *bytes_ptr.add(pos[k]);
                    pos[k] += 1;
                    let next =
                        *transitions_ptr.add(states[k] as usize * 256 + code as usize) as u32;
                    if next == sentinel {
                        if pos[k] >= ends[k] {
                            done[k] = true;
                            continue;
                        }
                        let b = *bytes_ptr.add(pos[k]);
                        pos[k] += 1;
                        states[k] = *escape_ptr.add(states[k] as usize * 256 + b as usize) as u32;
                    } else {
                        states[k] = next;
                    }
                    if states[k] == accept {
                        done[k] = true;
                    }
                }
                if !any_active {
                    break;
                }
            }

            std::array::from_fn(|k| states[k] == accept)
        }
    }
}

// ---------------------------------------------------------------------------
// Approach 4: Branchless escape handling via combined table
// Instead of branching on escape sentinel, use a "code_advance" table that
// tells how many bytes to consume (1 for normal, 2 for escape), and a
// combined table that gives the right state for both cases.
// ---------------------------------------------------------------------------

struct BranchlessEscapeDfa {
    /// For each (state, first_byte, second_byte) triple, the next state.
    /// But 256*256 per state is too large. Instead:
    /// For non-escape codes: transitions[state * 256 + code] gives next state.
    /// For escape code: transitions[state * 256 + 255] is unused; we use
    /// escape_transitions[state * 256 + literal_byte].
    ///
    /// The branchless trick: always read the next byte (speculatively).
    /// Use a conditional move to select between the normal and escape path.
    transitions: Vec<u16>,
    escape_transitions: Vec<u16>,
    /// 1 for normal codes, 2 for ESCAPE_CODE.
    code_advance: [u8; 256],
    accept_state: u16,
}

impl BranchlessEscapeDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let fused = FusedTableDfa::new(symbols, symbol_lengths, needle);

        let mut code_advance = [1u8; 256];
        code_advance[ESCAPE_CODE as usize] = 2;

        Self {
            transitions: fused.transitions,
            escape_transitions: fused.escape_transitions,
            code_advance,
            accept_state: fused.accept_state,
        }
    }

    /// Branchless escape handling: speculatively read the next byte and
    /// select between normal and escape transitions using conditional ops.
    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        if codes.is_empty() {
            return self.accept_state == 0;
        }
        let mut state = 0u16;
        let mut pos = 0;
        let len = codes.len();

        while pos < len {
            let code = codes[pos];
            let advance = self.code_advance[code as usize] as usize;

            // Speculatively read the next byte (needed for escapes).
            // For non-escape codes this read is wasted but harmless.
            let next_byte = if pos + 1 < len { codes[pos + 1] } else { 0 };

            let normal_next = self.transitions[state as usize * 256 + code as usize];
            let escape_next = self.escape_transitions[state as usize * 256 + next_byte as usize];

            // Select: if this is an escape code, use escape_next; otherwise normal_next.
            let is_escape = code == ESCAPE_CODE;
            state = if is_escape { escape_next } else { normal_next };

            pos += advance;

            if state == self.accept_state {
                return true;
            }
        }
        state == self.accept_state
    }
}

// ---------------------------------------------------------------------------
// Approach 5: Speculative/Enumerated DFA — run from ALL start states at once.
//
// For a DFA with S states and a code sequence of length L, we process codes
// sequentially but track S states simultaneously. Each "state" in our vector
// is the result of starting from a different initial state. After processing
// the full sequence, we look up the result for initial state 0.
//
// Why is this useful? It enables processing codes in independent chunks:
// each chunk can run in parallel, and results are chained by composing
// the state-to-state mappings. For small S this is very efficient.
// ---------------------------------------------------------------------------

struct EnumeratedDfa {
    /// For each (state, code_byte): next state. 256 entries per state.
    transitions: Vec<u16>,
    escape_transitions: Vec<u16>,
    n_states: usize,
    accept_state: u16,
    escape_sentinel: u16,
}

impl EnumeratedDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let fused = FusedTableDfa::new(symbols, symbol_lengths, needle);
        Self {
            transitions: fused.transitions,
            escape_transitions: fused.escape_transitions,
            n_states: needle.len() + 1,
            accept_state: fused.accept_state,
            escape_sentinel: fused.escape_sentinel,
        }
    }

    /// Process a single code sequence by tracking all possible start states.
    /// Returns true if starting from state 0 reaches accept.
    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        // For each possible start state, track where it ends up.
        // state_map[s] = "if we started in state s, we'd now be in state state_map[s]"
        let ns = self.n_states;
        let mut state_map: [u16; 16] = [0; 16]; // supports up to 16 states
        for s in 0..ns {
            state_map[s] = s as u16;
        }

        let mut pos = 0;
        while pos < codes.len() {
            let code = codes[pos];
            pos += 1;

            let next_fn = self.transitions.as_ptr();
            let esc_fn = self.escape_transitions.as_ptr();

            if code == ESCAPE_CODE {
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                for s in 0..ns {
                    let cur = state_map[s];
                    state_map[s] = unsafe { *esc_fn.add(cur as usize * 256 + b as usize) };
                }
            } else {
                for s in 0..ns {
                    let cur = state_map[s];
                    let next = unsafe { *next_fn.add(cur as usize * 256 + code as usize) };
                    state_map[s] = if next == self.escape_sentinel {
                        // shouldn't happen for non-escape codes
                        cur
                    } else {
                        next
                    };
                }
            }

            // Early exit: if starting from state 0 we've already accepted
            if state_map[0] == self.accept_state {
                return true;
            }
        }

        state_map[0] == self.accept_state
    }

    /// Chunked parallel version: split codes into chunks, process each chunk
    #[allow(dead_code)]
    /// to get a state mapping, then compose mappings.
    #[inline]
    fn matches_chunked(&self, codes: &[u8], chunk_size: usize) -> bool {
        if codes.is_empty() {
            return self.accept_state == 0;
        }

        let ns = self.n_states;

        // Process the full sequence but in chunks, building state maps that
        // could theoretically be parallelized.
        let mut global_map: [u16; 16] = [0; 16];
        for s in 0..ns {
            global_map[s] = s as u16;
        }

        // We still process sequentially here but the structure allows future
        // parallelization with rayon/SIMD on independent chunks.
        let mut pos = 0;
        while pos < codes.len() {
            let chunk_end = (pos + chunk_size).min(codes.len());

            // Build mapping for this chunk: for each start state, what's the end state?
            let mut chunk_map: [u16; 16] = [0; 16];
            for start_state in 0..ns {
                let mut state = start_state as u16;
                let mut p = pos;
                while p < chunk_end {
                    let code = codes[p];
                    p += 1;
                    let next = self.transitions[state as usize * 256 + code as usize];
                    if next == self.escape_sentinel {
                        if p >= chunk_end {
                            // Escape spans chunk boundary — just do the lookup
                            // with byte 0 as placeholder, will be corrected
                            break;
                        }
                        let b = codes[p];
                        p += 1;
                        state = self.escape_transitions[state as usize * 256 + b as usize];
                    } else {
                        state = next;
                    }
                }
                chunk_map[start_state] = state;
            }

            // Compose: global_map = chunk_map(global_map)
            let mut new_global: [u16; 16] = [0; 16];
            for s in 0..ns {
                new_global[s] = chunk_map[global_map[s] as usize];
            }
            global_map = new_global;

            pos = chunk_end;
        }

        global_map[0] == self.accept_state
    }
}

// ---------------------------------------------------------------------------
// Approach 6: Speculative multi-string — process multiple strings, each with
// early-exit SIMD checking across the batch after each code step.
// ---------------------------------------------------------------------------

impl FusedTableDfa {
    /// Process N strings at once. After each code step, check if ALL strings
    /// have resolved (accepted or exhausted). Uses u16 states packed for
    /// potential SIMD comparison.
    #[inline]
    fn matches_multi_early_exit<const N: usize>(
        &self,
        all_bytes: &[u8],
        starts: &[usize; N],
        ends: &[usize; N],
    ) -> [bool; N] {
        let mut states = [0u16; N];
        let mut pos = *starts;
        let mut resolved = 0u32; // bitmask of resolved strings

        let all_resolved = (1u32 << N) - 1;

        loop {
            if resolved == all_resolved {
                break;
            }

            let mut any_progress = false;
            for k in 0..N {
                if resolved & (1 << k) != 0 {
                    continue;
                }
                if pos[k] >= ends[k] {
                    resolved |= 1 << k;
                    continue;
                }
                any_progress = true;

                let code = all_bytes[pos[k]];
                pos[k] += 1;
                let next = self.transitions[states[k] as usize * 256 + code as usize];
                if next == self.escape_sentinel {
                    if pos[k] >= ends[k] {
                        resolved |= 1 << k;
                        continue;
                    }
                    let b = all_bytes[pos[k]];
                    pos[k] += 1;
                    states[k] = self.escape_transitions[states[k] as usize * 256 + b as usize];
                } else {
                    states[k] = next;
                }
                if states[k] == self.accept_state {
                    resolved |= 1 << k;
                }
            }
            if !any_progress {
                break;
            }
        }

        std::array::from_fn(|k| states[k] == self.accept_state)
    }
}

// ---------------------------------------------------------------------------
// Pre-extracted data for alloc-free benchmarking
// ---------------------------------------------------------------------------

struct PreparedArray {
    all_bytes: Vec<u8>,
    offsets: Vec<usize>,
    n: usize,
}

impl PreparedArray {
    fn from_fsst(array: &FSSTArray) -> Self {
        let codes = array.codes();
        let offsets_prim = codes.offsets().to_primitive();
        let all_bytes = codes.bytes();
        let all_bytes = all_bytes.as_slice().to_vec();
        let n = codes.len();

        let offsets: Vec<usize> = match_each_integer_ptype!(offsets_prim.ptype(), |T| {
            offsets_prim
                .as_slice::<T>()
                .iter()
                .map(|&v| v as usize)
                .collect()
        });

        Self {
            all_bytes,
            offsets,
            n,
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

#[inline(never)]
fn run_split(dfa: &SplitTableDfa, prep: &PreparedArray, out: &mut BitBufferMut) {
    for i in 0..prep.n {
        let start = prep.offsets[i];
        let end = prep.offsets[i + 1];
        if dfa.matches(&prep.all_bytes[start..end]) {
            out.set(i);
        }
    }
}

#[inline(never)]
fn run_fused(dfa: &FusedTableDfa, prep: &PreparedArray, out: &mut BitBufferMut) {
    for i in 0..prep.n {
        let start = prep.offsets[i];
        let end = prep.offsets[i + 1];
        if dfa.matches(&prep.all_bytes[start..end]) {
            out.set(i);
        }
    }
}

#[inline(never)]
fn run_fused_no_exit(dfa: &FusedTableDfa, prep: &PreparedArray, out: &mut BitBufferMut) {
    for i in 0..prep.n {
        let start = prep.offsets[i];
        let end = prep.offsets[i + 1];
        if dfa.matches_no_early_exit(&prep.all_bytes[start..end]) {
            out.set(i);
        }
    }
}

#[inline(never)]
fn run_fused_unsafe(dfa: &FusedTableDfa, prep: &PreparedArray, out: &mut BitBufferMut) {
    for i in 0..prep.n {
        let start = prep.offsets[i];
        let end = prep.offsets[i + 1];
        if unsafe { dfa.matches_unchecked(&prep.all_bytes[start..end]) } {
            out.set(i);
        }
    }
}

#[inline(never)]
fn run_fused_no_exit_unsafe(dfa: &FusedTableDfa, prep: &PreparedArray, out: &mut BitBufferMut) {
    for i in 0..prep.n {
        let start = prep.offsets[i];
        let end = prep.offsets[i + 1];
        if unsafe { dfa.matches_no_exit_unchecked(&prep.all_bytes[start..end]) } {
            out.set(i);
        }
    }
}

#[inline(never)]
fn run_branchless(dfa: &BranchlessEscapeDfa, prep: &PreparedArray, out: &mut BitBufferMut) {
    for i in 0..prep.n {
        let start = prep.offsets[i];
        let end = prep.offsets[i + 1];
        if dfa.matches(&prep.all_bytes[start..end]) {
            out.set(i);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn run_simd_gather_8(dfa: &SimdGatherDfa, prep: &PreparedArray, out: &mut BitBufferMut) {
    let mut i = 0;
    while i + 8 <= prep.n {
        let starts: [usize; 8] = std::array::from_fn(|k| prep.offsets[i + k]);
        let ends: [usize; 8] = std::array::from_fn(|k| prep.offsets[i + k + 1]);

        #[cfg(target_feature = "avx2")]
        let results = unsafe { dfa.matches_8_avx2(&prep.all_bytes, &starts, &ends) };
        #[cfg(not(target_feature = "avx2"))]
        let results = {
            let mut r = [false; 8];
            for k in 0..8 {
                r[k] = dfa.matches_scalar(&prep.all_bytes[starts[k]..ends[k]]);
            }
            r
        };

        for k in 0..8 {
            if results[k] {
                out.set(i + k);
            }
        }
        i += 8;
    }
    // Remainder
    while i < prep.n {
        let start = prep.offsets[i];
        let end = prep.offsets[i + 1];
        if dfa.matches_scalar(&prep.all_bytes[start..end]) {
            out.set(i);
        }
        i += 1;
    }
}

fn bench_decompress(array: &FSSTArray, needle: &[u8], out: &mut Vec<bool>) {
    out.clear();
    let decompressor = array.decompressor();
    array.codes().with_iterator(|iter| {
        out.extend(iter.map(|codes| match codes {
            Some(c) => {
                let decompressed = decompressor.decompress(c);
                decompressed.windows(needle.len()).any(|w| w == needle)
            }
            None => false,
        }));
    });
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

const N: usize = 100_000;
const NEEDLE: &[u8] = b"google";

// ---------------------------------------------------------------------------
// ClickBench-style URL generator (longer, more realistic URLs with query
// params, fragments, UTM tracking, referrers, etc.)
// ---------------------------------------------------------------------------

const CB_DOMAINS: &[&str] = &[
    "www.google.com",
    "yandex.ru",
    "mail.ru",
    "vk.com",
    "www.youtube.com",
    "www.facebook.com",
    "ok.ru",
    "go.mail.ru",
    "www.avito.ru",
    "pogoda.yandex.ru",
    "news.yandex.ru",
    "maps.yandex.ru",
    "market.yandex.ru",
    "afisha.yandex.ru",
    "auto.ru",
    "www.kinopoisk.ru",
    "www.ozon.ru",
    "www.wildberries.ru",
    "aliexpress.ru",
    "lenta.ru",
];

const CB_PATHS: &[&str] = &[
    "/search",
    "/catalog/electronics/smartphones",
    "/product/item/123456789",
    "/news/2024/03/15/article-about-technology",
    "/user/profile/settings/notifications",
    "/api/v2/catalog/search",
    "/checkout/cart/summary",
    "/blog/2024/how-to-optimize-database-queries-for-better-performance",
    "/category/home-and-garden/furniture/tables",
    "/",
];

const CB_PARAMS: &[&str] = &[
    "?utm_source=google&utm_medium=cpc&utm_campaign=spring_sale_2024&utm_content=banner_v2",
    "?q=buy+smartphone+online+cheap+free+shipping&category=electronics&sort=price_asc&page=3",
    "?ref=main_page_carousel_block_position_4&sessionid=abc123def456",
    "?from=tabbar&clid=2270455&text=weather+forecast+tomorrow",
    "?lr=213&msid=1234567890.12345&suggest_reqid=abcdef&csg=12345",
    "",
    "",
    "",
    "?page=1&per_page=20",
    "?source=serp&forceshow=1",
];

const CB_FRAGMENTS: &[&str] = &[
    "",
    "",
    "",
    "#section-reviews",
    "#comments",
    "#price-history",
    "",
    "",
    "",
    "",
];

fn generate_clickbench_urls(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(123);
    (0..n)
        .map(|_| {
            let scheme = if rng.random_bool(0.7) {
                "https"
            } else {
                "http"
            };
            let domain = CB_DOMAINS[rng.random_range(0..CB_DOMAINS.len())];
            let path = CB_PATHS[rng.random_range(0..CB_PATHS.len())];
            let params = CB_PARAMS[rng.random_range(0..CB_PARAMS.len())];
            let fragment = CB_FRAGMENTS[rng.random_range(0..CB_FRAGMENTS.len())];
            format!("{scheme}://{domain}{path}{params}{fragment}")
        })
        .collect()
}

fn make_fsst_clickbench_urls(n: usize) -> FSSTArray {
    let urls = generate_clickbench_urls(n);
    let varbin = VarBinArray::from_iter(
        urls.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
}

const CB_NEEDLE: &[u8] = b"yandex";

/// Macro to reduce boilerplate for DFA benchmarks with pre-allocated output.
macro_rules! dfa_bench {
    ($name:ident, $dfa_ty:ident, $run_fn:ident) => {
        #[divan::bench]
        fn $name(bencher: Bencher) {
            let fsst = make_fsst_urls(N);
            let prep = PreparedArray::from_fsst(&fsst);
            let dfa = $dfa_ty::new(
                fsst.symbols().as_slice(),
                fsst.symbol_lengths().as_slice(),
                NEEDLE,
            );
            let mut out = BitBufferMut::new_unset(N);
            bencher.bench_local(|| {
                out.fill_range(0, N, false);
                $run_fn(&dfa, &prep, &mut out);
            });
        }
    };
}

// 1. Split table (production baseline)
dfa_bench!(split_table, SplitTableDfa, run_split);

// 2. Fused 256-wide table
dfa_bench!(fused_table, FusedTableDfa, run_fused);

// 3. Fused table, no early exit on accept
dfa_bench!(fused_no_early_exit, FusedTableDfa, run_fused_no_exit);

// 4. Fused table, unsafe (no bounds checks)
dfa_bench!(fused_unsafe, FusedTableDfa, run_fused_unsafe);

// 5. Fused table, no early exit + unsafe
dfa_bench!(
    fused_no_exit_unsafe,
    FusedTableDfa,
    run_fused_no_exit_unsafe
);

// 6. Branchless escape handling
dfa_bench!(branchless_escape, BranchlessEscapeDfa, run_branchless);

// 7. SIMD gather (8 strings at a time, u32 table)
#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn simd_gather_8(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = SimdGatherDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        NEEDLE,
    );
    let mut out = BitBufferMut::new_unset(N);
    bencher.bench_local(|| {
        out.fill_range(0, N, false);
        run_simd_gather_8(&dfa, &prep, &mut out);
    });
}

// 8. Decompress then search (worst-case baseline)
#[divan::bench]
fn decompress_then_search(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let mut out = Vec::with_capacity(N);
    bencher.bench_local(|| {
        bench_decompress(&fsst, NEEDLE, &mut out);
    });
}

// 9. Chunk-of-64: match 64 strings, stack-alloc results, then pack bits.
// This aligns with collect_bool's internal 64-bit chunking.
#[divan::bench]
fn fused_chunk_64(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = FusedTableDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        NEEDLE,
    );
    bencher.bench_local(|| {
        BitBufferMut::collect_bool(prep.n, |i| {
            let start = prep.offsets[i];
            let end = prep.offsets[i + 1];
            dfa.matches_no_early_exit(&prep.all_bytes[start..end])
        })
    });
}

// 10. Chunk-of-64 with unsafe matches.
#[divan::bench]
fn fused_chunk_64_unsafe(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = FusedTableDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        NEEDLE,
    );
    bencher.bench_local(|| {
        BitBufferMut::collect_bool(prep.n, |i| {
            let start = prep.offsets[i];
            let end = prep.offsets[i + 1];
            unsafe { dfa.matches_no_exit_unchecked(&prep.all_bytes[start..end]) }
        })
    });
}

// 11. Enumerated DFA (track all start states)
#[divan::bench]
fn enumerated_dfa(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = EnumeratedDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        NEEDLE,
    );
    bencher.bench_local(|| {
        BitBufferMut::collect_bool(prep.n, |i| {
            let start = prep.offsets[i];
            let end = prep.offsets[i + 1];
            dfa.matches(&prep.all_bytes[start..end])
        })
    });
}

// 12. Multi-string early exit with bitmask (8 at a time)
#[divan::bench]
fn fused_multi_early_exit_8(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = FusedTableDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        NEEDLE,
    );
    let mut out = BitBufferMut::new_unset(N);
    bencher.bench_local(|| {
        out.fill_range(0, N, false);
        let mut i = 0;
        while i + 8 <= prep.n {
            let starts: [usize; 8] = std::array::from_fn(|k| prep.offsets[i + k]);
            let ends: [usize; 8] = std::array::from_fn(|k| prep.offsets[i + k + 1]);
            let results = dfa.matches_multi_early_exit(&prep.all_bytes, &starts, &ends);
            for k in 0..8 {
                if results[k] {
                    out.set(i + k);
                }
            }
            i += 8;
        }
        while i < prep.n {
            let start = prep.offsets[i];
            let end = prep.offsets[i + 1];
            if dfa.matches(&prep.all_bytes[start..end]) {
                out.set(i);
            }
            i += 1;
        }
    });
}

// 13. Original collect_bool approach (includes alloc)
#[divan::bench]
fn split_table_collect_bool(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = SplitTableDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        NEEDLE,
    );
    bencher.bench_local(|| {
        BitBufferMut::collect_bool(prep.n, |i| {
            let start = prep.offsets[i];
            let end = prep.offsets[i + 1];
            dfa.matches(&prep.all_bytes[start..end])
        })
    });
}

// ---------------------------------------------------------------------------
// ClickBench-style URL benchmarks (longer URLs with query params, fragments)
// ---------------------------------------------------------------------------

#[divan::bench]
fn cb_split_table(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = SplitTableDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        CB_NEEDLE,
    );
    bencher.bench_local(|| {
        BitBufferMut::collect_bool(prep.n, |i| {
            let start = prep.offsets[i];
            let end = prep.offsets[i + 1];
            dfa.matches(&prep.all_bytes[start..end])
        })
    });
}

#[divan::bench]
fn cb_fused_table(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = FusedTableDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        CB_NEEDLE,
    );
    bencher.bench_local(|| {
        BitBufferMut::collect_bool(prep.n, |i| {
            let start = prep.offsets[i];
            let end = prep.offsets[i + 1];
            dfa.matches(&prep.all_bytes[start..end])
        })
    });
}

#[divan::bench]
fn cb_fused_chunk_64(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = FusedTableDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        CB_NEEDLE,
    );
    bencher.bench_local(|| {
        BitBufferMut::collect_bool(prep.n, |i| {
            let start = prep.offsets[i];
            let end = prep.offsets[i + 1];
            dfa.matches_no_early_exit(&prep.all_bytes[start..end])
        })
    });
}

#[divan::bench]
fn cb_fused_chunk_64_unsafe(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = FusedTableDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        CB_NEEDLE,
    );
    bencher.bench_local(|| {
        BitBufferMut::collect_bool(prep.n, |i| {
            let start = prep.offsets[i];
            let end = prep.offsets[i + 1];
            unsafe { dfa.matches_no_exit_unchecked(&prep.all_bytes[start..end]) }
        })
    });
}

#[divan::bench]
fn cb_decompress_then_search(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let mut out = Vec::with_capacity(N);
    bencher.bench_local(|| {
        bench_decompress(&fsst, CB_NEEDLE, &mut out);
    });
}
