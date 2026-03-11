// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::missing_safety_doc
)]

use aho_corasick::AhoCorasick;
use daachorse::DoubleArrayAhoCorasick;
use divan::Bencher;
use fsst::ESCAPE_CODE;
use fsst::Symbol;
use memchr::memmem;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use regex_automata::dfa::regex::Regex as DfaRegex;
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
// Approach 5: u8 state table — halve table size (u16→u8) since states fit in
// a byte. Smaller tables = better cache utilization.
// ---------------------------------------------------------------------------

struct CompactDfa {
    /// u8 transitions, 256 entries per state.
    transitions: Vec<u8>,
    escape_transitions: Vec<u8>,
    accept_state: u8,
    escape_sentinel: u8,
}

impl CompactDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let fused = FusedTableDfa::new(symbols, symbol_lengths, needle);
        Self {
            transitions: fused.transitions.iter().map(|&v| v as u8).collect(),
            escape_transitions: fused.escape_transitions.iter().map(|&v| v as u8).collect(),
            accept_state: fused.accept_state as u8,
            escape_sentinel: fused.escape_sentinel as u8,
        }
    }

    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
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

    #[inline]
    fn matches_no_early_exit(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
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

    /// Unsafe no-exit variant.
    #[inline]
    unsafe fn matches_no_exit_unchecked(&self, codes: &[u8]) -> bool {
        unsafe {
            let mut state = 0u8;
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
// Approach 6: Streaming scan — process the ENTIRE codes buffer in one pass,
// resetting state at string boundaries. Avoids per-string slice overhead
// and is friendlier to the hardware prefetcher.
// ---------------------------------------------------------------------------

#[inline(never)]
#[allow(dead_code)]
fn streaming_scan_fused(
    dfa: &FusedTableDfa,
    all_bytes: &[u8],
    offsets: &[usize],
    n: usize,
) -> BitBufferMut {
    BitBufferMut::collect_bool(n, |i| {
        // The collect_bool closure is called sequentially for i=0..n.
        // We rely on the sequential access pattern being prefetch-friendly.
        let start = offsets[i];
        let end = offsets[i + 1];
        dfa.matches(&all_bytes[start..end])
    })
}

/// True streaming: single pass through all_bytes with offset-based reset.
#[inline(never)]
fn streaming_scan_continuous(
    dfa: &CompactDfa,
    all_bytes: &[u8],
    offsets: &[usize],
    n: usize,
    out: &mut BitBufferMut,
) {
    let mut string_idx = 0;
    let mut state = 0u8;
    let mut next_boundary = offsets[1];
    let mut matched = false;

    let mut pos = offsets[0];
    let total_end = offsets[n];

    while pos < total_end {
        // Check if we've crossed into a new string.
        while pos >= next_boundary {
            // Record result for the just-finished string.
            if matched || state == dfa.accept_state {
                out.set(string_idx);
            }
            string_idx += 1;
            if string_idx >= n {
                return;
            }
            state = 0;
            matched = false;
            next_boundary = offsets[string_idx + 1];
        }

        let code = all_bytes[pos];
        pos += 1;
        let next = dfa.transitions[state as usize * 256 + code as usize];
        if next == dfa.escape_sentinel {
            if pos < next_boundary {
                let b = all_bytes[pos];
                pos += 1;
                state = dfa.escape_transitions[state as usize * 256 + b as usize];
            }
        } else {
            state = next;
        }
        if state == dfa.accept_state {
            matched = true;
        }
    }

    // Handle the last string.
    if string_idx < n && (matched || state == dfa.accept_state) {
        out.set(string_idx);
    }
}

// ---------------------------------------------------------------------------
// Approach 7: Prefilter — build a bitmask of codes that could possibly
// contribute to matching the needle. Skip DFA for strings where no code
// belongs to that set.
// ---------------------------------------------------------------------------

struct PrefilterDfa {
    inner: CompactDfa,
    /// For each code byte (0..255), true if that code could produce any byte
    /// present in the needle (i.e., the symbol's bytes intersect needle's bytes).
    relevant_codes: [bool; 256],
}

impl PrefilterDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let inner = CompactDfa::new(symbols, symbol_lengths, needle);

        // Build set of bytes that appear in the needle.
        let mut needle_bytes = [false; 256];
        for &b in needle {
            needle_bytes[b as usize] = true;
        }

        // For each symbol code, check if any of its bytes appear in the needle.
        let mut relevant_codes = [false; 256];
        for (code, (sym, &sym_len)) in symbols.iter().zip(symbol_lengths.iter()).enumerate() {
            let sym_bytes = sym.to_u64().to_le_bytes();
            for &b in &sym_bytes[..sym_len as usize] {
                if needle_bytes[b as usize] {
                    relevant_codes[code] = true;
                    break;
                }
            }
        }
        // Escape code is always relevant (literal bytes could be anything).
        relevant_codes[ESCAPE_CODE as usize] = true;

        Self {
            inner,
            relevant_codes,
        }
    }

    /// Quick check: does this code sequence contain any code that could
    /// contribute to the needle match?
    #[inline]
    fn could_match(&self, codes: &[u8]) -> bool {
        codes.iter().any(|&c| self.relevant_codes[c as usize])
    }

    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        if !self.could_match(codes) {
            return false;
        }
        self.inner.matches(codes)
    }

    #[inline]
    fn matches_no_early_exit(&self, codes: &[u8]) -> bool {
        if !self.could_match(codes) {
            return false;
        }
        self.inner.matches_no_early_exit(codes)
    }
}

// ---------------------------------------------------------------------------
// Approach 8: State-zero skip DFA — skip runs of codes that keep state=0.
//
// Precompute a 256-byte lookup: for each code byte, does transitioning from
// state 0 stay in state 0? If so, that code is "trivial" and can be skipped.
// Process codes in chunks: scan for the first non-trivial code, then run
// the scalar DFA from there. This is most effective when the needle is rare
// (most codes are trivial), which is the common case for selective predicates.
// ---------------------------------------------------------------------------

struct StateZeroSkipDfa {
    inner: CompactDfa,
    /// For each code byte (0..255), true if it keeps state 0 → state 0.
    trivial: [bool; 256],
}

impl StateZeroSkipDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let inner = CompactDfa::new(symbols, symbol_lengths, needle);

        let mut trivial = [false; 256];
        for code in 0..256 {
            // A code is trivial if from state 0 it goes back to state 0
            // and it's not the escape sentinel.
            let next = inner.transitions[code]; // state 0 * 256 + code
            trivial[code] = next == 0 && code as u8 != ESCAPE_CODE;
        }

        Self { inner, trivial }
    }

    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        // Skip leading trivial codes.
        let mut start = 0;
        while start < codes.len() && self.trivial[codes[start] as usize] {
            start += 1;
        }
        if start == codes.len() {
            return self.inner.accept_state == 0;
        }
        // Run the DFA from the first non-trivial code.
        self.inner.matches_no_early_exit(&codes[start..])
    }
}

// ---------------------------------------------------------------------------
// Approach 9: Shift-based DFA — pack all state transitions into a u64.
//
// For a DFA with S ≤ 21 states (3 bits each fit in 63 bits of a u64),
// we store the transitions for ALL states for a given input byte in one u64.
// Transition: next_state = (table[code_byte] >> (state * BITS)) & MASK
//
// The key advantage: the table load depends only on code_byte (known from
// the input stream), NOT on the current state. This breaks the load-use
// dependency chain that makes traditional table-lookup DFAs slow (~4 cycle
// L1 latency per transition). With the shift-based approach, the table
// value can be loaded while the previous transition's shift is executing.
// ---------------------------------------------------------------------------

struct ShiftDfa {
    /// For each code byte (0..255): a u64 packing all state transitions.
    /// Bits [state*3 .. state*3+3) encode the next state for that input.
    transitions: [u64; 256],
    /// Same layout for escape byte transitions.
    escape_transitions: [u64; 256],
    accept_state: u8,
    escape_sentinel: u8,
}

impl ShiftDfa {
    const BITS: u32 = 4; // bits per state (supports up to 16 states = 2^4)
    const MASK: u64 = (1 << Self::BITS) - 1;

    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        assert!(
            needle.len() + 2 <= (1 << Self::BITS),
            "needle too long for 4-bit states (max 14 chars)"
        );

        let fused = FusedTableDfa::new(symbols, symbol_lengths, needle);

        // Pack the fused u16 transitions into u64 shift tables.
        let n_states = needle.len() + 1;
        let escape_sentinel_u8 = fused.escape_sentinel as u8;

        let mut transitions = [0u64; 256];
        let mut escape_transitions = [0u64; 256];

        for code_byte in 0..256usize {
            let mut packed = 0u64;
            for state in 0..n_states {
                let next = fused.transitions[state * 256 + code_byte];
                // Map the escape sentinel to a value that fits in 3 bits.
                let next_u8 = if next == fused.escape_sentinel {
                    escape_sentinel_u8
                } else {
                    next as u8
                };
                packed |= (next_u8 as u64) << (state as u32 * Self::BITS);
            }
            transitions[code_byte] = packed;
        }

        for byte_val in 0..256usize {
            let mut packed = 0u64;
            for state in 0..n_states {
                let next = fused.escape_transitions[state * 256 + byte_val] as u8;
                packed |= (next as u64) << (state as u32 * Self::BITS);
            }
            escape_transitions[byte_val] = packed;
        }

        Self {
            transitions,
            escape_transitions,
            accept_state: fused.accept_state as u8,
            escape_sentinel: escape_sentinel_u8,
        }
    }

    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        let mut pos = 0;
        while pos < codes.len() {
            if state == self.accept_state {
                return true;
            }
            let code = codes[pos];
            pos += 1;
            // The table load depends only on `code`, not on `state`.
            // The shift depends on `state` but is a fast register op.
            let packed = self.transitions[code as usize];
            let next = ((packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            if next == self.escape_sentinel {
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                let esc_packed = self.escape_transitions[b as usize];
                state = ((esc_packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            } else {
                state = next;
            }
        }
        state == self.accept_state
    }

    #[inline]
    fn matches_no_early_exit(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        let mut pos = 0;
        while pos < codes.len() {
            let code = codes[pos];
            pos += 1;
            let packed = self.transitions[code as usize];
            let next = ((packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            if next == self.escape_sentinel {
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                let esc_packed = self.escape_transitions[b as usize];
                state = ((esc_packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            } else {
                state = next;
            }
        }
        state == self.accept_state
    }
}

// ---------------------------------------------------------------------------
// Hybrid 1: Prefilter + ShiftDfa — skip strings with no relevant codes,
// then use the fastest DFA (ShiftDfa) for survivors.
// ---------------------------------------------------------------------------

struct PrefilterShiftDfa {
    inner: ShiftDfa,
    relevant_codes: [bool; 256],
}

impl PrefilterShiftDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let inner = ShiftDfa::new(symbols, symbol_lengths, needle);

        let mut needle_bytes = [false; 256];
        for &b in needle {
            needle_bytes[b as usize] = true;
        }

        let mut relevant_codes = [false; 256];
        for (code, (sym, &sym_len)) in symbols.iter().zip(symbol_lengths.iter()).enumerate() {
            let sym_bytes = sym.to_u64().to_le_bytes();
            for &b in &sym_bytes[..sym_len as usize] {
                if needle_bytes[b as usize] {
                    relevant_codes[code] = true;
                    break;
                }
            }
        }
        relevant_codes[ESCAPE_CODE as usize] = true;

        Self {
            inner,
            relevant_codes,
        }
    }

    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        if !codes.iter().any(|&c| self.relevant_codes[c as usize]) {
            return false;
        }
        self.inner.matches_no_early_exit(codes)
    }
}

// ---------------------------------------------------------------------------
// Hybrid 2: StateZero skip + ShiftDfa — skip leading trivial codes,
// then use ShiftDfa for the remainder.
// ---------------------------------------------------------------------------

struct StateZeroShiftDfa {
    inner: ShiftDfa,
    trivial: [bool; 256],
}

impl StateZeroShiftDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let inner = ShiftDfa::new(symbols, symbol_lengths, needle);

        let mut trivial = [false; 256];
        for code in 0..256 {
            let packed = inner.transitions[code];
            let next = (packed & ShiftDfa::MASK) as u8;
            trivial[code] = next == 0 && code as u8 != ESCAPE_CODE;
        }

        Self { inner, trivial }
    }

    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        let mut start = 0;
        while start < codes.len() && self.trivial[codes[start] as usize] {
            start += 1;
        }
        if start == codes.len() {
            return self.inner.accept_state == 0;
        }
        self.inner.matches_no_early_exit(&codes[start..])
    }
}

// ---------------------------------------------------------------------------
// Approach 9: Sheng DFA — use SSSE3 PSHUFB for transitions.
//
// The state is a byte position in an XMM register. For each input byte,
// we load a 16-byte shuffle mask and do PSHUFB(mask, state_vec).
// PSHUFB uses the low 4 bits of each byte lane as an index into the mask,
// producing the next state. With ≤16 states this is a single instruction.
//
// The shuffle mask load depends only on the input byte (not on state),
// so it can be loaded in parallel with the previous PSHUFB's execution.
// Throughput: ~1 byte/cycle (limited by PSHUFB throughput of 1/cycle on
// most microarchitectures).
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
struct ShengDfa {
    /// 256 shuffle masks, one per possible input byte.
    /// Each mask is 16 bytes: mask[i] = next_state when current state == i.
    masks: Vec<std::arch::x86_64::__m128i>,
    /// 256 escape masks for escaped byte values.
    escape_masks: Vec<std::arch::x86_64::__m128i>,
    accept_state: u8,
    escape_sentinel: u8,
}

#[cfg(target_arch = "x86_64")]
impl ShengDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        use std::arch::x86_64::_mm_set_epi8;

        let fused = FusedTableDfa::new(symbols, symbol_lengths, needle);
        let escape_sentinel = fused.escape_sentinel as u8;

        let mut masks = Vec::with_capacity(256);
        let mut escape_masks = Vec::with_capacity(256);

        for code_byte in 0..256usize {
            let mut mask_bytes = [0u8; 16];
            for state in 0..16 {
                if state < needle.len() + 1 {
                    let next = fused.transitions[state * 256 + code_byte];
                    mask_bytes[state] = if next == fused.escape_sentinel {
                        escape_sentinel
                    } else {
                        next as u8
                    };
                }
            }
            masks.push(unsafe {
                _mm_set_epi8(
                    mask_bytes[15] as i8,
                    mask_bytes[14] as i8,
                    mask_bytes[13] as i8,
                    mask_bytes[12] as i8,
                    mask_bytes[11] as i8,
                    mask_bytes[10] as i8,
                    mask_bytes[9] as i8,
                    mask_bytes[8] as i8,
                    mask_bytes[7] as i8,
                    mask_bytes[6] as i8,
                    mask_bytes[5] as i8,
                    mask_bytes[4] as i8,
                    mask_bytes[3] as i8,
                    mask_bytes[2] as i8,
                    mask_bytes[1] as i8,
                    mask_bytes[0] as i8,
                )
            });
        }

        for byte_val in 0..256usize {
            let mut mask_bytes = [0u8; 16];
            for state in 0..16 {
                if state < needle.len() + 1 {
                    mask_bytes[state] = fused.escape_transitions[state * 256 + byte_val] as u8;
                }
            }
            escape_masks.push(unsafe {
                _mm_set_epi8(
                    mask_bytes[15] as i8,
                    mask_bytes[14] as i8,
                    mask_bytes[13] as i8,
                    mask_bytes[12] as i8,
                    mask_bytes[11] as i8,
                    mask_bytes[10] as i8,
                    mask_bytes[9] as i8,
                    mask_bytes[8] as i8,
                    mask_bytes[7] as i8,
                    mask_bytes[6] as i8,
                    mask_bytes[5] as i8,
                    mask_bytes[4] as i8,
                    mask_bytes[3] as i8,
                    mask_bytes[2] as i8,
                    mask_bytes[1] as i8,
                    mask_bytes[0] as i8,
                )
            });
        }

        Self {
            masks,
            escape_masks,
            accept_state: fused.accept_state as u8,
            escape_sentinel,
        }
    }

    #[inline]
    #[target_feature(enable = "ssse3")]
    unsafe fn matches(&self, codes: &[u8]) -> bool {
        use std::arch::x86_64::_mm_extract_epi8;
        use std::arch::x86_64::_mm_set1_epi8;
        use std::arch::x86_64::_mm_shuffle_epi8;

        unsafe {
            let mut state_vec = _mm_set1_epi8(0);
            let mut pos = 0;

            while pos < codes.len() {
                let cur_state = _mm_extract_epi8::<0>(state_vec) as u8;
                if cur_state == self.accept_state {
                    return true;
                }

                let code = codes[pos];
                pos += 1;

                // One PSHUFB: the mask load depends only on `code`, not state.
                let next_vec = _mm_shuffle_epi8(self.masks[code as usize], state_vec);
                let next_state = _mm_extract_epi8::<0>(next_vec) as u8;

                if next_state == self.escape_sentinel {
                    if pos >= codes.len() {
                        return false;
                    }
                    let b = codes[pos];
                    pos += 1;
                    state_vec = _mm_shuffle_epi8(self.escape_masks[b as usize], state_vec);
                } else {
                    state_vec = next_vec;
                }
            }

            _mm_extract_epi8::<0>(state_vec) as u8 == self.accept_state
        }
    }

    #[inline]
    #[target_feature(enable = "ssse3")]
    unsafe fn matches_no_early_exit(&self, codes: &[u8]) -> bool {
        use std::arch::x86_64::_mm_extract_epi8;
        use std::arch::x86_64::_mm_set1_epi8;
        use std::arch::x86_64::_mm_shuffle_epi8;

        unsafe {
            let mut state_vec = _mm_set1_epi8(0);
            let mut pos = 0;

            while pos < codes.len() {
                let code = codes[pos];
                pos += 1;

                let next_vec = _mm_shuffle_epi8(self.masks[code as usize], state_vec);
                let next_state = _mm_extract_epi8::<0>(next_vec) as u8;

                if next_state == self.escape_sentinel {
                    if pos >= codes.len() {
                        return false;
                    }
                    let b = codes[pos];
                    pos += 1;
                    state_vec = _mm_shuffle_epi8(self.escape_masks[b as usize], state_vec);
                } else {
                    state_vec = next_vec;
                }
            }

            _mm_extract_epi8::<0>(state_vec) as u8 == self.accept_state
        }
    }
}

// ---------------------------------------------------------------------------
// Approach 10: Speculative/Enumerated DFA — run from ALL start states at once.
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

#[inline(never)]
fn run_compact(dfa: &CompactDfa, prep: &PreparedArray, out: &mut BitBufferMut) {
    for i in 0..prep.n {
        let start = prep.offsets[i];
        let end = prep.offsets[i + 1];
        if dfa.matches(&prep.all_bytes[start..end]) {
            out.set(i);
        }
    }
}

#[inline(never)]
fn run_prefilter(dfa: &PrefilterDfa, prep: &PreparedArray, out: &mut BitBufferMut) {
    for i in 0..prep.n {
        let start = prep.offsets[i];
        let end = prep.offsets[i + 1];
        if dfa.matches(&prep.all_bytes[start..end]) {
            out.set(i);
        }
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
// Alloc-free decompress + match: reuse a buffer, inline the decompress logic.
// This measures pure decompress+search cost without per-string allocation.
// ---------------------------------------------------------------------------

/// Decompress FSST codes into `buf`, returning the number of bytes written.
/// This avoids all allocation by writing into a caller-provided buffer.
#[inline]
fn decompress_into(codes: &[u8], symbols: &[Symbol], symbol_lengths: &[u8], buf: &mut Vec<u8>) {
    buf.clear();
    let mut pos = 0;
    while pos < codes.len() {
        let code = codes[pos];
        pos += 1;
        if code == ESCAPE_CODE {
            if pos < codes.len() {
                buf.push(codes[pos]);
                pos += 1;
            }
        } else {
            let sym = symbols[code as usize].to_u64().to_le_bytes();
            let len = symbol_lengths[code as usize] as usize;
            buf.extend_from_slice(&sym[..len]);
        }
    }
}

/// Alloc-free decompress + sliding window match using PreparedArray.
/// Pre-allocates the decompression buffer once outside the benchmark loop.
#[inline(never)]
fn run_decompress_match(
    prep: &PreparedArray,
    symbols: &[Symbol],
    symbol_lengths: &[u8],
    needle: &[u8],
    buf: &mut Vec<u8>,
    out: &mut BitBufferMut,
) {
    for i in 0..prep.n {
        let start = prep.offsets[i];
        let end = prep.offsets[i + 1];
        decompress_into(&prep.all_bytes[start..end], symbols, symbol_lengths, buf);
        if buf.windows(needle.len()).any(|w| w == needle) {
            out.set(i);
        }
    }
}

/// Alloc-free decompress + memmem match using PreparedArray.
#[inline(never)]
fn run_decompress_memmem(
    prep: &PreparedArray,
    symbols: &[Symbol],
    symbol_lengths: &[u8],
    needle: &[u8],
    buf: &mut Vec<u8>,
    out: &mut BitBufferMut,
) {
    let finder = memmem::Finder::new(needle);
    for i in 0..prep.n {
        let start = prep.offsets[i];
        let end = prep.offsets[i + 1];
        decompress_into(&prep.all_bytes[start..end], symbols, symbol_lengths, buf);
        if finder.find(buf).is_some() {
            out.set(i);
        }
    }
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

// ---------------------------------------------------------------------------
// Log lines generator (Apache/nginx-style access logs)
// ---------------------------------------------------------------------------

const LOG_METHODS: &[&str] = &["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"];
const LOG_PATHS: &[&str] = &[
    "/api/v1/users",
    "/api/v2/products/search",
    "/healthcheck",
    "/static/js/app.bundle.min.js",
    "/favicon.ico",
    "/login",
    "/dashboard/analytics",
    "/api/v1/orders/12345/status",
    "/graphql",
    "/metrics",
];
const LOG_STATUS: &[u16] = &[
    200, 200, 200, 200, 200, 201, 301, 302, 400, 403, 404, 500, 502,
];
const LOG_IPS: &[&str] = &[
    "192.168.1.1",
    "10.0.0.42",
    "172.16.0.100",
    "203.0.113.50",
    "198.51.100.23",
    "8.8.8.8",
    "1.1.1.1",
    "74.125.200.100",
    "151.101.1.69",
    "93.184.216.34",
];
const LOG_UAS: &[&str] = &[
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "curl/7.81.0",
    "python-requests/2.28.1",
    "Go-http-client/1.1",
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
];

fn generate_log_lines(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(456);
    (0..n)
        .map(|_| {
            let ip = LOG_IPS[rng.random_range(0..LOG_IPS.len())];
            let method = LOG_METHODS[rng.random_range(0..LOG_METHODS.len())];
            let path = LOG_PATHS[rng.random_range(0..LOG_PATHS.len())];
            let status = LOG_STATUS[rng.random_range(0..LOG_STATUS.len())];
            let size = rng.random_range(100..50000);
            let ua = LOG_UAS[rng.random_range(0..LOG_UAS.len())];
            format!(
                r#"{ip} - - [15/Mar/2024:10:{:02}:{:02} +0000] "{method} {path} HTTP/1.1" {status} {size} "-" "{ua}""#,
                rng.random_range(0..60u32),
                rng.random_range(0..60u32),
            )
        })
        .collect()
}

fn make_fsst_log_lines(n: usize) -> FSSTArray {
    let lines = generate_log_lines(n);
    let varbin = VarBinArray::from_iter(
        lines.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
}

const LOG_NEEDLE: &[u8] = b"Googlebot";

// ---------------------------------------------------------------------------
// JSON strings generator (typical API response payloads)
// ---------------------------------------------------------------------------

const JSON_NAMES: &[&str] = &[
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack",
];
const JSON_CITIES: &[&str] = &[
    "New York",
    "London",
    "Tokyo",
    "Berlin",
    "Sydney",
    "Toronto",
    "Paris",
    "Mumbai",
    "São Paulo",
    "Seoul",
];
const JSON_TAGS: &[&str] = &[
    "premium",
    "verified",
    "admin",
    "moderator",
    "subscriber",
    "trial",
    "enterprise",
    "developer",
];

fn generate_json_strings(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(789);
    (0..n)
        .map(|_| {
            let name = JSON_NAMES[rng.random_range(0..JSON_NAMES.len())];
            let city = JSON_CITIES[rng.random_range(0..JSON_CITIES.len())];
            let age = rng.random_range(18..80u32);
            let tag1 = JSON_TAGS[rng.random_range(0..JSON_TAGS.len())];
            let tag2 = JSON_TAGS[rng.random_range(0..JSON_TAGS.len())];
            let id = rng.random_range(10000..99999u32);
            format!(
                r#"{{"id":{id},"name":"{name}","age":{age},"city":"{city}","tags":["{tag1}","{tag2}"],"active":true}}"#
            )
        })
        .collect()
}

fn make_fsst_json_strings(n: usize) -> FSSTArray {
    let jsons = generate_json_strings(n);
    let varbin = VarBinArray::from_iter(
        jsons.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
}

const JSON_NEEDLE: &[u8] = b"enterprise";

// ---------------------------------------------------------------------------
// File paths generator (Unix-style paths with various depths)
// ---------------------------------------------------------------------------

const PATH_ROOTS: &[&str] = &[
    "/home/user",
    "/var/log",
    "/etc",
    "/usr/local/bin",
    "/opt/app",
    "/tmp",
    "/srv/www",
    "/data/warehouse",
];
const PATH_DIRS: &[&str] = &[
    "src",
    "build",
    "dist",
    "node_modules",
    "target/release",
    "config",
    ".cache",
    "logs/2024",
    "backups/daily",
    "migrations",
];
const PATH_FILES: &[&str] = &[
    "main.rs",
    "index.ts",
    "config.yaml",
    "Dockerfile",
    "schema.sql",
    "app.log",
    "data.parquet",
    "model.onnx",
    "README.md",
    "package.json",
];

fn generate_file_paths(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(321);
    (0..n)
        .map(|_| {
            let root = PATH_ROOTS[rng.random_range(0..PATH_ROOTS.len())];
            let dir = PATH_DIRS[rng.random_range(0..PATH_DIRS.len())];
            let file = PATH_FILES[rng.random_range(0..PATH_FILES.len())];
            let depth = rng.random_range(0..3u32);
            let mut path = format!("{root}/{dir}");
            for _ in 0..depth {
                let subdir = PATH_DIRS[rng.random_range(0..PATH_DIRS.len())];
                path.push('/');
                path.push_str(subdir);
            }
            path.push('/');
            path.push_str(file);
            path
        })
        .collect()
}

fn make_fsst_file_paths(n: usize) -> FSSTArray {
    let paths = generate_file_paths(n);
    let varbin = VarBinArray::from_iter(
        paths.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
}

const PATH_NEEDLE: &[u8] = b"target/release";

// ---------------------------------------------------------------------------
// Email addresses generator
// ---------------------------------------------------------------------------

const EMAIL_USERS: &[&str] = &[
    "john.doe",
    "jane.smith",
    "admin",
    "support",
    "no-reply",
    "sales.team",
    "dev+test",
    "marketing",
    "info",
    "contact.us",
];
const EMAIL_DOMAINS: &[&str] = &[
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "company.io",
    "example.org",
    "mail.ru",
    "protonmail.com",
    "fastmail.com",
    "icloud.com",
    "hey.com",
];

fn generate_emails(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(654);
    (0..n)
        .map(|_| {
            let user = EMAIL_USERS[rng.random_range(0..EMAIL_USERS.len())];
            let domain = EMAIL_DOMAINS[rng.random_range(0..EMAIL_DOMAINS.len())];
            let suffix = rng.random_range(0..1000u32);
            format!("{user}{suffix}@{domain}")
        })
        .collect()
}

fn make_fsst_emails(n: usize) -> FSSTArray {
    let emails = generate_emails(n);
    let varbin = VarBinArray::from_iter(
        emails.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
}

const EMAIL_NEEDLE: &[u8] = b"gmail";

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

// 8. Decompress then search (worst-case baseline, allocates per string)
#[divan::bench]
fn decompress_then_search(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let mut out = Vec::with_capacity(N);
    bencher.bench_local(|| {
        bench_decompress(&fsst, NEEDLE, &mut out);
    });
}

// 8b. Alloc-free decompress + sliding window match
#[divan::bench]
fn decompress_no_alloc(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let symbols = fsst.symbols();
    let symbol_lengths = fsst.symbol_lengths();
    let mut buf = Vec::with_capacity(256);
    let mut out = BitBufferMut::new_unset(N);
    bencher.bench_local(|| {
        out.fill_range(0, N, false);
        run_decompress_match(
            &prep,
            symbols.as_slice(),
            symbol_lengths.as_slice(),
            NEEDLE,
            &mut buf,
            &mut out,
        );
    });
}

// 8c. Alloc-free decompress + memmem (SIMD substring search)
#[divan::bench]
fn decompress_no_alloc_memmem(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let symbols = fsst.symbols();
    let symbol_lengths = fsst.symbol_lengths();
    let mut buf = Vec::with_capacity(256);
    let mut out = BitBufferMut::new_unset(N);
    bencher.bench_local(|| {
        out.fill_range(0, N, false);
        run_decompress_memmem(
            &prep,
            symbols.as_slice(),
            symbol_lengths.as_slice(),
            NEEDLE,
            &mut buf,
            &mut out,
        );
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

// 11. Compact u8 table (halved table size)
dfa_bench!(compact_table, CompactDfa, run_compact);

// 12. Compact u8 + collect_bool
#[divan::bench]
fn compact_chunk_64(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = CompactDfa::new(
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

// 13. Compact u8 + collect_bool + unsafe
#[divan::bench]
fn compact_chunk_64_unsafe(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = CompactDfa::new(
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

// 14. Prefilter (skip strings with no relevant codes)
dfa_bench!(prefilter, PrefilterDfa, run_prefilter);

// 15. Prefilter + collect_bool
#[divan::bench]
fn prefilter_chunk_64(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = PrefilterDfa::new(
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

// 16. Streaming continuous scan (single pass through all codes)
#[divan::bench]
fn streaming_continuous(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = CompactDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        NEEDLE,
    );
    let mut out = BitBufferMut::new_unset(N);
    bencher.bench_local(|| {
        out.fill_range(0, N, false);
        streaming_scan_continuous(&dfa, &prep.all_bytes, &prep.offsets, prep.n, &mut out);
    });
}

// 17. Shift-based DFA (u64 packed transitions)
#[divan::bench]
fn shift_dfa(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = ShiftDfa::new(
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

// 18. Shift-based DFA, no early exit
#[divan::bench]
fn shift_dfa_no_exit(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = ShiftDfa::new(
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

// 19. Sheng DFA (PSHUFB transitions)
#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn sheng_dfa(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = ShengDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        NEEDLE,
    );
    bencher.bench_local(|| {
        BitBufferMut::collect_bool(prep.n, |i| {
            let start = prep.offsets[i];
            let end = prep.offsets[i + 1];
            unsafe { dfa.matches(&prep.all_bytes[start..end]) }
        })
    });
}

// 20. Sheng DFA, no early exit
#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn sheng_dfa_no_exit(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = ShengDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        NEEDLE,
    );
    bencher.bench_local(|| {
        BitBufferMut::collect_bool(prep.n, |i| {
            let start = prep.offsets[i];
            let end = prep.offsets[i + 1];
            unsafe { dfa.matches_no_early_exit(&prep.all_bytes[start..end]) }
        })
    });
}

// 21. Enumerated DFA (track all start states)
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

// Aho-Corasick on decompressed data: decompress each string then search with aho-corasick
#[divan::bench]
fn aho_corasick_decompress(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let ac = AhoCorasick::new([NEEDLE]).unwrap();
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        let decompressor = fsst.decompressor();
        fsst.codes().with_iterator(|iter| {
            out.extend(iter.map(|codes| match codes {
                Some(c) => {
                    let decompressed = decompressor.decompress(c);
                    ac.is_match(&decompressed)
                }
                None => false,
            }));
        });
        out
    });
}

// Aho-Corasick on raw (canonicalized) bytes: decompress the whole array up front,
// then search each string using aho-corasick's SIMD-accelerated search
#[divan::bench]
fn aho_corasick_on_raw_bytes(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let canonical = fsst.to_canonical().unwrap().into_varbinview();
    let ac = AhoCorasick::new([NEEDLE]).unwrap();
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        canonical.with_iterator(|iter| {
            out.extend(iter.map(|s| match s {
                Some(bytes) => ac.is_match(bytes),
                None => false,
            }));
        });
        out
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
fn cb_shift_dfa(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = ShiftDfa::new(
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

#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn cb_sheng_dfa(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = ShengDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        CB_NEEDLE,
    );
    bencher.bench_local(|| {
        BitBufferMut::collect_bool(prep.n, |i| {
            let start = prep.offsets[i];
            let end = prep.offsets[i + 1];
            unsafe { dfa.matches_no_early_exit(&prep.all_bytes[start..end]) }
        })
    });
}

#[divan::bench]
fn cb_compact_chunk_64_unsafe(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = CompactDfa::new(
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
fn cb_prefilter_chunk_64(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = PrefilterDfa::new(
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
fn cb_streaming_continuous(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = CompactDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        CB_NEEDLE,
    );
    let mut out = BitBufferMut::new_unset(N);
    bencher.bench_local(|| {
        out.fill_range(0, N, false);
        streaming_scan_continuous(&dfa, &prep.all_bytes, &prep.offsets, prep.n, &mut out);
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

#[divan::bench]
fn cb_decompress_no_alloc(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let symbols = fsst.symbols();
    let symbol_lengths = fsst.symbol_lengths();
    let mut buf = Vec::with_capacity(512);
    let mut out = BitBufferMut::new_unset(N);
    bencher.bench_local(|| {
        out.fill_range(0, N, false);
        run_decompress_match(
            &prep,
            symbols.as_slice(),
            symbol_lengths.as_slice(),
            CB_NEEDLE,
            &mut buf,
            &mut out,
        );
    });
}

#[divan::bench]
fn cb_decompress_no_alloc_memmem(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let symbols = fsst.symbols();
    let symbol_lengths = fsst.symbol_lengths();
    let mut buf = Vec::with_capacity(512);
    let mut out = BitBufferMut::new_unset(N);
    bencher.bench_local(|| {
        out.fill_range(0, N, false);
        run_decompress_memmem(
            &prep,
            symbols.as_slice(),
            symbol_lengths.as_slice(),
            CB_NEEDLE,
            &mut buf,
            &mut out,
        );
    });
}

#[divan::bench]
fn cb_aho_corasick_decompress(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let ac = AhoCorasick::new([CB_NEEDLE]).unwrap();
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        let decompressor = fsst.decompressor();
        fsst.codes().with_iterator(|iter| {
            out.extend(iter.map(|codes| match codes {
                Some(c) => {
                    let decompressed = decompressor.decompress(c);
                    ac.is_match(&decompressed)
                }
                None => false,
            }));
        });
        out
    });
}

#[divan::bench]
fn cb_aho_corasick_on_raw_bytes(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let canonical = fsst.to_canonical().unwrap().into_varbinview();
    let ac = AhoCorasick::new([CB_NEEDLE]).unwrap();
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        canonical.with_iterator(|iter| {
            out.extend(iter.map(|s| match s {
                Some(bytes) => ac.is_match(bytes),
                None => false,
            }));
        });
        out
    });
}

// ---------------------------------------------------------------------------
// Benchmarks for additional data types (log lines, JSON, file paths, emails)
// ---------------------------------------------------------------------------

/// Macro for benchmarks on a specific data generator + needle combo.
macro_rules! data_bench {
    ($name:ident, $make_fn:ident, $needle:expr, $dfa_ty:ident, $match_method:ident) => {
        #[divan::bench]
        fn $name(bencher: Bencher) {
            let fsst = $make_fn(N);
            let prep = PreparedArray::from_fsst(&fsst);
            let dfa = $dfa_ty::new(
                fsst.symbols().as_slice(),
                fsst.symbol_lengths().as_slice(),
                $needle,
            );
            bencher.bench_local(|| {
                BitBufferMut::collect_bool(prep.n, |i| {
                    let start = prep.offsets[i];
                    let end = prep.offsets[i + 1];
                    dfa.$match_method(&prep.all_bytes[start..end])
                })
            });
        }
    };
}

// Log lines: long strings (~150 chars), low match rate for "Googlebot"
data_bench!(
    log_split_table,
    make_fsst_log_lines,
    LOG_NEEDLE,
    SplitTableDfa,
    matches
);
data_bench!(
    log_shift_dfa,
    make_fsst_log_lines,
    LOG_NEEDLE,
    ShiftDfa,
    matches_no_early_exit
);
data_bench!(
    log_compact_no_exit,
    make_fsst_log_lines,
    LOG_NEEDLE,
    CompactDfa,
    matches_no_early_exit
);
data_bench!(
    log_fused_no_exit,
    make_fsst_log_lines,
    LOG_NEEDLE,
    FusedTableDfa,
    matches_no_early_exit
);

#[divan::bench]
fn log_decompress(bencher: Bencher) {
    let fsst = make_fsst_log_lines(N);
    let mut out = Vec::with_capacity(N);
    bencher.bench_local(|| {
        bench_decompress(&fsst, LOG_NEEDLE, &mut out);
    });
}

// JSON strings: structured data (~80-100 chars), searching for "enterprise"
data_bench!(
    json_split_table,
    make_fsst_json_strings,
    JSON_NEEDLE,
    SplitTableDfa,
    matches
);
data_bench!(
    json_shift_dfa,
    make_fsst_json_strings,
    JSON_NEEDLE,
    ShiftDfa,
    matches_no_early_exit
);
data_bench!(
    json_compact_no_exit,
    make_fsst_json_strings,
    JSON_NEEDLE,
    CompactDfa,
    matches_no_early_exit
);
data_bench!(
    json_fused_no_exit,
    make_fsst_json_strings,
    JSON_NEEDLE,
    FusedTableDfa,
    matches_no_early_exit
);

#[divan::bench]
fn json_decompress(bencher: Bencher) {
    let fsst = make_fsst_json_strings(N);
    let mut out = Vec::with_capacity(N);
    bencher.bench_local(|| {
        bench_decompress(&fsst, JSON_NEEDLE, &mut out);
    });
}

// File paths: medium-length (~40-80 chars), searching for "target/release"
data_bench!(
    path_split_table,
    make_fsst_file_paths,
    PATH_NEEDLE,
    SplitTableDfa,
    matches
);
data_bench!(
    path_shift_dfa,
    make_fsst_file_paths,
    PATH_NEEDLE,
    ShiftDfa,
    matches_no_early_exit
);
data_bench!(
    path_compact_no_exit,
    make_fsst_file_paths,
    PATH_NEEDLE,
    CompactDfa,
    matches_no_early_exit
);
data_bench!(
    path_fused_no_exit,
    make_fsst_file_paths,
    PATH_NEEDLE,
    FusedTableDfa,
    matches_no_early_exit
);

#[divan::bench]
fn path_decompress(bencher: Bencher) {
    let fsst = make_fsst_file_paths(N);
    let mut out = Vec::with_capacity(N);
    bencher.bench_local(|| {
        bench_decompress(&fsst, PATH_NEEDLE, &mut out);
    });
}

// Email addresses: short strings (~20-30 chars), searching for "gmail"
data_bench!(
    email_split_table,
    make_fsst_emails,
    EMAIL_NEEDLE,
    SplitTableDfa,
    matches
);
data_bench!(
    email_shift_dfa,
    make_fsst_emails,
    EMAIL_NEEDLE,
    ShiftDfa,
    matches_no_early_exit
);
data_bench!(
    email_compact_no_exit,
    make_fsst_emails,
    EMAIL_NEEDLE,
    CompactDfa,
    matches_no_early_exit
);
data_bench!(
    email_fused_no_exit,
    make_fsst_emails,
    EMAIL_NEEDLE,
    FusedTableDfa,
    matches_no_early_exit
);

#[divan::bench]
fn email_decompress(bencher: Bencher) {
    let fsst = make_fsst_emails(N);
    let mut out = Vec::with_capacity(N);
    bencher.bench_local(|| {
        bench_decompress(&fsst, EMAIL_NEEDLE, &mut out);
    });
}

// ---------------------------------------------------------------------------
// memchr::memmem benchmarks — SIMD-accelerated substring search on decompressed data
// ---------------------------------------------------------------------------

#[divan::bench]
fn memmem_decompress_urls(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let finder = memmem::Finder::new(NEEDLE);
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        let decompressor = fsst.decompressor();
        fsst.codes().with_iterator(|iter| {
            out.extend(iter.map(|codes| match codes {
                Some(c) => {
                    let decompressed = decompressor.decompress(c);
                    finder.find(&decompressed).is_some()
                }
                None => false,
            }));
        });
        out
    });
}

#[divan::bench]
fn memmem_on_raw_bytes_urls(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let canonical = fsst.to_canonical().unwrap().into_varbinview();
    let finder = memmem::Finder::new(NEEDLE);
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        canonical.with_iterator(|iter| {
            out.extend(iter.map(|s| match s {
                Some(bytes) => finder.find(bytes).is_some(),
                None => false,
            }));
        });
        out
    });
}

#[divan::bench]
fn cb_memmem_decompress(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let finder = memmem::Finder::new(CB_NEEDLE);
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        let decompressor = fsst.decompressor();
        fsst.codes().with_iterator(|iter| {
            out.extend(iter.map(|codes| match codes {
                Some(c) => {
                    let decompressed = decompressor.decompress(c);
                    finder.find(&decompressed).is_some()
                }
                None => false,
            }));
        });
        out
    });
}

#[divan::bench]
fn cb_memmem_on_raw_bytes(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let canonical = fsst.to_canonical().unwrap().into_varbinview();
    let finder = memmem::Finder::new(CB_NEEDLE);
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        canonical.with_iterator(|iter| {
            out.extend(iter.map(|s| match s {
                Some(bytes) => finder.find(bytes).is_some(),
                None => false,
            }));
        });
        out
    });
}

// ---------------------------------------------------------------------------
// Low match rate (~0.001%) benchmarks — needle appears in ~1/100K strings.
// Tests performance when almost no string matches (common in large datasets).
// Uses random alphanumeric strings with a rare injected match.
// ---------------------------------------------------------------------------

const RARE_NEEDLE: &[u8] = b"xyzzy";

/// Generate N random alphanumeric strings (~40 chars each), injecting the needle
/// into approximately `match_rate` fraction of them.
fn generate_rare_match_strings(n: usize, match_rate: f64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(999);
    let charset: &[u8] = b"abcdefghijklmnopqrstuvwABCDEFGHIJKLMNOPQRSTUVW0123456789-_.:/";
    (0..n)
        .map(|_| {
            let len = rng.random_range(30..60);
            let mut s: String = (0..len)
                .map(|_| charset[rng.random_range(0..charset.len())] as char)
                .collect();
            if rng.random_bool(match_rate) {
                // Inject needle at random position
                let pos = rng.random_range(0..s.len().saturating_sub(RARE_NEEDLE.len()) + 1);
                s.replace_range(
                    pos..pos + RARE_NEEDLE.len().min(s.len() - pos),
                    std::str::from_utf8(RARE_NEEDLE).unwrap(),
                );
            }
            s
        })
        .collect()
}

fn make_fsst_rare_match(n: usize) -> FSSTArray {
    let strings = generate_rare_match_strings(n, 0.00001); // ~0.001%
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
}

data_bench!(
    rare_split_table,
    make_fsst_rare_match,
    RARE_NEEDLE,
    SplitTableDfa,
    matches
);
data_bench!(
    rare_shift_dfa,
    make_fsst_rare_match,
    RARE_NEEDLE,
    ShiftDfa,
    matches_no_early_exit
);
data_bench!(
    rare_compact_no_exit,
    make_fsst_rare_match,
    RARE_NEEDLE,
    CompactDfa,
    matches_no_early_exit
);
data_bench!(
    rare_fused_no_exit,
    make_fsst_rare_match,
    RARE_NEEDLE,
    FusedTableDfa,
    matches_no_early_exit
);

#[divan::bench]
fn rare_decompress(bencher: Bencher) {
    let fsst = make_fsst_rare_match(N);
    let mut out = Vec::with_capacity(N);
    bencher.bench_local(|| {
        bench_decompress(&fsst, RARE_NEEDLE, &mut out);
    });
}

#[divan::bench]
fn rare_memmem_decompress(bencher: Bencher) {
    let fsst = make_fsst_rare_match(N);
    let finder = memmem::Finder::new(RARE_NEEDLE);
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        let decompressor = fsst.decompressor();
        fsst.codes().with_iterator(|iter| {
            out.extend(iter.map(|codes| match codes {
                Some(c) => {
                    let decompressed = decompressor.decompress(c);
                    finder.find(&decompressed).is_some()
                }
                None => false,
            }));
        });
        out
    });
}

#[divan::bench]
fn rare_prefilter(bencher: Bencher) {
    let fsst = make_fsst_rare_match(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = PrefilterDfa::new(
        fsst.symbols().as_slice(),
        fsst.symbol_lengths().as_slice(),
        RARE_NEEDLE,
    );
    bencher.bench_local(|| {
        BitBufferMut::collect_bool(prep.n, |i| {
            let start = prep.offsets[i];
            let end = prep.offsets[i + 1];
            dfa.matches_no_early_exit(&prep.all_bytes[start..end])
        })
    });
}

data_bench!(
    rare_state_zero_skip,
    make_fsst_rare_match,
    RARE_NEEDLE,
    StateZeroSkipDfa,
    matches
);

// State-zero skip on URLs (moderate match rate)
data_bench!(
    state_zero_skip_urls,
    make_fsst_urls,
    NEEDLE,
    StateZeroSkipDfa,
    matches
);

// State-zero skip on ClickBench URLs
#[divan::bench]
fn cb_state_zero_skip(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = StateZeroSkipDfa::new(
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

// ---------------------------------------------------------------------------
// Alloc-free decompress benchmarks for all data types
// ---------------------------------------------------------------------------

macro_rules! decompress_no_alloc_bench {
    ($name:ident, $make_fn:ident, $needle:expr, $bufsz:expr) => {
        #[divan::bench]
        fn $name(bencher: Bencher) {
            let fsst = $make_fn(N);
            let prep = PreparedArray::from_fsst(&fsst);
            let symbols = fsst.symbols();
            let symbol_lengths = fsst.symbol_lengths();
            let mut buf = Vec::with_capacity($bufsz);
            let mut out = BitBufferMut::new_unset(N);
            bencher.bench_local(|| {
                out.fill_range(0, N, false);
                run_decompress_memmem(
                    &prep,
                    symbols.as_slice(),
                    symbol_lengths.as_slice(),
                    $needle,
                    &mut buf,
                    &mut out,
                );
            });
        }
    };
}

decompress_no_alloc_bench!(
    log_decompress_no_alloc,
    make_fsst_log_lines,
    LOG_NEEDLE,
    256
);
decompress_no_alloc_bench!(
    json_decompress_no_alloc,
    make_fsst_json_strings,
    JSON_NEEDLE,
    256
);
decompress_no_alloc_bench!(
    path_decompress_no_alloc,
    make_fsst_file_paths,
    PATH_NEEDLE,
    256
);
decompress_no_alloc_bench!(
    email_decompress_no_alloc,
    make_fsst_emails,
    EMAIL_NEEDLE,
    64
);
decompress_no_alloc_bench!(
    rare_decompress_no_alloc,
    make_fsst_rare_match,
    RARE_NEEDLE,
    128
);

// ---------------------------------------------------------------------------
// regex-automata DFA benchmarks
// ---------------------------------------------------------------------------

#[divan::bench]
fn regex_automata_dense_decompress(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let re = DfaRegex::new(std::str::from_utf8(NEEDLE).unwrap()).unwrap();
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        let decompressor = fsst.decompressor();
        fsst.codes().with_iterator(|iter| {
            out.extend(iter.map(|codes| match codes {
                Some(c) => {
                    let decompressed = decompressor.decompress(c);
                    re.is_match(&decompressed)
                }
                None => false,
            }));
        });
        out
    });
}

#[divan::bench]
fn regex_automata_dense_on_raw_bytes(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let canonical = fsst.to_canonical().unwrap().into_varbinview();
    let re = DfaRegex::new(std::str::from_utf8(NEEDLE).unwrap()).unwrap();
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        canonical.with_iterator(|iter| {
            out.extend(iter.map(|s| match s {
                Some(bytes) => re.is_match(bytes),
                None => false,
            }));
        });
        out
    });
}

#[divan::bench]
fn regex_automata_sparse_decompress(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let dense = DfaRegex::new(std::str::from_utf8(NEEDLE).unwrap()).unwrap();
    let (fwd, rev) = (
        dense.forward().to_sparse().unwrap(),
        dense.reverse().to_sparse().unwrap(),
    );
    let re = regex_automata::dfa::regex::Regex::builder().build_from_dfas(fwd, rev);
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        let decompressor = fsst.decompressor();
        fsst.codes().with_iterator(|iter| {
            out.extend(iter.map(|codes| match codes {
                Some(c) => {
                    let decompressed = decompressor.decompress(c);
                    re.is_match(&decompressed)
                }
                None => false,
            }));
        });
        out
    });
}

#[divan::bench]
fn regex_automata_sparse_on_raw_bytes(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let canonical = fsst.to_canonical().unwrap().into_varbinview();
    let dense = DfaRegex::new(std::str::from_utf8(NEEDLE).unwrap()).unwrap();
    let (fwd, rev) = (
        dense.forward().to_sparse().unwrap(),
        dense.reverse().to_sparse().unwrap(),
    );
    let re = regex_automata::dfa::regex::Regex::builder().build_from_dfas(fwd, rev);
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        canonical.with_iterator(|iter| {
            out.extend(iter.map(|s| match s {
                Some(bytes) => re.is_match(bytes),
                None => false,
            }));
        });
        out
    });
}

// ---------------------------------------------------------------------------
// jetscii benchmarks — PCMPESTRI-based substring search
// ---------------------------------------------------------------------------

#[divan::bench]
fn jetscii_decompress(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let finder = jetscii::ByteSubstring::new(NEEDLE);
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        let decompressor = fsst.decompressor();
        fsst.codes().with_iterator(|iter| {
            out.extend(iter.map(|codes| match codes {
                Some(c) => {
                    let decompressed = decompressor.decompress(c);
                    finder.find(&decompressed).is_some()
                }
                None => false,
            }));
        });
        out
    });
}

#[divan::bench]
fn jetscii_on_raw_bytes(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let canonical = fsst.to_canonical().unwrap().into_varbinview();
    let finder = jetscii::ByteSubstring::new(NEEDLE);
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        canonical.with_iterator(|iter| {
            out.extend(iter.map(|s| match s {
                Some(bytes) => finder.find(bytes).is_some(),
                None => false,
            }));
        });
        out
    });
}

// ---------------------------------------------------------------------------
// daachorse benchmarks — double-array Aho-Corasick
// ---------------------------------------------------------------------------

#[divan::bench]
fn daachorse_decompress(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let ac = DoubleArrayAhoCorasick::<u32>::new([NEEDLE]).unwrap();
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        let decompressor = fsst.decompressor();
        fsst.codes().with_iterator(|iter| {
            out.extend(iter.map(|codes| match codes {
                Some(c) => {
                    let decompressed = decompressor.decompress(c);
                    ac.find_iter(&decompressed).next().is_some()
                }
                None => false,
            }));
        });
        out
    });
}

#[divan::bench]
fn daachorse_on_raw_bytes(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    let canonical = fsst.to_canonical().unwrap().into_varbinview();
    let ac = DoubleArrayAhoCorasick::<u32>::new([NEEDLE]).unwrap();
    bencher.bench_local(|| {
        let mut out = Vec::with_capacity(N);
        canonical.with_iterator(|iter| {
            out.extend(iter.map(|s| match s {
                Some(bytes) => ac.find_iter(bytes).next().is_some(),
                None => false,
            }));
        });
        out
    });
}

// ---------------------------------------------------------------------------
// Hybrid DFA benchmarks
// ---------------------------------------------------------------------------

data_bench!(
    prefilter_shift_urls,
    make_fsst_urls,
    NEEDLE,
    PrefilterShiftDfa,
    matches
);
data_bench!(
    prefilter_shift_rare,
    make_fsst_rare_match,
    RARE_NEEDLE,
    PrefilterShiftDfa,
    matches
);
data_bench!(
    state_zero_shift_urls,
    make_fsst_urls,
    NEEDLE,
    StateZeroShiftDfa,
    matches
);
data_bench!(
    state_zero_shift_rare,
    make_fsst_rare_match,
    RARE_NEEDLE,
    StateZeroShiftDfa,
    matches
);

#[divan::bench]
fn cb_prefilter_shift(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = PrefilterShiftDfa::new(
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
fn cb_state_zero_shift(bencher: Bencher) {
    let fsst = make_fsst_clickbench_urls(N);
    let prep = PreparedArray::from_fsst(&fsst);
    let dfa = StateZeroShiftDfa::new(
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

// ---------------------------------------------------------------------------
// Decompress-only benchmarks (no search) — measures the raw cost of FSST
// decompression for each dataset. Compare against DFA search on compressed
// codes to see the speedup from avoiding decompression entirely.
// ---------------------------------------------------------------------------

/// Decompress all strings without searching. Measures pure decompression cost.
#[inline(never)]
fn run_decompress_only(
    prep: &PreparedArray,
    symbols: &[Symbol],
    symbol_lengths: &[u8],
    buf: &mut Vec<u8>,
) {
    for i in 0..prep.n {
        let start = prep.offsets[i];
        let end = prep.offsets[i + 1];
        decompress_into(&prep.all_bytes[start..end], symbols, symbol_lengths, buf);
        // Force the compiler not to optimize away the decompression.
        std::hint::black_box(buf.len());
    }
}

macro_rules! decompress_only_bench {
    ($name:ident, $make_fn:ident, $bufsz:expr) => {
        #[divan::bench]
        fn $name(bencher: Bencher) {
            let fsst = $make_fn(N);
            let prep = PreparedArray::from_fsst(&fsst);
            let symbols = fsst.symbols();
            let symbol_lengths = fsst.symbol_lengths();
            let mut buf = Vec::with_capacity($bufsz);
            bencher.bench_local(|| {
                run_decompress_only(
                    &prep,
                    symbols.as_slice(),
                    symbol_lengths.as_slice(),
                    &mut buf,
                );
            });
        }
    };
}

decompress_only_bench!(urls_decompress_only, make_fsst_urls, 256);
decompress_only_bench!(cb_decompress_only, make_fsst_clickbench_urls, 512);
decompress_only_bench!(log_decompress_only, make_fsst_log_lines, 256);
decompress_only_bench!(json_decompress_only, make_fsst_json_strings, 256);
decompress_only_bench!(path_decompress_only, make_fsst_file_paths, 256);
decompress_only_bench!(email_decompress_only, make_fsst_emails, 64);
decompress_only_bench!(rare_decompress_only, make_fsst_rare_match, 128);

// ---------------------------------------------------------------------------
// Vortex array LIKE kernel benchmarks — end-to-end through the full vortex
// execution framework. This measures the production code path including
// array construction, kernel dispatch, and result materialization.
// ---------------------------------------------------------------------------

use std::sync::LazyLock;

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::session::ArraySession;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

macro_rules! vortex_like_bench {
    ($name:ident, $make_fn:ident, $pattern:expr) => {
        #[divan::bench]
        fn $name(bencher: Bencher) {
            let fsst = $make_fn(N);
            let len = fsst.len();
            let arr = fsst.into_array();
            let pattern = ConstantArray::new($pattern, len).into_array();
            bencher.bench_local(|| {
                Like.try_new_array(len, LikeOptions::default(), [arr.clone(), pattern.clone()])
                    .unwrap()
                    .into_array()
                    .execute::<Canonical>(&mut SESSION.create_execution_ctx())
                    .unwrap()
            });
        }
    };
}

vortex_like_bench!(vortex_like_urls, make_fsst_urls, "%google%");
vortex_like_bench!(vortex_like_cb, make_fsst_clickbench_urls, "%yandex%");
vortex_like_bench!(vortex_like_log, make_fsst_log_lines, "%Googlebot%");
vortex_like_bench!(vortex_like_json, make_fsst_json_strings, "%enterprise%");
vortex_like_bench!(vortex_like_path, make_fsst_file_paths, "%target/release%");
vortex_like_bench!(vortex_like_email, make_fsst_emails, "%gmail%");
vortex_like_bench!(vortex_like_rare, make_fsst_rare_match, "%xyzzy%");
