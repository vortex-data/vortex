// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

use fsst::ESCAPE_CODE;
use fsst::Symbol;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::BoolArray;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::FSST;
use crate::FSSTArray;

impl LikeKernel for FSST {
    #[allow(clippy::cast_possible_truncation)]
    fn like(
        array: &FSSTArray,
        pattern: &ArrayRef,
        options: LikeOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(pattern_scalar) = pattern.as_constant() else {
            return Ok(None);
        };

        if options.case_insensitive {
            return Ok(None);
        }

        let Some(pattern_str) = pattern_scalar.as_utf8().value() else {
            return Ok(None);
        };

        let Some(like_kind) = LikeKind::parse(pattern_str) else {
            return Ok(None);
        };

        let symbols = array.symbols();
        let symbol_lengths = array.symbol_lengths();
        let negated = options.negated;

        // Access the underlying codes VarBinArray buffers directly to avoid
        // dyn Iterator overhead from with_iterator.
        let codes = array.codes();
        let offsets = codes.offsets().to_primitive();
        let all_bytes = codes.bytes();
        let all_bytes = all_bytes.as_slice();
        let n = codes.len();

        let result = match like_kind {
            LikeKind::Prefix(prefix) => {
                let prefix = prefix.as_bytes();
                // FsstPrefixDfa uses 4-bit shift packing: prefix_len + 2 states must fit in 16.
                if prefix.len() + 2 > (1 << FsstPrefixDfa::BITS) {
                    return Ok(None);
                }
                let dfa = FsstPrefixDfa::new(symbols.as_slice(), symbol_lengths.as_slice(), prefix);
                match_each_integer_ptype!(offsets.ptype(), |T| {
                    let off = offsets.as_slice::<T>();
                    dfa_scan_to_bitbuf(n, off, all_bytes, negated, |codes| dfa.matches(codes))
                })
            }
            LikeKind::Contains(needle) => {
                let needle = needle.as_bytes();
                if needle.len() <= BranchlessShiftDfa::MAX_NEEDLE_LEN {
                    let dfa = BranchlessShiftDfa::new(
                        symbols.as_slice(),
                        symbol_lengths.as_slice(),
                        needle,
                    );
                    match_each_integer_ptype!(offsets.ptype(), |T| {
                        let off = offsets.as_slice::<T>();
                        dfa_scan_to_bitbuf(n, off, all_bytes, negated, |codes| dfa.matches(codes))
                    })
                } else if needle.len() <= FlatBranchlessDfa::MAX_NEEDLE_LEN {
                    let dfa = FlatBranchlessDfa::new(
                        symbols.as_slice(),
                        symbol_lengths.as_slice(),
                        needle,
                    );
                    match_each_integer_ptype!(offsets.ptype(), |T| {
                        let off = offsets.as_slice::<T>();
                        dfa_scan_to_bitbuf(n, off, all_bytes, negated, |codes| dfa.matches(codes))
                    })
                } else {
                    let dfa =
                        FsstContainsDfa::new(symbols.as_slice(), symbol_lengths.as_slice(), needle);
                    match_each_integer_ptype!(offsets.ptype(), |T| {
                        let off = offsets.as_slice::<T>();
                        dfa_scan_to_bitbuf(n, off, all_bytes, negated, |codes| dfa.matches(codes))
                    })
                }
            }
        };

        // FSST delegates validity to its codes array, so we can read it
        // directly without cloning the entire FSSTArray into an ArrayRef.
        let validity = array
            .codes()
            .validity()?
            .union_nullability(pattern_scalar.dtype().nullability());

        Ok(Some(BoolArray::new(result, validity).into_array()))
    }
}

/// Scan all strings through a DFA matcher, packing results directly into a
/// `BitBuffer` one u64 word (64 strings) at a time. This avoids the overhead
/// of `BitBufferMut::collect_bool`'s cross-crate closure indirection and
/// guarantees the compiler can see the full loop body for optimization.
// TODO: add N-way ILP overrun scan for higher throughput on short strings.
#[inline]
fn dfa_scan_to_bitbuf<T, F>(
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    matcher: F,
) -> BitBuffer
where
    T: vortex_array::dtype::IntegerPType,
    F: Fn(&[u8]) -> bool,
{
    let n_words = n / 64;
    let remainder = n % 64;
    let mut words: BufferMut<u64> = BufferMut::with_capacity(n.div_ceil(64));

    for chunk in 0..n_words {
        let base = chunk * 64;
        let mut word = 0u64;
        let mut start: usize = offsets[base].as_();
        for bit in 0..64 {
            let end: usize = offsets[base + bit + 1].as_();
            word |= ((matcher(&all_bytes[start..end]) != negated) as u64) << bit;
            start = end;
        }
        // SAFETY: we allocated capacity for n.div_ceil(64) words.
        unsafe { words.push_unchecked(word) };
    }

    if remainder != 0 {
        let base = n_words * 64;
        let mut word = 0u64;
        let mut start: usize = offsets[base].as_();
        for bit in 0..remainder {
            let end: usize = offsets[base + bit + 1].as_();
            word |= ((matcher(&all_bytes[start..end]) != negated) as u64) << bit;
            start = end;
        }
        unsafe { words.push_unchecked(word) };
    }

    BitBuffer::new(words.into_byte_buffer().freeze(), n)
}

/// The subset of LIKE patterns we can handle without decompression.
enum LikeKind<'a> {
    /// `prefix%`
    Prefix(&'a str),
    /// `%needle%`
    Contains(&'a str),
}

impl<'a> LikeKind<'a> {
    fn parse(pattern: &'a str) -> Option<Self> {
        if pattern == "%" {
            return Some(LikeKind::Prefix(""));
        }

        // Find first wildcard.
        let first_wild = pattern.find(['%', '_'])?;

        // `_` as first wildcard means we can't handle it.
        if pattern.as_bytes()[first_wild] == b'_' {
            return None;
        }

        // `prefix%` — single trailing %
        if first_wild > 0 && &pattern[first_wild..] == "%" {
            return Some(LikeKind::Prefix(&pattern[..first_wild]));
        }

        // `%needle%` — leading and trailing %, no inner wildcards
        if first_wild == 0
            && pattern.len() > 2
            && pattern.as_bytes()[pattern.len() - 1] == b'%'
            && !pattern[1..pattern.len() - 1].contains(['%', '_'])
        {
            return Some(LikeKind::Contains(&pattern[1..pattern.len() - 1]));
        }

        None
    }
}

// ---------------------------------------------------------------------------
// DFA for prefix matching (LIKE 'prefix%')
// ---------------------------------------------------------------------------

/// Precomputed shift-based DFA for prefix matching on FSST codes.
///
/// States 0..prefix_len track match progress, plus ACCEPT and FAIL.
/// Uses the same shift-based approach as the contains DFA: all state
/// transitions packed into a `u64` per code byte. For prefixes longer
/// than 13 characters, falls back to a fused u8 table.
struct FsstPrefixDfa {
    /// Packed transitions: `(table[code] >> (state * 4)) & 0xF` gives next state.
    transitions: [u64; 256],
    /// Packed escape transitions for literal bytes.
    escape_transitions: [u64; 256],
    accept_state: u8,
    fail_state: u8,
}

impl FsstPrefixDfa {
    const BITS: u32 = 4;
    const MASK: u64 = (1 << Self::BITS) - 1;

    fn new(symbols: &[Symbol], symbol_lengths: &[u8], prefix: &[u8]) -> Self {
        // prefix.len() + 2 states (0..prefix_len, accept, fail) must fit in 4 bits.
        debug_assert!(prefix.len() + 2 <= (1 << Self::BITS));

        let n_symbols = symbols.len();
        let accept_state = prefix.len() as u8;
        let fail_state = prefix.len() as u8 + 1;
        let n_states = prefix.len() + 2;

        // Build per-symbol and per-escape-byte transitions into flat tables.
        let mut sym_trans = vec![fail_state; n_states * n_symbols];
        let mut esc_trans = vec![fail_state; n_states * 256];

        for state in 0..n_states {
            if state as u8 == accept_state {
                for code in 0..n_symbols {
                    sym_trans[state * n_symbols + code] = accept_state;
                }
                for b in 0..256 {
                    esc_trans[state * 256 + b] = accept_state;
                }
                continue;
            }
            if state as u8 == fail_state {
                continue;
            }

            for code in 0..n_symbols {
                let sym = symbols[code].to_u64().to_le_bytes();
                let sym_len = symbol_lengths[code] as usize;
                let remaining = prefix.len() - state;
                let cmp = sym_len.min(remaining);

                if sym[..cmp] == prefix[state..state + cmp] {
                    let next = state + cmp;
                    sym_trans[state * n_symbols + code] = if next >= prefix.len() {
                        accept_state
                    } else {
                        next as u8
                    };
                }
            }

            for b in 0..256usize {
                if b as u8 == prefix[state] {
                    let next = state + 1;
                    esc_trans[state * 256 + b] = if next >= prefix.len() {
                        accept_state
                    } else {
                        next as u8
                    };
                }
            }
        }

        // Fuse symbol transitions into a 256-wide table.
        let escape_sentinel = fail_state + 1;
        let mut fused = vec![fail_state; n_states * 256];
        for state in 0..n_states {
            for code in 0..n_symbols {
                fused[state * 256 + code] = sym_trans[state * n_symbols + code];
            }
            fused[state * 256 + ESCAPE_CODE as usize] = escape_sentinel;
        }

        // Pack into u64 shift tables.
        let mut transitions = [0u64; 256];
        for code_byte in 0..256usize {
            let mut packed = 0u64;
            for state in 0..n_states {
                packed |= (fused[state * 256 + code_byte] as u64) << (state as u32 * Self::BITS);
            }
            transitions[code_byte] = packed;
        }

        let mut escape_transitions = [0u64; 256];
        for byte_val in 0..256usize {
            let mut packed = 0u64;
            for state in 0..n_states {
                packed |= (esc_trans[state * 256 + byte_val] as u64) << (state as u32 * Self::BITS);
            }
            escape_transitions[byte_val] = packed;
        }

        Self {
            transitions,
            escape_transitions,
            accept_state,
            fail_state,
        }
    }

    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        let mut pos = 0;
        while pos < codes.len() {
            let code = codes[pos];
            pos += 1;
            let packed = self.transitions[code as usize];
            let next = ((packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            if next == self.fail_state + 1 {
                // Escape sentinel: read literal byte.
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
            if state == self.accept_state {
                return true;
            }
            if state == self.fail_state {
                return false;
            }
        }
        state == self.accept_state
    }
}

// ---------------------------------------------------------------------------
// DFA for contains matching (LIKE '%needle%')
// ---------------------------------------------------------------------------

/// Contains DFA for long needles (>14 chars). Short needles (len <= 7) are
/// handled by `BranchlessShiftDfa`, medium needles (8-14) by
/// `FlatBranchlessDfa`.
enum FsstContainsDfa {
    /// Shift-based DFA for medium needles (len 8-14).
    Shift(Box<ShiftDfa>),
    /// Fused u8 table DFA for long needles (len > 14).
    Fused(FusedDfa),
}

impl FsstContainsDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        if needle.len() <= ShiftDfa::MAX_NEEDLE_LEN {
            FsstContainsDfa::Shift(Box::new(ShiftDfa::new(symbols, symbol_lengths, needle)))
        } else {
            FsstContainsDfa::Fused(FusedDfa::new(symbols, symbol_lengths, needle))
        }
    }

    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        match self {
            FsstContainsDfa::Shift(dfa) => dfa.matches(codes),
            FsstContainsDfa::Fused(dfa) => dfa.matches(codes),
        }
    }
}

/// Branchless escape-folded DFA for short needles (len <= 7).
///
/// Folds escape handling into the state space so that `matches()` is
/// completely branchless (except for loop control). The state layout is:
/// - States 0..N-1: normal match-progress states
/// - State N: accept (sticky for all inputs)
/// - States N+1..2N: escape states (state `s+N+1` means "was in state `s`,
///   just consumed ESCAPE_CODE")
///
/// Total states: 2N+1. With 4-bit packing, max N=7.
///
/// Uses a decomposed hierarchical lookup that processes 4 code bytes per
/// loop iteration with only ~3 KB of tables:
///
/// 1. **Equivalence class table** (256 B): maps each code byte to a class
///    id. Bytes with identical transition u64s share a class -- typically
///    only ~6-10 classes exist (needle chars + escape + "miss-all").
/// 2. **Pair-compose table** (~N^2 B): maps `(class0, class1)` to a 2-byte
///    palette index. Typically ~36 entries.
/// 3. **4-byte compose table** (~M^2 x 8 B): maps `(palette0, palette1)` to
///    the composed packed u64 for all 4 bytes. Typically ~81 entries = 648 B.
///
/// Each loop iteration: 4 class lookups (parallel, 256 B table) -> 2
/// pair-compose lookups (parallel, ~36 B table) -> 1 compose lookup
/// (~648 B table) -> 1 shift+mask. All tables fit in L1 cache.
struct BranchlessShiftDfa {
    /// Maps each code byte to its equivalence class. Bytes with the same
    /// packed transition u64 share a class. (256 bytes)
    eq_class: [u8; 256],
    /// Maps `(class0 * n_classes + class1)` -> 2-byte palette index.
    pair_compose: Vec<u8>,
    /// Number of equivalence classes (stride for pair_compose).
    n_classes: usize,
    /// Maps `(palette0 * n_palette + palette1)` -> composed packed u64
    /// for 4 bytes.
    compose_4b: Vec<u64>,
    /// Number of unique 2-byte palette entries (stride for compose_4b).
    n_palette: usize,
    /// 1-byte fallback transitions for trailing bytes.
    transitions_1b: [u64; 256],
    /// 2-byte palette for the remainder path (2-3 trailing bytes).
    palette_2b: Vec<u64>,
    accept_state: u8,
}

impl BranchlessShiftDfa {
    const BITS: u32 = 4;
    const MASK: u64 = (1 << Self::BITS) - 1;
    /// Maximum needle length: need 2N+1 states to fit in 16 slots (4 bits).
    /// 2*7+1 = 15 <= 16, so max N = 7.
    const MAX_NEEDLE_LEN: usize = 7;

    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let n = needle.len();
        debug_assert!(n <= Self::MAX_NEEDLE_LEN);

        let accept_state = n as u8;
        let total_states = 2 * n + 1;
        debug_assert!(total_states <= (1 << Self::BITS));

        let transitions_1b =
            Self::build_1b_transitions(symbols, symbol_lengths, needle, total_states);

        // Build equivalence classes: group bytes with identical transition u64.
        let mut eq_class = [0u8; 256];
        let mut class_representatives: Vec<u64> = Vec::new();
        for byte_val in 0..256usize {
            let t = transitions_1b[byte_val];
            let cls = class_representatives
                .iter()
                .position(|&v| v == t)
                .unwrap_or_else(|| {
                    class_representatives.push(t);
                    class_representatives.len() - 1
                });
            eq_class[byte_val] = cls as u8;
        }
        let n_classes = class_representatives.len();

        // Build pair-compose: for each (class0, class1), compose the two
        // 1-byte transitions and deduplicate into a 2-byte palette.
        let (pair_compose, palette_2b) =
            Self::build_pair_compose(&class_representatives, n_classes, total_states);

        // Build 4-byte composition: compose_4b[p0 * n + p1] gives the packed
        // u64 for applying palette_2b[p0] then palette_2b[p1] in sequence.
        let n_palette = palette_2b.len();
        let compose_4b = Self::build_compose_4b(&palette_2b, total_states);

        Self {
            eq_class,
            pair_compose,
            n_classes,
            compose_4b,
            n_palette,
            transitions_1b,
            palette_2b,
            accept_state,
        }
    }

    /// Build the 1-byte packed transition table from FSST symbols and
    /// a byte-level KMP table, folding escape handling into the state space.
    fn build_1b_transitions(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        needle: &[u8],
        total_states: usize,
    ) -> [u64; 256] {
        let n = needle.len();
        let n_symbols = symbols.len();
        let accept_state = n as u8;
        let n_normal_states = n + 1;

        let byte_table = kmp_byte_transitions(needle);

        // Build per-symbol transitions for normal states.
        let mut sym_trans = vec![0u8; n_normal_states * n_symbols];
        for state in 0..n_normal_states {
            for code in 0..n_symbols {
                if state as u8 == accept_state {
                    sym_trans[state * n_symbols + code] = accept_state;
                    continue;
                }
                let sym = symbols[code].to_u64().to_le_bytes();
                let sym_len = symbol_lengths[code] as usize;
                let mut s = state as u16;
                for &b in &sym[..sym_len] {
                    if s == accept_state as u16 {
                        break;
                    }
                    s = byte_table[s as usize * 256 + b as usize];
                }
                sym_trans[state * n_symbols + code] = s as u8;
            }
        }

        // Build fused transition table with escape folding.
        let mut fused = vec![0u8; total_states * 256];
        for code_byte in 0..256usize {
            for s in 0..n {
                if code_byte == ESCAPE_CODE as usize {
                    fused[s * 256 + code_byte] = (s + n + 1) as u8;
                } else if code_byte < n_symbols {
                    fused[s * 256 + code_byte] = sym_trans[s * n_symbols + code_byte];
                }
            }
            fused[n * 256 + code_byte] = accept_state;
            for s in 0..n {
                let esc_state = s + n + 1;
                let next = byte_table[s * 256 + code_byte] as u8;
                fused[esc_state * 256 + code_byte] = next;
            }
        }

        // Pack into u64 shift table.
        let mut transitions = [0u64; 256];
        for code_byte in 0..256usize {
            let mut packed = 0u64;
            for state in 0..total_states {
                packed |= (fused[state * 256 + code_byte] as u64) << (state as u32 * Self::BITS);
            }
            transitions[code_byte] = packed;
        }
        transitions
    }

    /// Build the pair-compose table and 2-byte palette from equivalence
    /// class representatives.
    fn build_pair_compose(
        class_reps: &[u64],
        n_classes: usize,
        total_states: usize,
    ) -> (Vec<u8>, Vec<u64>) {
        let mut pair_compose = vec![0u8; n_classes * n_classes];
        let mut palette_2b: Vec<u64> = Vec::new();

        for c0 in 0..n_classes {
            for c1 in 0..n_classes {
                let t0 = class_reps[c0];
                let t1 = class_reps[c1];
                let mut packed = 0u64;
                for state in 0..total_states {
                    let mid = ((t0 >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
                    let final_s = ((t1 >> (mid as u32 * Self::BITS)) & Self::MASK) as u8;
                    packed |= (final_s as u64) << (state as u32 * Self::BITS);
                }
                let idx = palette_2b
                    .iter()
                    .position(|&v| v == packed)
                    .unwrap_or_else(|| {
                        palette_2b.push(packed);
                        palette_2b.len() - 1
                    });
                pair_compose[c0 * n_classes + c1] = idx as u8;
            }
        }
        (pair_compose, palette_2b)
    }

    /// Compose pairs of 2-byte palette entries into a 4-byte lookup table.
    fn build_compose_4b(palette_2b: &[u64], total_states: usize) -> Vec<u64> {
        let n = palette_2b.len();
        let mut compose = vec![0u64; n * n];
        for p0 in 0..n {
            for p1 in 0..n {
                let mut packed = 0u64;
                for state in 0..total_states {
                    let mid = ((palette_2b[p0] >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
                    let final_s =
                        ((palette_2b[p1] >> (mid as u32 * Self::BITS)) & Self::MASK) as u8;
                    packed |= (final_s as u64) << (state as u32 * Self::BITS);
                }
                compose[p0 * n + p1] = packed;
            }
        }
        compose
    }

    /// Process remaining bytes after the interleaved common prefix.
    #[inline]
    fn finish_tail(&self, mut state: u8, codes: &[u8]) -> u8 {
        let chunks = codes.chunks_exact(4);
        let rem = chunks.remainder();

        for chunk in chunks {
            let ec0 = unsafe { *self.eq_class.get_unchecked(chunk[0] as usize) } as usize;
            let ec1 = unsafe { *self.eq_class.get_unchecked(chunk[1] as usize) } as usize;
            let ec2 = unsafe { *self.eq_class.get_unchecked(chunk[2] as usize) } as usize;
            let ec3 = unsafe { *self.eq_class.get_unchecked(chunk[3] as usize) } as usize;
            let p0 =
                unsafe { *self.pair_compose.get_unchecked(ec0 * self.n_classes + ec1) } as usize;
            let p1 =
                unsafe { *self.pair_compose.get_unchecked(ec2 * self.n_classes + ec3) } as usize;
            let packed = unsafe { *self.compose_4b.get_unchecked(p0 * self.n_palette + p1) };
            state = ((packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
        }

        if rem.len() >= 2 {
            let ec0 = self.eq_class[rem[0] as usize] as usize;
            let ec1 = self.eq_class[rem[1] as usize] as usize;
            let p = self.pair_compose[ec0 * self.n_classes + ec1] as usize;
            let packed = self.palette_2b[p];
            state = ((packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            if rem.len() == 3 {
                let packed = self.transitions_1b[rem[2] as usize];
                state = ((packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            }
        } else if rem.len() == 1 {
            let packed = self.transitions_1b[rem[0] as usize];
            state = ((packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
        }

        state
    }

    /// Branchless matching processing four code bytes per iteration.
    #[inline(never)]
    fn matches(&self, codes: &[u8]) -> bool {
        self.finish_tail(0, codes) == self.accept_state
    }
}

/// Flat u8 escape-folded DFA for medium needles (8-14 chars).
///
/// Like `BranchlessShiftDfa`, folds escape handling into the state space
/// (2N+1 states), but uses a flat `u8` transition table instead of
/// shift-packed `u64`. Supports up to 14-char needles (2*14+1 = 29 states).
/// Table size: 29 * 256 = 7,424 bytes, fits in L1.
struct FlatBranchlessDfa {
    /// transitions[state * 256 + byte] -> next state
    transitions: Vec<u8>,
    accept_state: u8,
}

impl FlatBranchlessDfa {
    const MAX_NEEDLE_LEN: usize = 14;

    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let n = needle.len();
        debug_assert!(n <= Self::MAX_NEEDLE_LEN);

        let accept_state = n as u8;
        let total_states = 2 * n + 1;
        let n_symbols = symbols.len();

        let byte_table = kmp_byte_transitions(needle);

        // Build per-symbol transitions for normal states.
        let mut sym_trans = vec![0u8; (n + 1) * n_symbols];
        for state in 0..=n {
            for code in 0..n_symbols {
                if state as u8 == accept_state {
                    sym_trans[state * n_symbols + code] = accept_state;
                    continue;
                }
                let sym = symbols[code].to_u64().to_le_bytes();
                let sym_len = symbol_lengths[code] as usize;
                let mut s = state as u16;
                for &b in &sym[..sym_len] {
                    if s == accept_state as u16 {
                        break;
                    }
                    s = byte_table[s as usize * 256 + b as usize];
                }
                sym_trans[state * n_symbols + code] = s as u8;
            }
        }

        // Build fused transition table with escape folding.
        let mut transitions = vec![0u8; total_states * 256];
        for code_byte in 0..256usize {
            // Normal states 0..n
            for s in 0..n {
                if code_byte == ESCAPE_CODE as usize {
                    transitions[s * 256 + code_byte] = (s + n + 1) as u8;
                } else if code_byte < n_symbols {
                    transitions[s * 256 + code_byte] = sym_trans[s * n_symbols + code_byte];
                }
            }
            // Accept state (sticky)
            transitions[n * 256 + code_byte] = accept_state;
            // Escape states n+1..2n
            for s in 0..n {
                let esc_state = s + n + 1;
                let next = byte_table[s * 256 + code_byte] as u8;
                transitions[esc_state * 256 + code_byte] = next;
            }
        }

        Self {
            transitions,
            accept_state,
        }
    }

    #[inline(never)]
    fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        for &byte in codes {
            state = self.transitions[state as usize * 256 + byte as usize];
        }
        state == self.accept_state
    }
}

/// Shift-based DFA: packs all state transitions into a `u64` per input byte.
///
/// For a DFA with S states (S <= 16, using 4 bits each), we store transitions
/// for ALL states in one `u64`. Transition: `next = (table[code] >> (state * 4)) & 0xF`.
///
/// Supports needles up to 14 characters (needle.len() + 2 <= 16 to fit escape
/// sentinel). This covers virtually all practical LIKE patterns.
struct ShiftDfa {
    /// For each code byte (0..255): a `u64` packing all state transitions.
    /// Bits `[state*4 .. state*4+4)` encode the next state for that input.
    transitions: [u64; 256],
    /// Same layout for escape byte transitions.
    escape_transitions: [u64; 256],
    accept_state: u8,
    escape_sentinel: u8,
}

impl ShiftDfa {
    const BITS: u32 = 4;
    const MASK: u64 = (1 << Self::BITS) - 1;
    /// Maximum needle length: 2^BITS - 2 (need room for accept + sentinel).
    const MAX_NEEDLE_LEN: usize = (1 << Self::BITS) - 2;

    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        debug_assert!(needle.len() <= Self::MAX_NEEDLE_LEN);

        let n_symbols = symbols.len();
        let n_states = needle.len() + 1;
        let accept_state = needle.len() as u8;
        let escape_sentinel = needle.len() as u8 + 1;

        let byte_table = kmp_byte_transitions(needle);

        // Build per-symbol transitions into a flat table first.
        let mut sym_trans = vec![0u16; n_states * n_symbols];
        for state in 0..n_states {
            for code in 0..n_symbols {
                if state as u8 == accept_state {
                    sym_trans[state * n_symbols + code] = accept_state as u16;
                    continue;
                }
                let sym = symbols[code].to_u64().to_le_bytes();
                let sym_len = symbol_lengths[code] as usize;
                let mut s = state as u16;
                for &b in &sym[..sym_len] {
                    if s == accept_state as u16 {
                        break;
                    }
                    s = byte_table[s as usize * 256 + b as usize];
                }
                sym_trans[state * n_symbols + code] = s;
            }
        }

        // Build fused 256-wide table, then pack into u64 shift tables.
        let mut fused = vec![0u8; n_states * 256];
        for state in 0..n_states {
            for code in 0..n_symbols {
                fused[state * 256 + code] = sym_trans[state * n_symbols + code] as u8;
            }
            fused[state * 256 + ESCAPE_CODE as usize] = escape_sentinel;
        }

        let mut transitions = [0u64; 256];
        for code_byte in 0..256usize {
            let mut packed = 0u64;
            for state in 0..n_states {
                let next = fused[state * 256 + code_byte];
                packed |= (next as u64) << (state as u32 * Self::BITS);
            }
            transitions[code_byte] = packed;
        }

        let mut escape_transitions = [0u64; 256];
        for byte_val in 0..256usize {
            let mut packed = 0u64;
            for state in 0..n_states {
                let next = byte_table[state * 256 + byte_val] as u8;
                packed |= (next as u64) << (state as u32 * Self::BITS);
            }
            escape_transitions[byte_val] = packed;
        }

        Self {
            transitions,
            escape_transitions,
            accept_state,
            escape_sentinel,
        }
    }

    /// Match with iterator-based traversal.
    ///
    /// Using `iter.next()` instead of manual index + bounds check helps the
    /// compiler eliminate redundant bounds checks.
    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        let mut iter = codes.iter();
        while let Some(&code) = iter.next() {
            let packed = self.transitions[code as usize];
            let next = ((packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            if next == self.escape_sentinel {
                let Some(&b) = iter.next() else {
                    return false;
                };
                let esc_packed = self.escape_transitions[b as usize];
                state = ((esc_packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            } else {
                state = next;
            }
        }
        state == self.accept_state
    }
}

/// Fused 256-entry u8 table DFA. Fallback for needles > 14 characters.
struct FusedDfa {
    transitions: Vec<u8>,
    escape_transitions: Vec<u8>,
    accept_state: u8,
    escape_sentinel: u8,
}

impl FusedDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let n_symbols = symbols.len();
        let accept_state = needle.len() as u8;
        let n_states = needle.len() + 1;
        let escape_sentinel = needle.len() as u8 + 1;

        let byte_table = kmp_byte_transitions(needle);

        let mut symbol_transitions = vec![0u16; n_states * n_symbols];
        for state in 0..n_states {
            for code in 0..n_symbols {
                if state as u8 == accept_state {
                    symbol_transitions[state * n_symbols + code] = accept_state as u16;
                    continue;
                }
                let sym = symbols[code].to_u64().to_le_bytes();
                let sym_len = symbol_lengths[code] as usize;
                let mut s = state as u16;
                for &b in &sym[..sym_len] {
                    if s == accept_state as u16 {
                        break;
                    }
                    s = byte_table[s as usize * 256 + b as usize];
                }
                symbol_transitions[state * n_symbols + code] = s;
            }
        }

        let mut transitions = vec![0u8; n_states * 256];
        for state in 0..n_states {
            for code in 0..n_symbols {
                transitions[state * 256 + code] =
                    symbol_transitions[state * n_symbols + code] as u8;
            }
            transitions[state * 256 + ESCAPE_CODE as usize] = escape_sentinel;
        }

        let escape_transitions: Vec<u8> = byte_table.iter().map(|&v| v as u8).collect();

        Self {
            transitions,
            escape_transitions,
            accept_state,
            escape_sentinel,
        }
    }

    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
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
            if state == self.accept_state {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// KMP helpers
// ---------------------------------------------------------------------------

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

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::fns::like::Like;
    use vortex_array::scalar_fn::fns::like::LikeKernel;
    use vortex_array::scalar_fn::fns::like::LikeOptions;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::FSST;
    use crate::FSSTArray;
    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn make_fsst(strings: &[Option<&str>], nullability: Nullability) -> FSSTArray {
        let varbin = VarBinArray::from_iter(strings.iter().copied(), DType::Utf8(nullability));
        let compressor = fsst_train_compressor(&varbin);
        fsst_compress(varbin, &compressor)
    }

    fn run_like(array: FSSTArray, pattern: &str, opts: LikeOptions) -> VortexResult<BoolArray> {
        let len = array.len();
        let arr = array.into_array();
        let pattern = ConstantArray::new(pattern, len).into_array();
        let result = Like
            .try_new_array(len, opts, [arr, pattern])?
            .into_array()
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?;
        Ok(result.into_bool())
    }

    fn like(array: FSSTArray, pattern: &str) -> VortexResult<BoolArray> {
        run_like(array, pattern, LikeOptions::default())
    }

    #[test]
    fn test_like_prefix() -> VortexResult<()> {
        let fsst = make_fsst(
            &[
                Some("http://example.com"),
                Some("http://test.org"),
                Some("ftp://files.net"),
                Some("http://vortex.dev"),
                Some("ssh://server.io"),
            ],
            Nullability::NonNullable,
        );
        let result = like(fsst, "http%")?;
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([true, true, false, true, false])
        );
        Ok(())
    }

    #[test]
    fn test_like_prefix_with_nulls() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("hello"), None, Some("help"), None, Some("goodbye")],
            Nullability::Nullable,
        );
        let result = like(fsst, "hel%")?; // spellchecker:disable-line
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([Some(true), None, Some(true), None, Some(false)])
        );
        Ok(())
    }

    #[test]
    fn test_like_contains() -> VortexResult<()> {
        let fsst = make_fsst(
            &[
                Some("hello world"),
                Some("say hello"),
                Some("goodbye"),
                Some("hellooo"),
            ],
            Nullability::NonNullable,
        );
        let result = like(fsst, "%hello%")?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, true, false, true]));
        Ok(())
    }

    #[test]
    fn test_like_contains_cross_symbol() -> VortexResult<()> {
        let fsst = make_fsst(
            &[
                Some("the quick brown fox jumps over the lazy dog"),
                Some("a short string"),
                Some("the lazy dog sleeps"),
                Some("no match"),
            ],
            Nullability::NonNullable,
        );
        let result = like(fsst, "%lazy dog%")?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, false, true, false]));
        Ok(())
    }

    #[test]
    fn test_not_like_contains() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("foobar_sdf"), Some("sdf_start"), Some("nothing")],
            Nullability::NonNullable,
        );
        let opts = LikeOptions {
            negated: true,
            case_insensitive: false,
        };
        let result = run_like(fsst, "%sdf%", opts)?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([false, false, true]));
        Ok(())
    }

    #[test]
    fn test_like_match_all() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("abc"), Some(""), Some("xyz")],
            Nullability::NonNullable,
        );
        let result = like(fsst, "%")?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, true, true]));
        Ok(())
    }

    /// Call `LikeKernel::like` directly on the FSSTArray and verify it
    /// returns `Some(...)` (i.e. the kernel handles it, rather than
    /// returning `None` which would mean "fall back to decompress").
    #[test]
    fn test_like_prefix_kernel_handles() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("http://a.com"), Some("ftp://b.com")],
            Nullability::NonNullable,
        );
        let pattern = ConstantArray::new("http%", fsst.len()).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let result = <FSST as LikeKernel>::like(&fsst, &pattern, LikeOptions::default(), &mut ctx)?;
        assert!(result.is_some(), "FSST LikeKernel should handle prefix%");
        assert_arrays_eq!(result.unwrap(), BoolArray::from_iter([true, false]));
        Ok(())
    }

    /// Same direct-call check for the contains pattern `%needle%`.
    #[test]
    fn test_like_contains_kernel_handles() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("hello world"), Some("goodbye")],
            Nullability::NonNullable,
        );
        let pattern = ConstantArray::new("%world%", fsst.len()).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let result = <FSST as LikeKernel>::like(&fsst, &pattern, LikeOptions::default(), &mut ctx)?;
        assert!(result.is_some(), "FSST LikeKernel should handle %needle%");
        assert_arrays_eq!(result.unwrap(), BoolArray::from_iter([true, false]));
        Ok(())
    }

    /// Patterns we can't handle should return `None` (fall back).
    #[test]
    fn test_like_kernel_falls_back_for_complex_pattern() -> VortexResult<()> {
        let fsst = make_fsst(&[Some("abc"), Some("def")], Nullability::NonNullable);
        let mut ctx = SESSION.create_execution_ctx();

        // Suffix pattern -- not handled.
        let pattern = ConstantArray::new("%abc", fsst.len()).into_array();
        let result = <FSST as LikeKernel>::like(&fsst, &pattern, LikeOptions::default(), &mut ctx)?;
        assert!(result.is_none(), "suffix pattern should fall back");

        // Underscore wildcard -- not handled.
        let pattern = ConstantArray::new("a_c", fsst.len()).into_array();
        let result = <FSST as LikeKernel>::like(&fsst, &pattern, LikeOptions::default(), &mut ctx)?;
        assert!(result.is_none(), "underscore pattern should fall back");

        // Case-insensitive -- not handled.
        let pattern = ConstantArray::new("abc%", fsst.len()).into_array();
        let opts = LikeOptions {
            negated: false,
            case_insensitive: true,
        };
        let result = <FSST as LikeKernel>::like(&fsst, &pattern, opts, &mut ctx)?;
        assert!(result.is_none(), "ilike should fall back");

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Fuzz tests: compare FSST kernel against naive string matching
    // -----------------------------------------------------------------------

    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn random_string(rng: &mut StdRng, max_len: usize) -> String {
        let len = rng.random_range(0..=max_len);
        // Use a small alphabet to increase substring hit rate.
        (0..len)
            .map(|_| (b'a' + rng.random_range(0..6u8)) as char)
            .collect()
    }

    fn fuzz_contains(seed: u64, needle_len: usize, n_strings: usize) -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(seed);

        let needle: String = (0..needle_len)
            .map(|_| (b'a' + rng.random_range(0..6u8)) as char)
            .collect();

        let owned: Vec<String> = (0..n_strings)
            .map(|_| random_string(&mut rng, 80))
            .collect();
        let strings: Vec<Option<&str>> = owned.iter().map(|s| Some(s.as_str())).collect();

        let expected: Vec<bool> = owned.iter().map(|s| s.contains(&needle)).collect();

        let fsst = make_fsst(&strings, Nullability::NonNullable);
        let pattern = format!("%{needle}%");
        let result = run_like(fsst, &pattern, LikeOptions::default())?;

        let got: Vec<bool> = (0..n_strings)
            .map(|i| result.to_bit_buffer().value(i))
            .collect();

        for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                e, g,
                "mismatch at index {i}: string={:?}, needle={needle:?}, expected={e}, got={g}",
                &owned[i],
            );
        }
        Ok(())
    }

    fn fuzz_prefix(seed: u64, prefix_len: usize, n_strings: usize) -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(seed);

        let prefix: String = (0..prefix_len)
            .map(|_| (b'a' + rng.random_range(0..6u8)) as char)
            .collect();

        let owned: Vec<String> = (0..n_strings)
            .map(|_| random_string(&mut rng, 80))
            .collect();
        let strings: Vec<Option<&str>> = owned.iter().map(|s| Some(s.as_str())).collect();

        let expected: Vec<bool> = owned.iter().map(|s| s.starts_with(&prefix)).collect();

        let fsst = make_fsst(&strings, Nullability::NonNullable);
        let pattern = format!("{prefix}%");
        let result = run_like(fsst, &pattern, LikeOptions::default())?;

        let got: Vec<bool> = (0..n_strings)
            .map(|i| result.to_bit_buffer().value(i))
            .collect();

        for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                e, g,
                "mismatch at index {i}: string={:?}, prefix={prefix:?}, expected={e}, got={g}",
                &owned[i],
            );
        }
        Ok(())
    }

    /// Fuzz contains with short needles (1-7 chars) -> BranchlessShiftDfa
    #[test]
    fn fuzz_contains_short_needle() -> VortexResult<()> {
        for seed in 0..50 {
            for needle_len in 1..=7 {
                fuzz_contains(seed, needle_len, 200)?;
            }
        }
        Ok(())
    }

    /// Fuzz contains with medium needles (8-14 chars) -> FlatBranchlessDfa
    #[test]
    fn fuzz_contains_medium_needle() -> VortexResult<()> {
        for seed in 0..50 {
            for needle_len in [8, 10, 14] {
                fuzz_contains(seed, needle_len, 200)?;
            }
        }
        Ok(())
    }

    /// Fuzz contains with long needles (>14 chars) -> FsstContainsDfa
    #[test]
    fn fuzz_contains_long_needle() -> VortexResult<()> {
        for seed in 0..30 {
            for needle_len in [15, 20, 30] {
                fuzz_contains(seed, needle_len, 200)?;
            }
        }
        Ok(())
    }

    /// Fuzz prefix matching
    #[test]
    fn fuzz_prefix_matching() -> VortexResult<()> {
        for seed in 0..50 {
            for prefix_len in [1, 3, 5, 10, 13] {
                fuzz_prefix(seed, prefix_len, 200)?;
            }
        }
        Ok(())
    }
}
