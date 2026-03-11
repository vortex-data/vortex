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

use crate::FSSTArray;
use crate::FSSTVTable;

impl LikeKernel for FSSTVTable {
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
                let dfa = FsstPrefixDfa::new(symbols.as_slice(), symbol_lengths.as_slice(), prefix);
                match_each_integer_ptype!(offsets.ptype(), |T| {
                    let off = offsets.as_slice::<T>();
                    dfa_scan_to_bitbuf(n, off, all_bytes, negated, |codes| dfa.matches(codes))
                })
            }
            LikeKind::Contains(needle) => {
                let needle = needle.as_bytes();
                let dfa =
                    FsstContainsDfa::new(symbols.as_slice(), symbol_lengths.as_slice(), needle);
                match_each_integer_ptype!(offsets.ptype(), |T| {
                    let off = offsets.as_slice::<T>();
                    dfa_scan_to_bitbuf(n, off, all_bytes, negated, |codes| dfa.matches(codes))
                })
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
        for bit in 0..64 {
            let i = base + bit;
            let start: usize = offsets[i].as_();
            let end: usize = offsets[i + 1].as_();
            word |= ((matcher(&all_bytes[start..end]) != negated) as u64) << bit;
        }
        // SAFETY: we allocated capacity for n.div_ceil(64) words.
        unsafe { words.push_unchecked(word) };
    }

    if remainder != 0 {
        let base = n_words * 64;
        let mut word = 0u64;
        for bit in 0..remainder {
            let i = base + bit;
            let start: usize = offsets[i].as_();
            let end: usize = offsets[i + 1].as_();
            word |= ((matcher(&all_bytes[start..end]) != negated) as u64) << bit;
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

/// Precomputed KMP-based DFA for substring matching on FSST codes.
///
/// Uses a shift-based DFA that packs all state transitions into a `u64` per
/// code byte. The table load depends only on the code byte (not on the current
/// state), breaking the load-use dependency chain that makes traditional
/// table-lookup DFAs slow (~4 cycle L1 latency per transition). With the
/// shift-based approach, the table value can be loaded while the previous
/// transition's shift is executing.
///
/// For needles longer than [`ShiftDfa::MAX_NEEDLE_LEN`], falls back to a
/// fused 256-entry u8 table.
enum FsstContainsDfa {
    Shift(Box<ShiftDfa>),
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

    /// Match without per-iteration early-exit. The accept state is sticky
    /// (transitions to itself), so final state == accept means we matched.
    /// Removing the branch from the hot loop improves throughput.
    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
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

    use crate::FSSTArray;
    use crate::FSSTVTable;
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
        let result = run_like(fsst, "http%", LikeOptions::default())?;
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
        let result = run_like(fsst, "hel%", LikeOptions::default())?;
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
        let result = run_like(fsst, "%hello%", LikeOptions::default())?;
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
        let result = run_like(fsst, "%lazy dog%", LikeOptions::default())?;
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
        let result = run_like(fsst, "%", LikeOptions::default())?;
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

        let result =
            <FSSTVTable as LikeKernel>::like(&fsst, &pattern, LikeOptions::default(), &mut ctx)?;
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

        let result =
            <FSSTVTable as LikeKernel>::like(&fsst, &pattern, LikeOptions::default(), &mut ctx)?;
        assert!(result.is_some(), "FSST LikeKernel should handle %needle%");
        assert_arrays_eq!(result.unwrap(), BoolArray::from_iter([true, false]));
        Ok(())
    }

    /// Patterns we can't handle should return `None` (fall back).
    #[test]
    fn test_like_kernel_falls_back_for_complex_pattern() -> VortexResult<()> {
        let fsst = make_fsst(&[Some("abc"), Some("def")], Nullability::NonNullable);
        let mut ctx = SESSION.create_execution_ctx();

        // Suffix pattern — not handled.
        let pattern = ConstantArray::new("%abc", fsst.len()).into_array();
        let result =
            <FSSTVTable as LikeKernel>::like(&fsst, &pattern, LikeOptions::default(), &mut ctx)?;
        assert!(result.is_none(), "suffix pattern should fall back");

        // Underscore wildcard — not handled.
        let pattern = ConstantArray::new("a_c", fsst.len()).into_array();
        let result =
            <FSSTVTable as LikeKernel>::like(&fsst, &pattern, LikeOptions::default(), &mut ctx)?;
        assert!(result.is_none(), "underscore pattern should fall back");

        // Case-insensitive — not handled.
        let pattern = ConstantArray::new("abc%", fsst.len()).into_array();
        let opts = LikeOptions {
            negated: false,
            case_insensitive: true,
        };
        let result = <FSSTVTable as LikeKernel>::like(&fsst, &pattern, opts, &mut ctx)?;
        assert!(result.is_none(), "ilike should fall back");

        Ok(())
    }
}
