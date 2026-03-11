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
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
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
                    BitBufferMut::collect_bool(n, |i| {
                        let start = off[i] as usize;
                        let end = off[i + 1] as usize;
                        dfa.matches(&all_bytes[start..end]) != negated
                    })
                    .freeze()
                })
            }
            LikeKind::Contains(needle) => {
                let needle = needle.as_bytes();
                let matcher =
                    build_contains_matcher(symbols.as_slice(), symbol_lengths.as_slice(), needle);
                match_each_integer_ptype!(offsets.ptype(), |T| {
                    let off = offsets.as_slice::<T>();
                    BitBufferMut::collect_bool(n, |i| {
                        let start = off[i] as usize;
                        let end = off[i + 1] as usize;
                        matcher.matches(&all_bytes[start..end]) != negated
                    })
                    .freeze()
                })
            }
        };

        let validity = Validity::copy_from_array(&array.clone().into_array())?
            .union_nullability(pattern_scalar.dtype().nullability());

        Ok(Some(BoolArray::new(result, validity).into_array()))
    }
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

/// Precomputed DFA for prefix matching on FSST codes.
///
/// States 0..prefix_len track match progress, plus ACCEPT and FAIL.
/// One table lookup per FSST code — no per-byte inner loop.
struct FsstPrefixDfa {
    symbol_transitions: Vec<u16>,
    escape_transitions: Vec<u16>,
    n_symbols: usize,
    accept_state: u16,
    fail_state: u16,
}

impl FsstPrefixDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], prefix: &[u8]) -> Self {
        let n_symbols = symbols.len();
        let accept_state = prefix.len() as u16;
        let fail_state = prefix.len() as u16 + 1;
        let n_states = prefix.len() + 2;

        let mut symbol_transitions = vec![fail_state; n_states * n_symbols];
        let mut escape_transitions = vec![fail_state; n_states * 256];

        for state in 0..n_states {
            if state as u16 == accept_state {
                for code in 0..n_symbols {
                    symbol_transitions[state * n_symbols + code] = accept_state;
                }
                for b in 0..256 {
                    escape_transitions[state * 256 + b] = accept_state;
                }
                continue;
            }
            if state as u16 == fail_state {
                continue;
            }

            for code in 0..n_symbols {
                let sym = symbols[code].to_u64().to_le_bytes();
                let sym_len = symbol_lengths[code] as usize;
                let remaining = prefix.len() - state;
                let cmp = sym_len.min(remaining);

                if sym[..cmp] == prefix[state..state + cmp] {
                    let next = state + cmp;
                    symbol_transitions[state * n_symbols + code] = if next >= prefix.len() {
                        accept_state
                    } else {
                        next as u16
                    };
                }
            }

            for b in 0..256usize {
                if b as u8 == prefix[state] {
                    let next = state + 1;
                    escape_transitions[state * 256 + b] = if next >= prefix.len() {
                        accept_state
                    } else {
                        next as u16
                    };
                }
            }
        }

        Self {
            symbol_transitions,
            escape_transitions,
            n_symbols,
            accept_state,
            fail_state,
        }
    }

    fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u16;
        let mut pos = 0;
        while pos < codes.len() {
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
                debug_assert!((code as usize) < self.n_symbols);
                state = self.symbol_transitions[state as usize * self.n_symbols + code as usize];
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

/// Maximum needle length for the shift-based DFA (4 bits per state = 16 states,
/// minus 1 for the accept state and 1 for the escape sentinel).
const SHIFT_DFA_MAX_NEEDLE_LEN: usize = 14;

/// Trait for contains-matching on FSST-encoded data.
trait FsstContainsMatcher {
    fn matches(&self, codes: &[u8]) -> bool;
}

/// Shift-based DFA for substring matching on FSST codes.
///
/// Packs all state transitions into a `[u64; 256]` array. The table load
/// depends only on the input byte (not on the current state), breaking the
/// load-use dependency chain that makes traditional table-lookup DFAs slow.
/// The state-dependent work is just a shift + mask on a register value.
///
/// Supports needles up to 14 bytes (4 bits per state, 16 states max).
struct FsstContainsShiftDfa {
    /// For each code byte (0..255): a u64 packing all state transitions.
    /// Bits `[state*4 .. state*4+4)` encode the next state for that input.
    transitions: [u64; 256],
    /// Same layout for escape byte transitions.
    escape_transitions: [u64; 256],
    accept_state: u8,
    escape_sentinel: u8,
}

impl FsstContainsShiftDfa {
    const BITS: u32 = 4;
    const MASK: u64 = (1 << Self::BITS) - 1;

    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        debug_assert!(needle.len() + 2 <= (1 << Self::BITS));

        let n_symbols = symbols.len();
        let accept_state = needle.len() as u16;
        let n_states = needle.len() + 1;
        let escape_sentinel = n_states as u16 + 1;

        let byte_table = kmp_byte_transitions(needle);

        // Build the fused 256-wide symbol transition table.
        let mut fused_transitions = vec![0u16; n_states * 256];
        for state in 0..n_states {
            // Pre-fill symbol codes.
            for code in 0..n_symbols {
                if state as u16 == accept_state {
                    fused_transitions[state * 256 + code] = accept_state;
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
                fused_transitions[state * 256 + code] = s;
            }
            // Mark escape code with sentinel.
            fused_transitions[state * 256 + ESCAPE_CODE as usize] = escape_sentinel;
        }

        let escape_sentinel_u8 = escape_sentinel as u8;

        // Pack into shift tables.
        let mut transitions = [0u64; 256];
        let mut escape_transitions_packed = [0u64; 256];

        for code_byte in 0..256usize {
            let mut packed = 0u64;
            for state in 0..n_states {
                let next = fused_transitions[state * 256 + code_byte];
                let next_u8 = if next == escape_sentinel {
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
                let next = byte_table[state * 256 + byte_val] as u8;
                packed |= (next as u64) << (state as u32 * Self::BITS);
            }
            escape_transitions_packed[byte_val] = packed;
        }

        Self {
            transitions,
            escape_transitions: escape_transitions_packed,
            accept_state: accept_state as u8,
            escape_sentinel: escape_sentinel_u8,
        }
    }
}

impl FsstContainsMatcher for FsstContainsShiftDfa {
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
            if state == self.accept_state {
                return true;
            }
        }
        state == self.accept_state
    }
}

/// Fallback split-table DFA for needles longer than [`SHIFT_DFA_MAX_NEEDLE_LEN`].
///
/// Uses separate symbol and escape transition tables with `u16` states.
struct FsstContainsSplitDfa {
    symbol_transitions: Vec<u16>,
    escape_transitions: Vec<u16>,
    n_symbols: usize,
    accept_state: u16,
}

impl FsstContainsSplitDfa {
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
}

impl FsstContainsMatcher for FsstContainsSplitDfa {
    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u16;
        let mut pos = 0;
        while pos < codes.len() {
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
                debug_assert!((code as usize) < self.n_symbols);
                state = self.symbol_transitions[state as usize * self.n_symbols + code as usize];
            }
            if state == self.accept_state {
                return true;
            }
        }
        false
    }
}

/// Build the best available contains matcher for the given needle length.
fn build_contains_matcher(
    symbols: &[Symbol],
    symbol_lengths: &[u8],
    needle: &[u8],
) -> Box<dyn FsstContainsMatcher> {
    if needle.len() <= SHIFT_DFA_MAX_NEEDLE_LEN {
        Box::new(FsstContainsShiftDfa::new(symbols, symbol_lengths, needle))
    } else {
        Box::new(FsstContainsSplitDfa::new(symbols, symbol_lengths, needle))
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
