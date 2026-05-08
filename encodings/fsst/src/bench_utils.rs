// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Low-level benchmarking helpers for DFA variants.
//!
//! Exposes raw DFA scanning functions so benchmark binaries can call matchers
//! directly, bypassing the LikeKernel framework overhead.

#![expect(clippy::unwrap_used)]

use fsst::Symbol;
#[expect(deprecated)]
use vortex_array::ToCanonical;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_array::match_each_integer_ptype;

use crate::FSSTArray;
use crate::array::FSSTArrayExt;
use crate::dfa::FlatContainsDfa;
use crate::dfa::FlatContainsDfaBaseline;
use crate::dfa::FlatContainsDfaClasses;
use crate::dfa::FlatContainsDfaClassesPre;
use crate::dfa::FlatContainsDfaEscapeFolded;
use crate::dfa::dfa_scan_to_bitbuf;
use vortex_buffer::BitBuffer;
use vortex_array::dtype::IntegerPType;
#[cfg(feature = "shufti-counters")]
use crate::dfa::{SHUFTI_CODES_SKIPPED, SHUFTI_SKIP_CALLS, SHUFTI_SKIP_FIRED};

/// Reset the shufti skip counters to zero (no-op if counters are not enabled).
#[cfg(feature = "shufti-counters")]
pub fn reset_shufti_counters() {
    SHUFTI_SKIP_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
    SHUFTI_SKIP_FIRED.store(0, std::sync::atomic::Ordering::Relaxed);
    SHUFTI_CODES_SKIPPED.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Read (calls, fired, codes_skipped) from the shufti skip counters.
#[cfg(feature = "shufti-counters")]
pub fn read_shufti_counters() -> (u64, u64, u64) {
    (
        SHUFTI_SKIP_CALLS.load(std::sync::atomic::Ordering::Relaxed),
        SHUFTI_SKIP_FIRED.load(std::sync::atomic::Ordering::Relaxed),
        SHUFTI_CODES_SKIPPED.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Scan all strings in `fsst` with `FlatContainsDfaBaseline` for `needle`.
///
/// Returns the count of set bits in the result bitmask (to prevent dead-code
/// elimination and for a meaningful check value).
pub fn scan_baseline_contains(fsst: &FSSTArray, needle: &[u8]) -> usize {
    let symbols: Vec<Symbol> = fsst.symbols().as_slice().to_vec();
    let symbol_lengths: Vec<u8> = fsst.symbol_lengths().as_slice().to_vec();
    let dfa = FlatContainsDfaBaseline::new(&symbols, &symbol_lengths, needle).unwrap();

    let codes = fsst.codes();
    #[expect(deprecated)]
    let offsets = codes.offsets().to_primitive();
    let all_bytes = codes.bytes();
    let n = codes.len();

    let result = match_each_integer_ptype!(offsets.ptype(), |T| {
        dfa_scan_to_bitbuf(n, offsets.as_slice::<T>(), all_bytes.as_slice(), false, |c| {
            dfa.matches(c)
        })
    });
    result.true_count()
}

/// Scan all strings in `fsst` with `FlatContainsDfaClasses` (variant B) for `needle`.
///
/// Returns the count of set bits in the result bitmask.
pub fn scan_classes_contains(fsst: &FSSTArray, needle: &[u8]) -> usize {
    let symbols: Vec<Symbol> = fsst.symbols().as_slice().to_vec();
    let symbol_lengths: Vec<u8> = fsst.symbol_lengths().as_slice().to_vec();
    let dfa = FlatContainsDfaClasses::new(&symbols, &symbol_lengths, needle).unwrap();

    let codes = fsst.codes();
    #[expect(deprecated)]
    let offsets = codes.offsets().to_primitive();
    let all_bytes = codes.bytes();
    let n = codes.len();

    let result = match_each_integer_ptype!(offsets.ptype(), |T| {
        dfa_scan_to_bitbuf(n, offsets.as_slice::<T>(), all_bytes.as_slice(), false, |c| {
            dfa.matches(c)
        })
    });
    result.true_count()
}

/// Build a `FlatContainsDfaClasses` and return its `n_classes`, used by the
/// reporting binary to characterize how aggressively the table compressed.
pub fn classes_n_classes(fsst: &FSSTArray, needle: &[u8]) -> u16 {
    let symbols: Vec<Symbol> = fsst.symbols().as_slice().to_vec();
    let symbol_lengths: Vec<u8> = fsst.symbol_lengths().as_slice().to_vec();
    let dfa = FlatContainsDfaClasses::new(&symbols, &symbol_lengths, needle).unwrap();
    dfa.n_classes()
}

/// Scan all strings in `fsst` with `FlatContainsDfaClassesPre` (variant C) for `needle`.
///
/// Pre-classifies the entire `all_bytes` buffer once before the per-string
/// DFA scan reads from it. The classification cost is amortized across all
/// strings sharing the buffer.
pub fn scan_pre_classified_contains(fsst: &FSSTArray, needle: &[u8]) -> usize {
    let symbols: Vec<Symbol> = fsst.symbols().as_slice().to_vec();
    let symbol_lengths: Vec<u8> = fsst.symbol_lengths().as_slice().to_vec();
    let dfa = FlatContainsDfaClassesPre::new(&symbols, &symbol_lengths, needle).unwrap();

    let codes = fsst.codes();
    #[expect(deprecated)]
    let offsets = codes.offsets().to_primitive();
    let all_bytes_buf = codes.bytes();
    let all_bytes = all_bytes_buf.as_slice();
    let n = codes.len();

    let classified = dfa.classify_bulk(all_bytes);

    let result: BitBuffer = match_each_integer_ptype!(offsets.ptype(), |T| {
        scan_pre_to_bitbuf::<T>(n, offsets.as_slice::<T>(), &classified, all_bytes, &dfa)
    });
    result.true_count()
}

fn scan_pre_to_bitbuf<T>(
    n: usize,
    offsets: &[T],
    classified: &[u8],
    all_bytes: &[u8],
    dfa: &FlatContainsDfaClassesPre,
) -> BitBuffer
where
    T: IntegerPType,
{
    let mut start: usize = offsets[0].as_();
    BitBuffer::collect_bool(n, |i| {
        let end: usize = offsets[i + 1].as_();
        let result = dfa.matches_pre(&classified[start..end], &all_bytes[start..end]);
        start = end;
        result
    })
}

/// Scan all strings in `fsst` with `FlatContainsDfaEscapeFolded` (variant E)
/// for `needle`. Limited to needles ≤127 bytes.
pub fn scan_escape_folded_contains(fsst: &FSSTArray, needle: &[u8]) -> usize {
    let symbols: Vec<Symbol> = fsst.symbols().as_slice().to_vec();
    let symbol_lengths: Vec<u8> = fsst.symbol_lengths().as_slice().to_vec();
    let dfa = FlatContainsDfaEscapeFolded::new(&symbols, &symbol_lengths, needle).unwrap();

    let codes = fsst.codes();
    #[expect(deprecated)]
    let offsets = codes.offsets().to_primitive();
    let all_bytes = codes.bytes();
    let n = codes.len();

    let result = match_each_integer_ptype!(offsets.ptype(), |T| {
        dfa_scan_to_bitbuf(n, offsets.as_slice::<T>(), all_bytes.as_slice(), false, |c| {
            dfa.matches(c)
        })
    });
    result.true_count()
}

/// Build an escape-folded DFA and return its total state count `2N + 1`.
pub fn escape_folded_n_states(fsst: &FSSTArray, needle: &[u8]) -> u16 {
    let symbols: Vec<Symbol> = fsst.symbols().as_slice().to_vec();
    let symbol_lengths: Vec<u8> = fsst.symbol_lengths().as_slice().to_vec();
    let dfa = FlatContainsDfaEscapeFolded::new(&symbols, &symbol_lengths, needle).unwrap();
    dfa.n_states()
}

/// Variant D: decompress each FSST string and run `memchr::memmem::Finder` (the
/// same literal matcher `regex-automata` delegates to for literal patterns).
///
/// This is the "delegation" baseline: no FSST-code-level DFA at all; use a
/// gold-standard byte-level substring matcher on the decoded text. It tests
/// whether the FSST-code-level DFA pushdown is worth the complexity vs simply
/// decompressing per row and using an off-the-shelf matcher.
pub fn scan_decompress_memmem_contains(fsst: &FSSTArray, needle: &[u8]) -> usize {
    let symbols = fsst.symbols();
    let symbols = symbols.as_slice();
    let symbol_lengths = fsst.symbol_lengths();
    let symbol_lengths = symbol_lengths.as_slice();

    let n_symbols = symbols.len();
    let mut expansions = vec![0u8; n_symbols * 8];
    let mut exp_lens = vec![0u8; n_symbols];
    for (i, (sym, &len)) in symbols.iter().zip(symbol_lengths.iter()).enumerate() {
        let bytes = sym.to_u64().to_le_bytes();
        let len_usize = usize::from(len);
        expansions[i * 8..i * 8 + len_usize].copy_from_slice(&bytes[..len_usize]);
        exp_lens[i] = len;
    }

    let codes = fsst.codes();
    #[expect(deprecated)]
    let offsets = codes.offsets().to_primitive();
    let all_bytes_buf = codes.bytes();
    let all_bytes = all_bytes_buf.as_slice();
    let n = codes.len();

    let finder = memchr::memmem::Finder::new(needle);

    let result: BitBuffer = match_each_integer_ptype!(offsets.ptype(), |T| {
        scan_decompress_memmem_inner::<T>(
            n,
            offsets.as_slice::<T>(),
            all_bytes,
            &expansions,
            &exp_lens,
            &finder,
        )
    });
    result.true_count()
}

fn scan_decompress_memmem_inner<T>(
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    expansions: &[u8],
    exp_lens: &[u8],
    finder: &memchr::memmem::Finder<'_>,
) -> BitBuffer
where
    T: IntegerPType,
{
    use fsst::ESCAPE_CODE;

    let mut start: usize = offsets[0].as_();
    let mut decoded: Vec<u8> = Vec::with_capacity(256);
    BitBuffer::collect_bool(n, |i| {
        let end: usize = offsets[i + 1].as_();
        let s = &all_bytes[start..end];
        start = end;

        decoded.clear();
        let mut p = 0usize;
        while p < s.len() {
            let c = s[p];
            p += 1;
            if c == ESCAPE_CODE {
                if p < s.len() {
                    decoded.push(s[p]);
                    p += 1;
                }
            } else {
                let cu = usize::from(c);
                if cu < exp_lens.len() {
                    let len = usize::from(exp_lens[cu]);
                    decoded.extend_from_slice(&expansions[cu * 8..cu * 8 + len]);
                }
            }
        }
        finder.find(&decoded).is_some()
    })
}

/// Scan all strings in `fsst` with the shufti `FlatContainsDfa` for `needle`.
///
/// Returns the count of set bits in the result bitmask.
pub fn scan_shufti_contains(fsst: &FSSTArray, needle: &[u8]) -> usize {
    let symbols: Vec<Symbol> = fsst.symbols().as_slice().to_vec();
    let symbol_lengths: Vec<u8> = fsst.symbol_lengths().as_slice().to_vec();
    let dfa = FlatContainsDfa::new(&symbols, &symbol_lengths, needle).unwrap();

    let codes = fsst.codes();
    #[expect(deprecated)]
    let offsets = codes.offsets().to_primitive();
    let all_bytes = codes.bytes();
    let n = codes.len();

    let result = match_each_integer_ptype!(offsets.ptype(), |T| {
        dfa_scan_to_bitbuf(n, offsets.as_slice::<T>(), all_bytes.as_slice(), false, |c| {
            dfa.matches(c)
        })
    });
    result.true_count()
}
