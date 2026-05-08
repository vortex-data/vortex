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
use crate::dfa::dfa_scan_to_bitbuf;
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
