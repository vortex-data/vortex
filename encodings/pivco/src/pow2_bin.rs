// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `Pow2-Bin + Raw-Bits` outer value bijection (`150` bounded symbols + trailing raw bits).
//!
//! Maps every `u64` value bijectively to `(sym, ebits, raw)`:
//! - `0..32` -> `sym = u`, `ebits = 0`, `raw = 0`
//! - `>= 32` -> `k = floor(log2(u))`, `ebits = k - 1`, two MSBs select even/odd symbol in
//!   `32..150`, and the low `ebits` bits form the verbatim `raw` tail.

#![expect(clippy::cast_possible_truncation)]

use vortex_array::dtype::PType;

/// Maximum `Pow2-Bin` alphabet size (`0..=149`).
pub const MAX_SYMS: usize = 150;

/// Base unsigned value for each symbol `0..150` in the `Pow2-Bin + Raw-Bits` bijection.
pub(crate) static BASE_TABLE: [u64; MAX_SYMS] = build_base_table();

/// Number of verbatim trailing raw bits (`0` or `4..=62`) for each symbol `0..150`.
pub static EBITS_TABLE: [u8; MAX_SYMS] = build_ebits_table();

const fn build_base_table() -> [u64; MAX_SYMS] {
    let mut table = [0u64; MAX_SYMS];
    let mut sym = 0usize;
    while sym < 32 {
        table[sym] = sym as u64;
        sym += 1;
    }
    while sym < MAX_SYMS {
        let rel = (sym - 32) as u8;
        let k = 5 + (rel >> 1);
        let ebits = k - 1;
        let msb2 = 2 + (rel & 1);
        table[sym] = (msb2 as u64) << ebits;
        sym += 1;
    }
    table
}

const fn build_ebits_table() -> [u8; MAX_SYMS] {
    let mut table = [0u8; MAX_SYMS];
    let mut sym = 32usize;
    while sym < MAX_SYMS {
        let rel = (sym - 32) as u8;
        let k = 5 + (rel >> 1);
        table[sym] = k - 1;
        sym += 1;
    }
    table
}

/// Decomposes `u` into `(sym, ebits, raw)` under the `Pow2-Bin + Raw-Bits` bijection.
#[inline]
pub(crate) const fn encode_pow2_bin(u: u64) -> (u8, u8, u64) {
    if u < 32 {
        (u as u8, 0, 0)
    } else {
        let k = (63 - u.leading_zeros()) as u8;
        let ebits = k - 1;
        let msb2 = ((u >> ebits) & 3) as u8;
        let sym = 32 + ((k - 5) << 1) + (msb2 - 2);
        let raw = u & ((1u64 << ebits) - 1);
        (sym, ebits, raw)
    }
}

/// Returns `(sym as usize, ebits)` for `u` without masking `raw`, used by fast histogram estimators.
#[inline]
pub const fn pow2_sym_and_ebits(u: u64) -> (usize, u8) {
    if u < 32 {
        (u as usize, 0)
    } else {
        let k = (63 - u.leading_zeros()) as u8;
        let ebits = k - 1;
        let msb2 = ((u >> ebits) & 3) as u8;
        let sym = 32 + ((k - 5) << 1) + (msb2 - 2);
        (sym as usize, ebits)
    }
}

/// Returns the largest `u64` value representable by `Pow2-Bin` symbol `sym` (`0..150`).
#[inline]
pub(crate) const fn sym_max_value(sym: u8) -> u64 {
    let idx = sym as usize;
    let eb = EBITS_TABLE[idx];
    let mask = if eb == 0 { 0 } else { (1u64 << eb) - 1 };
    BASE_TABLE[idx] | mask
}

/// Returns the maximum non-negative integer value representable by integer [`PType`], or `None`
/// for non-integer types.
#[inline]
pub(crate) const fn max_nonneg_u64(ptype: PType) -> Option<u64> {
    match ptype {
        PType::U8 => Some(u8::MAX as u64),
        PType::U16 => Some(u16::MAX as u64),
        PType::U32 => Some(u32::MAX as u64),
        PType::U64 => Some(u64::MAX),
        PType::I8 => Some(i8::MAX as u64),
        PType::I16 => Some(i16::MAX as u64),
        PType::I32 => Some(i32::MAX as u64),
        PType::I64 => Some(i64::MAX as u64),
        _ => None,
    }
}
