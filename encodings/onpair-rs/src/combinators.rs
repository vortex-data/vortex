// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Bitmap combinators: AND / OR / NOT over LSB-first packed selection
// vectors. Lets callers compose any predicates this crate produces — or
// any external predicate that uses the same packing — into arbitrary
// boolean expressions.
//
// These operate on the bitmap *result* of each predicate rather than on a
// composed automaton. For the perf-critical compressed-domain composition
// (one scan over the token stream evaluating `A AND NOT B` in lockstep),
// see the `TokenAutomaton` traits in `onpair_cpp` — not yet ported.

/// Length in bytes of a bitmap holding `n_rows` bits.
#[inline]
pub fn bitmap_len(n_rows: usize) -> usize {
    n_rows.div_ceil(8)
}

/// Trailing-bit mask for the final byte of an `n_rows` bitmap so we don't
/// flip the unused high bits during NOT.
#[inline]
fn tail_mask(n_rows: usize) -> u8 {
    match n_rows % 8 {
        0 => 0xFF,
        r => (1u8 << r) - 1,
    }
}

/// `a AND b`. Inputs must have the same length; panics otherwise.
pub fn bitmap_and(a: &[u8], b: &[u8]) -> Vec<u8> {
    assert_eq!(a.len(), b.len(), "bitmap_and: length mismatch");
    a.iter().zip(b).map(|(x, y)| x & y).collect()
}

/// `a OR b`. Inputs must have the same length; panics otherwise.
pub fn bitmap_or(a: &[u8], b: &[u8]) -> Vec<u8> {
    assert_eq!(a.len(), b.len(), "bitmap_or: length mismatch");
    a.iter().zip(b).map(|(x, y)| x | y).collect()
}

/// `NOT a`, treating exactly `n_rows` bits as the valid domain. Bits past
/// `n_rows` are zeroed so subsequent AND/OR operations don't see junk.
pub fn bitmap_not(a: &[u8], n_rows: usize) -> Vec<u8> {
    assert_eq!(a.len(), bitmap_len(n_rows), "bitmap_not: length mismatch for n_rows");
    let mut out: Vec<u8> = a.iter().map(|x| !x).collect();
    if let Some(last) = out.last_mut() {
        *last &= tail_mask(n_rows);
    }
    out
}

/// In-place variants are sometimes preferable to avoid an allocation.
pub fn bitmap_and_in_place(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "bitmap_and_in_place: length mismatch");
    for (d, s) in dst.iter_mut().zip(src) {
        *d &= s;
    }
}

pub fn bitmap_or_in_place(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "bitmap_or_in_place: length mismatch");
    for (d, s) in dst.iter_mut().zip(src) {
        *d |= s;
    }
}

pub fn bitmap_not_in_place(bits: &mut [u8], n_rows: usize) {
    assert_eq!(bits.len(), bitmap_len(n_rows), "bitmap_not_in_place: length mismatch");
    for b in bits.iter_mut() {
        *b = !*b;
    }
    if let Some(last) = bits.last_mut() {
        *last &= tail_mask(n_rows);
    }
}

/// Count the set bits in an LSB-packed bitmap of exactly `n_rows` valid
/// bits (ignores any padding bits in the final byte).
pub fn bitmap_popcount(a: &[u8], n_rows: usize) -> usize {
    assert_eq!(a.len(), bitmap_len(n_rows), "bitmap_popcount: length mismatch");
    if a.is_empty() {
        return 0;
    }
    let full = a[..a.len() - 1].iter().map(|b| b.count_ones() as usize).sum::<usize>();
    let tail = (a[a.len() - 1] & tail_mask(n_rows)).count_ones() as usize;
    full + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── bitmap_len ────────────────────────────────────────────────────────

    #[test]
    fn bitmap_len_rounds_up() {
        for n in [0usize, 1, 7, 8, 9, 15, 16, 100] {
            assert_eq!(bitmap_len(n), n.div_ceil(8), "n={n}");
        }
    }

    // ── AND ───────────────────────────────────────────────────────────────

    #[test]
    fn and_basic() {
        let a = vec![0b1100_1010];
        let b = vec![0b1010_1100];
        assert_eq!(bitmap_and(&a, &b), vec![0b1000_1000]);
    }

    #[test]
    fn and_with_all_zeros_is_zero() {
        let a = vec![0xFF, 0xFF];
        let z = vec![0x00, 0x00];
        assert_eq!(bitmap_and(&a, &z), vec![0, 0]);
    }

    #[test]
    fn and_with_all_ones_is_self() {
        let a = vec![0xAB, 0xCD];
        let o = vec![0xFF, 0xFF];
        assert_eq!(bitmap_and(&a, &o), a);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn and_panics_on_length_mismatch() {
        let _unused = bitmap_and(&[0u8], &[0u8, 0u8]);
    }

    // ── OR ────────────────────────────────────────────────────────────────

    #[test]
    fn or_basic() {
        let a = vec![0b1100_0000];
        let b = vec![0b0000_0011];
        assert_eq!(bitmap_or(&a, &b), vec![0b1100_0011]);
    }

    #[test]
    fn or_with_all_zeros_is_self() {
        let a = vec![0xAB, 0xCD];
        let z = vec![0x00, 0x00];
        assert_eq!(bitmap_or(&a, &z), a);
    }

    #[test]
    fn or_with_all_ones_is_ones() {
        let a = vec![0xAB, 0xCD];
        let o = vec![0xFF, 0xFF];
        assert_eq!(bitmap_or(&a, &o), o);
    }

    // ── NOT (mask-respecting) ─────────────────────────────────────────────

    #[test]
    fn not_clears_padding_bits() {
        // n_rows = 3 → tail_mask = 0b0000_0111
        let a = vec![0b0000_0010];
        let n = bitmap_not(&a, 3);
        // !0b0000_0010 = 0b1111_1101, masked to low 3 bits → 0b0000_0101
        assert_eq!(n, vec![0b0000_0101]);
    }

    #[test]
    fn not_full_byte_boundary() {
        // n_rows = 8 → tail_mask = 0xFF
        let a = vec![0b1010_0101];
        assert_eq!(bitmap_not(&a, 8), vec![0b0101_1010]);
    }

    #[test]
    fn not_multi_byte() {
        // n_rows = 12 → tail_mask = 0x0F
        let a = vec![0xAA, 0x05];
        // !0xAA = 0x55; !0x05 = 0xFA, masked to low 4 bits → 0x0A
        assert_eq!(bitmap_not(&a, 12), vec![0x55, 0x0A]);
    }

    #[test]
    fn not_then_and_eq_set_difference() {
        // (A AND NOT B) is the set difference A \ B.
        let a = vec![0b1111_0000];
        let b = vec![0b1100_0000];
        let not_b = bitmap_not(&b, 8);
        let diff = bitmap_and(&a, &not_b);
        assert_eq!(diff, vec![0b0011_0000]);
    }

    // ── In-place ──────────────────────────────────────────────────────────

    #[test]
    fn in_place_and_matches_out_of_place() {
        let a = vec![0xAB, 0xCD];
        let b = vec![0x0F, 0xF0];
        let mut a_mut = a.clone();
        bitmap_and_in_place(&mut a_mut, &b);
        assert_eq!(a_mut, bitmap_and(&a, &b));
    }

    #[test]
    fn in_place_or_matches_out_of_place() {
        let a = vec![0xAB, 0xCD];
        let b = vec![0x0F, 0xF0];
        let mut a_mut = a.clone();
        bitmap_or_in_place(&mut a_mut, &b);
        assert_eq!(a_mut, bitmap_or(&a, &b));
    }

    #[test]
    fn in_place_not_matches_out_of_place() {
        let a = vec![0xAB, 0x05];
        let n = 12;
        let mut a_mut = a.clone();
        bitmap_not_in_place(&mut a_mut, n);
        assert_eq!(a_mut, bitmap_not(&a, n));
    }

    // ── popcount ──────────────────────────────────────────────────────────

    #[test]
    fn popcount_basic() {
        assert_eq!(bitmap_popcount(&[0b0000_1011], 8), 3);
        assert_eq!(bitmap_popcount(&[0xFF, 0xFF], 16), 16);
        // n_rows=12 → bits 12..16 must be ignored.
        assert_eq!(bitmap_popcount(&[0xFF, 0xFF], 12), 12);
        assert_eq!(bitmap_popcount(&[0xFF, 0x0F], 12), 12);
        assert_eq!(bitmap_popcount(&[0xFF, 0xF0], 12), 8);
    }

    #[test]
    fn popcount_empty() {
        assert_eq!(bitmap_popcount(&[], 0), 0);
    }
}
