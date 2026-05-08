// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Hyperscan-style "shufti" 16-byte-at-a-time classifier.
//!
//! For a given DFA state, a byte code `c` is "interesting" (will change the state
//! or is an escape) iff `(low_nibble[c & 0xF] & high_nibble[c >> 4]) != 0`.
//!
//! The masks are 16-byte lookup tables — one indexed by low nibble, one by high nibble.
//! We compute them using two `PSHUFB` instructions (SSSE3), then AND the results;
//! any non-zero byte means an interesting code was found.
//!
//! This generalises the state-0 [`super::skip::SkipStrategy`] to ALL DFA states.

use fsst::ESCAPE_CODE;

/// Precomputed shufti nibble masks for a single DFA state.
///
/// For a code byte `c`:
/// - interesting iff `(lo[c & 0xF] & hi[c >> 4]) != 0`
#[derive(Clone, Debug)]
pub(super) struct ShuftiMask {
    /// 16-byte table indexed by `c & 0xF`.
    pub(super) lo: [u8; 16],
    /// 16-byte table indexed by `c >> 4`.
    pub(super) hi: [u8; 16],
}

impl ShuftiMask {
    /// Build a `ShuftiMask` for the given DFA transition row.
    ///
    /// A code `c` is interesting if `transition_row[c] != start_state` or `c == ESCAPE_CODE`.
    pub(super) fn from_transition_row(transition_row: &[u8], start_state: u8) -> Self {
        debug_assert!(transition_row.len() >= 256);

        // We use a single bit (bit 0) to mark interesting codes.
        // lo[c & 0xF] has bit 0 set for some nibble pattern,
        // hi[c >> 4] has bit 0 set for the corresponding high nibble.
        // We need: for each interesting code c, lo[c & 0xF] & hi[c >> 4] != 0.
        //
        // The simplest encoding: lo[nib] = mask of high nibbles that combine with `nib`
        // to form an interesting code. hi[high_nib] = 1 if any code in that high-nibble
        // group is interesting, paired with lo.
        //
        // We use the Hyperscan approach: partition codes by (hi4, lo4). For each lo4 nibble,
        // lo[lo4] has a bit set for each hi4 group that contains an interesting code.
        // hi[hi4] has that same bit set. So the AND is non-zero iff (hi4, lo4) = an interesting code.

        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];

        for code in 0u16..256 {
            let c = code as u8;
            let interesting =
                transition_row[usize::from(c)] != start_state || c == ESCAPE_CODE;
            if interesting {
                let lo_nib = usize::from(c & 0xF);
                let hi_nib = usize::from(c >> 4);
                // Use bit (hi_nib % 8) in byte hi_nib/8 — but since we only have 1 byte
                // per entry, we use a different scheme: use hi_nib as the bit index in lo,
                // and set hi[hi_nib] to signal "this high nibble participates".
                //
                // Concretely: we use 8 bits, treating hi_nib as a bit index (hi_nib can be
                // 0..15, so we need 2 bytes to cover all 16 high nibbles — but our tables
                // are only 1 byte wide per entry). Instead we use the "2-bit group" scheme:
                // encode lo[lo4] |= (1 << (hi4 % 8)), hi[hi4] |= (1 << (hi4 % 8)).
                // This can have false positives for pairs (lo4_a, hi4_a) and (lo4_b, hi4_b)
                // where hi4_a % 8 == hi4_b % 8, but the shufti skip is speculative anyway —
                // any false positive just means we process a code that leaves state unchanged,
                // which is correct (just slightly slow).
                let bit = 1u8 << (hi_nib % 8);
                lo[lo_nib] |= bit;
                hi[hi_nib] |= bit;
            }
        }

        Self { lo, hi }
    }

    /// Scalar fallback: find the next interesting byte in `codes[start..]`.
    #[inline]
    pub(super) fn find_next_scalar(&self, codes: &[u8], start: usize) -> Option<usize> {
        for (i, &c) in codes[start..].iter().enumerate() {
            if self.lo[usize::from(c & 0xF)] & self.hi[usize::from(c >> 4)] != 0 {
                return Some(start + i);
            }
        }
        None
    }

    /// SIMD-accelerated search: find the first interesting byte in `codes[start..]`.
    ///
    /// Uses SSSE3 `PSHUFB` to classify 16 bytes per iteration.
    ///
    /// # Safety
    ///
    /// Caller must ensure SSSE3 is available (checked via `is_x86_feature_detected!`).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "ssse3")]
    pub(super) unsafe fn find_next_ssse3(&self, codes: &[u8], start: usize) -> Option<usize> {
        use std::arch::x86_64::{
            __m128i, _mm_and_si128, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8,
            _mm_set1_epi8, _mm_shuffle_epi8, _mm_setzero_si128,
        };

        let slice = &codes[start..];
        let len = slice.len();
        let ptr = slice.as_ptr();

        // SAFETY: SSSE3 available per #[target_feature]; pointer arithmetic within bounds below.
        unsafe {
            let lo_vec = _mm_loadu_si128(self.lo.as_ptr().cast::<__m128i>());
            let hi_vec = _mm_loadu_si128(self.hi.as_ptr().cast::<__m128i>());
            let zero = _mm_setzero_si128();
            let lo_mask = _mm_set1_epi8(0x0F_u8 as i8);

            let mut i = 0usize;
            // Process 16 bytes at a time.
            while i + 16 <= len {
                let chunk = _mm_loadu_si128(ptr.add(i).cast::<__m128i>());
                // Low nibble of each byte
                let lo_idx = _mm_and_si128(chunk, lo_mask);
                // High nibble of each byte (shift right 4)
                let hi_idx = _mm_and_si128(
                    // logical shift: use _mm_srli_epi16 then mask — easier: cast trick
                    {
                        // _mm_srli_epi16 shifts 16-bit lanes; we want logical byte shift
                        // Use: (chunk >> 4) & 0x0F via the shuffle of hi nibble bits
                        // Actually: shuffle with hi nibble as index = chunk >> 4
                        // We compute (chunk >> 4) by shifting the 16-bit integers right by 4
                        // and masking off the top bits that bleed across byte boundaries.
                        let shifted =
                            std::arch::x86_64::_mm_srli_epi16(chunk, 4);
                        _mm_and_si128(shifted, lo_mask)
                    },
                    lo_mask,
                );
                // Shuffle: lo_result[i] = lo_vec[lo_idx[i]]
                let lo_result = _mm_shuffle_epi8(lo_vec, lo_idx);
                // Shuffle: hi_result[i] = hi_vec[hi_idx[i]]
                let hi_result = _mm_shuffle_epi8(hi_vec, hi_idx);
                // AND: non-zero byte = interesting code
                let matched = _mm_and_si128(lo_result, hi_result);
                // Check if any byte is non-zero
                let cmp = _mm_cmpeq_epi8(matched, zero);
                let mask = _mm_movemask_epi8(cmp) as u32;
                // mask has a 0 bit where matched[i] != 0 (i.e., interesting)
                let interesting_mask = (!mask) & 0xFFFF;
                if interesting_mask != 0 {
                    let first = interesting_mask.trailing_zeros() as usize;
                    return Some(start + i + first);
                }
                i += 16;
            }
            // Scalar tail for remaining bytes.
            self.find_next_scalar(codes, start + i)
        }
    }

    /// Find the next interesting byte, using SSSE3 when available.
    #[inline]
    pub(super) fn find_next(&self, codes: &[u8], start: usize) -> Option<usize> {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("ssse3") {
            // SAFETY: feature check just performed.
            return unsafe { self.find_next_ssse3(codes, start) };
        }
        self.find_next_scalar(codes, start)
    }
}

#[cfg(test)]
mod tests {
    use fsst::ESCAPE_CODE;

    use super::ShuftiMask;

    fn make_mask(interesting: &[u8]) -> ShuftiMask {
        // Build a fake 256-entry transition row where all listed codes are "interesting"
        // (i.e., transition to state != 0).
        let mut row = [0u8; 256];
        for &c in interesting {
            row[usize::from(c)] = 1; // != 0 = interesting
        }
        ShuftiMask::from_transition_row(&row, 0)
    }

    #[test]
    fn test_shufti_scalar_finds_interesting() {
        let mask = make_mask(&[b'a', b'z', ESCAPE_CODE]);
        let codes = b"\x00\x00\x00a\x00\x00";
        assert_eq!(mask.find_next_scalar(codes, 0), Some(3));
        let codes2 = b"\x00\x00\x00\x00\x00\x00";
        assert_eq!(mask.find_next_scalar(codes2, 0), None);
    }

    #[test]
    fn test_shufti_escape_always_interesting() {
        // Even if ESCAPE_CODE's transition == start_state, it must be flagged interesting.
        let mut row = [0u8; 256];
        // Explicitly make ESCAPE_CODE stay at state 0 in the row (it won't — from_transition_row
        // always marks it interesting regardless).
        row[usize::from(ESCAPE_CODE)] = 0;
        let mask = ShuftiMask::from_transition_row(&row, 0);
        // ESCAPE_CODE must still be found.
        let mut codes = vec![0u8; 20];
        codes[5] = ESCAPE_CODE;
        assert_eq!(mask.find_next_scalar(&codes, 0), Some(5));
    }

    #[test]
    fn test_shufti_no_interesting() {
        let row = [0u8; 256]; // all codes stay at state 0, and escape doesn't appear
        // But ESCAPE_CODE is still always interesting per spec.
        let mask = ShuftiMask::from_transition_row(&row, 0);
        let codes = vec![0u8; 32]; // no ESCAPE_CODE
        // Only ESCAPE_CODE (255) is interesting, but it's not in codes[].
        // So None — unless our code marks some other code interesting by accident.
        // Note: codes[i] = 0 here, and code 0 is not interesting (row[0]=0=start_state=0).
        assert_eq!(mask.find_next_scalar(&codes, 0), None);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_shufti_simd_matches_scalar() {
        if !is_x86_feature_detected!("ssse3") {
            return;
        }
        let interesting_codes: Vec<u8> = vec![b'a', b'b', 0x10, 0x20, ESCAPE_CODE];
        let mask = make_mask(&interesting_codes);

        // Build a test vector with interesting codes at known positions.
        let mut codes = vec![0u8; 48];
        codes[3] = b'a';
        codes[17] = b'b';
        codes[35] = 0x10;
        codes[47] = ESCAPE_CODE;

        for start in 0..codes.len() {
            let scalar = mask.find_next_scalar(&codes, start);
            let simd = unsafe { mask.find_next_ssse3(&codes, start) };
            assert_eq!(
                scalar, simd,
                "mismatch at start={start}: scalar={scalar:?}, simd={simd:?}"
            );
        }
    }
}
