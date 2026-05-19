// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Same-dict `compare_fused` for OnPair-encoded rows.
//!
//! Compares two rows in string-lex order while keeping the fast path in
//! u16-token-id space. Falls back to byte-level comparison only when tokens
//! differ, and only for as many tokens as needed to resolve the order.

use std::cmp::Ordering;

/// Compare two rows of OnPair token IDs in **string-lex order** of their
/// decoded bytes, assuming both rows share the dictionary described by
/// `dict_bytes` + `dict_offsets`.
#[inline]
pub fn compare_fused(
    a: &[u16],
    b: &[u16],
    dict_bytes: &[u8],
    dict_offsets: &[u32],
) -> Ordering {
    // Phase 1: scan equal tokens. This loop auto-vectorises to a u16
    // SIMD compare on x86_64 / aarch64 under LLVM's vectoriser.
    let common = a.len().min(b.len());
    let mut i = 0;
    while i < common && a[i] == b[i] {
        i += 1;
    }
    if i == common {
        return a.len().cmp(&b.len());
    }

    // Phase 2: tokens differ at position i — resolve via byte streams.
    let mut ca = ByteCursor::new(&a[i..], dict_bytes, dict_offsets);
    let mut cb = ByteCursor::new(&b[i..], dict_bytes, dict_offsets);
    loop {
        let ra = ca.peek();
        let rb = cb.peek();
        if ra.is_empty() && rb.is_empty() {
            return Ordering::Equal;
        }
        if ra.is_empty() {
            return Ordering::Less;
        }
        if rb.is_empty() {
            return Ordering::Greater;
        }
        let n = ra.len().min(rb.len());
        match ra[..n].cmp(&rb[..n]) {
            Ordering::Equal => {
                ca.advance(n);
                cb.advance(n);
            }
            ord => return ord,
        }
    }
}

struct ByteCursor<'a> {
    tokens: &'a [u16],
    dict_bytes: &'a [u8],
    dict_offsets: &'a [u32],
    token_idx: usize,
    byte_off: usize,
}

impl<'a> ByteCursor<'a> {
    #[inline]
    fn new(tokens: &'a [u16], dict_bytes: &'a [u8], dict_offsets: &'a [u32]) -> Self {
        Self {
            tokens,
            dict_bytes,
            dict_offsets,
            token_idx: 0,
            byte_off: 0,
        }
    }

    #[inline]
    fn peek(&self) -> &'a [u8] {
        if self.token_idx >= self.tokens.len() {
            return &[];
        }
        let tok = self.tokens[self.token_idx] as usize;
        let start = self.dict_offsets[tok] as usize;
        let end = self.dict_offsets[tok + 1] as usize;
        &self.dict_bytes[start + self.byte_off..end]
    }

    #[inline]
    fn advance(&mut self, n: usize) {
        self.byte_off += n;
        if self.token_idx < self.tokens.len() {
            let tok = self.tokens[self.token_idx] as usize;
            let tok_len = (self.dict_offsets[tok + 1] - self.dict_offsets[tok]) as usize;
            debug_assert!(self.byte_off <= tok_len);
            if self.byte_off >= tok_len {
                self.token_idx += 1;
                self.byte_off = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::onpair_compress;

    fn parts_of(
        out: &crate::encoders::OnPairOut,
    ) -> (Vec<u8>, Vec<u32>) {
        let p = out.col.parts().unwrap();
        (p.dict_bytes.to_vec(), p.dict_offsets.to_vec())
    }

    #[test]
    fn matches_byte_order_random() {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut rows: Vec<Vec<u8>> = (0..2000)
            .map(|i| format!("row-{:06}-tail-{}", i, i * 7 % 999).into_bytes())
            .collect();
        rows.shuffle(&mut rng);
        let out = onpair_compress(&rows, 12).unwrap();
        let (dict_bytes, dict_offsets) = parts_of(&out);

        for i in 0..rows.len() {
            for j in 0..rows.len() {
                let want = rows[i].cmp(&rows[j]);
                let got = compare_fused(
                    &out.tokens[i],
                    &out.tokens[j],
                    &dict_bytes,
                    &dict_offsets,
                );
                assert_eq!(want, got, "mismatch at i={i} j={j}");
            }
        }
    }

    #[test]
    fn boundary_prefix_case() {
        // Construct rows that share a long prefix and diverge mid-string; one
        // strictly extends the other (forces phase-2 boundary roll-over).
        // Lex order: "abc" < "abc_extended"; both share prefix "abc".
        let rows: Vec<Vec<u8>> = vec![
            b"the_quick_brown_fox".to_vec(),
            b"the_quick_brown_fox_jumps_over".to_vec(),
            b"the_quick_brown_dog".to_vec(),
        ];
        let out = onpair_compress(&rows, 12).unwrap();
        let (dict_bytes, dict_offsets) = parts_of(&out);
        // "the_quick_brown_dog" < "the_quick_brown_fox" (d < f) < extension
        assert_eq!(
            compare_fused(&out.tokens[2], &out.tokens[0], &dict_bytes, &dict_offsets),
            Ordering::Less
        );
        assert_eq!(
            compare_fused(&out.tokens[0], &out.tokens[1], &dict_bytes, &dict_offsets),
            Ordering::Less,
            "strict prefix should compare less"
        );
        assert_eq!(
            compare_fused(&out.tokens[1], &out.tokens[0], &dict_bytes, &dict_offsets),
            Ordering::Greater
        );
        assert_eq!(
            compare_fused(&out.tokens[0], &out.tokens[0], &dict_bytes, &dict_offsets),
            Ordering::Equal
        );
    }
}
