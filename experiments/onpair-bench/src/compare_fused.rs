// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Same-dict `compare_fused` for OnPair-encoded rows.
//!
//! Compares two rows in string-lex order while keeping the fast path in
//! u16-token-id space. Falls back to byte-level comparison only when tokens
//! differ, and only for as many tokens as needed to resolve the order.

use std::cmp::Ordering;

/// Per-token packed `(offset << 16) | length`. Mirror of what `onpair-rs`
/// builds internally as `Column::dict_table`. One indexed `u64` load per
/// token instead of two `u32` loads from `dict_offsets`.
pub fn build_dict_table(dict_offsets: &[u32]) -> Vec<u64> {
    let n = dict_offsets.len() - 1;
    let mut t = Vec::with_capacity(n);
    for i in 0..n {
        let off = dict_offsets[i] as u64;
        let len = (dict_offsets[i + 1] - dict_offsets[i]) as u64;
        debug_assert!(len <= 16);
        t.push((off << 16) | len);
    }
    t
}

/// Per-token packed first-8-bytes-BE (zero-padded for shorter tokens).
/// Numerically comparing two such `u64`s gives the correct lex order on
/// the first 8 bytes, with zero-padding standing in for "string ends here"
/// **provided tokens contain no internal null bytes** (true for UTF-8 text
/// and the URL/Title/l_comment datasets we test).
///
/// If two tokens' packed `u64`s are equal, both tokens have ≥ 8 bytes and
/// match on the first 8; the comparator falls to the slow path for bytes 8+.
pub fn build_token_prefix(dict_bytes: &[u8], dict_offsets: &[u32]) -> Vec<u64> {
    let n = dict_offsets.len() - 1;
    let mut t = Vec::with_capacity(n);
    for i in 0..n {
        let off = dict_offsets[i] as usize;
        let len = (dict_offsets[i + 1] - dict_offsets[i]) as usize;
        let take = len.min(8);
        let mut buf = [0u8; 8];
        buf[..take].copy_from_slice(&dict_bytes[off..off + take]);
        t.push(u64::from_be_bytes(buf));
    }
    t
}

/// Compare two rows of OnPair token IDs in **string-lex order** of their
/// decoded bytes (v1 — slice-cmp Phase 2). Kept for benchmarking.
#[inline]
pub fn compare_fused(
    a: &[u16],
    b: &[u16],
    dict_bytes: &[u8],
    dict_table: &[u64],
) -> Ordering {
    // Phase 1: scan equal tokens. SAFETY: we bound by `common`, so reads are
    // always in-range. The unchecked indexing lets LLVM vectorise the loop.
    let common = a.len().min(b.len());
    let mut i = 0;
    unsafe {
        while i < common && *a.get_unchecked(i) == *b.get_unchecked(i) {
            i += 1;
        }
    }
    if i == common {
        return a.len().cmp(&b.len());
    }

    // Phase 2 — first divergence (the overwhelmingly common case).
    //
    // We resolve here without a cursor abstraction. One packed `u64` load
    // per side gives us (offset, length); we slice into `dict_bytes` and
    // call the specialised `<[u8]>::cmp` which dispatches to memcmp.
    //
    // The slow cursor path is only entered when the diverging tokens are
    // in a prefix relationship (one's bytes are a strict prefix of the
    // other's). LPM training makes that rare.
    unsafe {
        let ta = *a.get_unchecked(i) as usize;
        let tb = *b.get_unchecked(i) as usize;
        let ea = *dict_table.get_unchecked(ta);
        let eb = *dict_table.get_unchecked(tb);
        let off_a = (ea >> 16) as usize;
        let len_a = (ea & 0xffff) as usize;
        let off_b = (eb >> 16) as usize;
        let len_b = (eb & 0xffff) as usize;
        let n = len_a.min(len_b);
        let sa = dict_bytes.get_unchecked(off_a..off_a + len_a);
        let sb = dict_bytes.get_unchecked(off_b..off_b + len_b);
        match sa[..n].cmp(&sb[..n]) {
            Ordering::Equal => {
                // Prefix relationship — rare. Fall to slow path.
                compare_fused_slow(a, b, i, n, dict_bytes, dict_table)
            }
            ord => ord,
        }
    }
}

/// Precompute a u64 "row prefix" per row = first 8 decoded bytes BE-packed.
/// Sort comparators can resolve random-pair comparisons in a single
/// `u64::cmp` against this prefix, skipping all dict access.
pub fn build_row_prefix(
    token_flat: &[u16],
    token_bd: &[u32],
    dict_bytes: &[u8],
    dict_table: &[u64],
) -> Vec<u64> {
    let n = token_bd.len() - 1;
    let mut out = Vec::with_capacity(n);
    for r in 0..n {
        let toks = &token_flat[token_bd[r] as usize..token_bd[r + 1] as usize];
        let mut buf = [0u8; 8];
        let mut filled = 0;
        for &tok in toks {
            if filled == 8 {
                break;
            }
            let entry = dict_table[tok as usize];
            let off = (entry >> 16) as usize;
            let len = (entry & 0xffff) as usize;
            let take = (8 - filled).min(len);
            buf[filled..filled + take].copy_from_slice(&dict_bytes[off..off + take]);
            filled += take;
        }
        out.push(u64::from_be_bytes(buf));
    }
    out
}

/// V3: Use a precomputed row-prefix u64 to resolve random-pair comparisons
/// without touching the dict at all. Falls to v1 for the rare tie case.
#[inline]
pub fn compare_fused_v3(
    a: &[u16],
    b: &[u16],
    row_prefix_a: u64,
    row_prefix_b: u64,
    row_len_a: usize,
    row_len_b: usize,
    dict_bytes: &[u8],
    dict_table: &[u64],
) -> Ordering {
    // Resolve from row prefix where possible. The trick: zero-padding in
    // the row prefix is safe only if the differing byte position is within
    // both rows' actual length.
    if row_prefix_a != row_prefix_b {
        let xor = row_prefix_a ^ row_prefix_b;
        let k = (xor.leading_zeros() / 8) as usize;
        if k < row_len_a && k < row_len_b {
            return row_prefix_a.cmp(&row_prefix_b);
        }
        // Difference in zero-padded territory: one row is shorter and
        // matches the other's prefix → shorter < longer.
        return row_len_a.cmp(&row_len_b);
    }
    // First 8 bytes of decoded rows match. Fall to full token compare.
    compare_fused(a, b, dict_bytes, dict_table)
}

/// V2: Same as `compare_fused`, but resolves the first Phase 2 divergence
/// from a packed u64 token prefix (first 7 bytes BE + length in low byte).
/// The vast majority of comparisons resolve in a single `u64::cmp`.
#[inline]
pub fn compare_fused_v2(
    a: &[u16],
    b: &[u16],
    dict_bytes: &[u8],
    dict_table: &[u64],
    token_prefix: &[u64],
) -> Ordering {
    // Phase 1: scan equal tokens.
    let common = a.len().min(b.len());
    let mut i = 0;
    unsafe {
        while i < common && *a.get_unchecked(i) == *b.get_unchecked(i) {
            i += 1;
        }
    }
    if i == common {
        return a.len().cmp(&b.len());
    }

    // Phase 2 fast path: one indexed u64 load + one u64 compare per side.
    //
    // The first-8-byte u64 packs each token's content left-aligned, zero-
    // padded. Zero-padding correctly represents string termination only if
    // the two tokens AGREE on the position of the difference being within
    // both tokens' real content — otherwise the zero is fictitious and the
    // real next byte comes from the NEXT token of that row.
    unsafe {
        let ta = *a.get_unchecked(i) as usize;
        let tb = *b.get_unchecked(i) as usize;
        let pa = *token_prefix.get_unchecked(ta);
        let pb = *token_prefix.get_unchecked(tb);
        let ea = *dict_table.get_unchecked(ta);
        let eb = *dict_table.get_unchecked(tb);
        let len_a = (ea & 0xffff) as usize;
        let len_b = (eb & 0xffff) as usize;
        if pa != pb {
            let xor = pa ^ pb;
            let k = (xor.leading_zeros() / 8) as usize;
            if k < len_a && k < len_b {
                // First differing byte is real content in both. Fast win.
                return pa.cmp(&pb);
            }
            // Difference was in zero-padded territory. Shorter token's
            // content matched the longer's prefix; resume after the shorter
            // ends so the cursor pulls bytes from the next token.
            let n = len_a.min(len_b);
            return compare_fused_slow(a, b, i, n, dict_bytes, dict_table);
        }
        // First 8 bytes (incl. padding) equal. For string data without
        // internal nulls this implies len_a >= 8 and len_b >= 8 and the
        // first 8 bytes truly match. Slow path resumes at byte 8.
        compare_fused_slow(a, b, i, 8, dict_bytes, dict_table)
    }
}

/// Slow path: the diverging tokens at position `i` matched on the first
/// `n_consumed` bytes. Continue comparing through subsequent tokens until we
/// can decide.
#[cold]
#[inline(never)]
fn compare_fused_slow(
    a: &[u16],
    b: &[u16],
    i: usize,
    n_consumed: usize,
    dict_bytes: &[u8],
    dict_table: &[u64],
) -> Ordering {
    let mut ca = ByteCursor::new(&a[i..], dict_bytes, dict_table);
    let mut cb = ByteCursor::new(&b[i..], dict_bytes, dict_table);
    ca.advance(n_consumed);
    cb.advance(n_consumed);
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
    dict_table: &'a [u64],
    token_idx: usize,
    byte_off: usize,
}

impl<'a> ByteCursor<'a> {
    #[inline]
    fn new(tokens: &'a [u16], dict_bytes: &'a [u8], dict_table: &'a [u64]) -> Self {
        Self {
            tokens,
            dict_bytes,
            dict_table,
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
        let entry = self.dict_table[tok];
        let off = (entry >> 16) as usize;
        let len = (entry & 0xffff) as usize;
        &self.dict_bytes[off + self.byte_off..off + len]
    }

    #[inline]
    fn advance(&mut self, mut n: usize) {
        while n > 0 && self.token_idx < self.tokens.len() {
            let tok = self.tokens[self.token_idx] as usize;
            let tok_len = (self.dict_table[tok] & 0xffff) as usize;
            let remaining = tok_len - self.byte_off;
            if n < remaining {
                self.byte_off += n;
                return;
            }
            n -= remaining;
            self.token_idx += 1;
            self.byte_off = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::onpair_compress;

    fn p_offsets_at(out: &crate::encoders::OnPairOut, tok: usize) -> &[u8] {
        let p = out.col.parts().unwrap();
        let s = p.dict_offsets[tok] as usize;
        let e = p.dict_offsets[tok + 1] as usize;
        // SAFETY: lifetime is tied to dict_bytes' container which outlives this call within test.
        let bytes: *const u8 = p.dict_bytes.as_ptr();
        unsafe { std::slice::from_raw_parts(bytes.add(s), e - s) }
    }

    fn parts_of(
        out: &crate::encoders::OnPairOut,
    ) -> (Vec<u8>, Vec<u64>, Vec<u64>) {
        let p = out.col.parts().unwrap();
        let table = build_dict_table(p.dict_offsets);
        let prefix = build_token_prefix(p.dict_bytes, p.dict_offsets);
        (p.dict_bytes.to_vec(), table, prefix)
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
        let (dict_bytes, dict_table, token_prefix) = parts_of(&out);

        for i in 0..rows.len() {
            for j in 0..rows.len() {
                let want = rows[i].cmp(&rows[j]);
                let got = compare_fused(
                    &out.tokens[i],
                    &out.tokens[j],
                    &dict_bytes,
                    &dict_table,
                );
                assert_eq!(want, got, "v1 mismatch at i={i} j={j}");
                let got2 = compare_fused_v2(
                    &out.tokens[i],
                    &out.tokens[j],
                    &dict_bytes,
                    &dict_table,
                    &token_prefix,
                );
                if want != got2 {
                    eprintln!("v2 mismatch i={i} j={j}");
                    eprintln!("  row[i] = {:?}", std::str::from_utf8(&rows[i]).unwrap());
                    eprintln!("  row[j] = {:?}", std::str::from_utf8(&rows[j]).unwrap());
                    eprintln!("  tokens[i] = {:?}", out.tokens[i]);
                    eprintln!("  tokens[j] = {:?}", out.tokens[j]);
                    for &t in &out.tokens[i] {
                        let s = p_offsets_at(&out, t as usize);
                        eprintln!("    a tok {t} = {:?}", std::str::from_utf8(s).unwrap_or("<bin>"));
                    }
                    for &t in &out.tokens[j] {
                        let s = p_offsets_at(&out, t as usize);
                        eprintln!("    b tok {t} = {:?}", std::str::from_utf8(s).unwrap_or("<bin>"));
                    }
                    eprintln!("  want = {:?}, got = {:?}", want, got2);
                    panic!("v2 mismatch");
                }
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
        let (dict_bytes, dict_table, token_prefix) = parts_of(&out);
        // "the_quick_brown_dog" < "the_quick_brown_fox" (d < f) < extension
        let cmp_fns: [Box<dyn Fn(&[u16], &[u16]) -> Ordering>; 2] = [
            Box::new(|a, b| compare_fused(a, b, &dict_bytes, &dict_table)),
            Box::new(|a, b| {
                compare_fused_v2(a, b, &dict_bytes, &dict_table, &token_prefix)
            }),
        ];
        for cmp_fn in &cmp_fns {
            assert_eq!(cmp_fn(&out.tokens[2], &out.tokens[0]), Ordering::Less);
            assert_eq!(
                cmp_fn(&out.tokens[0], &out.tokens[1]),
                Ordering::Less,
                "strict prefix should compare less"
            );
            assert_eq!(cmp_fn(&out.tokens[1], &out.tokens[0]), Ordering::Greater);
            assert_eq!(cmp_fn(&out.tokens[0], &out.tokens[0]), Ordering::Equal);
        }
    }
}
