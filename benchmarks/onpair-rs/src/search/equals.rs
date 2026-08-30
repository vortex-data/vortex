// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compressed-domain equality search.
//!
//! Two byte strings are equal iff they encode to the same code sequence — the
//! dictionary's uniqueness invariant guarantees distinct code sequences denote
//! distinct strings, and its completeness makes the encoder's segmentation
//! canonical. So equality needs no decoding and no dictionary at scan time:
//! [`tokenize`](super::tokenize()) the needle once into its code key, then compare
//! each row's codes against that key.

use crate::core::types::Token;

/// Whether `codes` equals the query's code sequence `query`.
#[inline]
pub fn equals(codes: &[Token], query: &[Token]) -> bool {
    if codes.len() != query.len() {
        return false;
    }
    for (a, b) in codes.iter().zip(query) {
        if a != b {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::tokenize;
    use crate::{Column, DEFAULT_CONFIG, compress};

    fn compress_rows(rows: &[&[u8]]) -> Column<u32> {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for r in rows {
            bytes.extend_from_slice(r);
            offsets.push(bytes.len() as u32);
        }
        compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap()
    }

    /// Decode row `k` to bytes via the into-buffer API, for the oracle.
    fn decode_row(view: crate::ColumnView<'_, u32>, k: usize) -> Vec<u8> {
        let mut buf =
            vec![std::mem::MaybeUninit::uninit(); view.row_decoded_len(k) + crate::DECODE_PADDING];
        // SAFETY: buffer sized for row `k`; view from a trusted column.
        let w = unsafe { view.decompress_row_into(k, &mut buf) };
        unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), w) }.to_vec()
    }

    /// Driving `equals` over every row must agree with a brute-force
    /// decode-and-compare oracle, for each needle.
    fn check(rows: &[&[u8]], needles: &[&[u8]]) {
        let col = compress_rows(rows);
        let view = col.view();
        for &needle in needles {
            let query = tokenize(needle, view.dict);
            let got: Vec<usize> = (0..view.num_rows())
                .filter(|&k| equals(view.row_codes(k), &query))
                .collect();
            let want: Vec<usize> = (0..view.num_rows())
                .filter(|&k| decode_row(view, k).as_slice() == needle)
                .collect();
            assert_eq!(got, want, "needle {needle:?}");
        }
    }

    #[test]
    fn finds_every_row_as_itself() {
        let rows: &[&[u8]] = &[b"alpha", b"beta", b"gamma", b"delta"];
        check(rows, rows);
    }

    #[test]
    fn returns_all_duplicate_matches() {
        let col = compress_rows(&[b"cat", b"dog", b"cat", b"bird", b"cat"]);
        let view = col.view();
        let query = tokenize(b"cat", view.dict);
        let hits: Vec<usize> = (0..view.num_rows())
            .filter(|&k| equals(view.row_codes(k), &query))
            .collect();
        assert_eq!(hits, vec![0, 2, 4]);
    }

    #[test]
    fn distinguishes_equality_from_prefix() {
        // "ab" matches only the exact row, never the longer "abc"/"abcd".
        let rows: &[&[u8]] = &[b"ab", b"abc", b"abcd", b"ab"];
        check(rows, &[b"ab", b"abc", b"abcd", b"a", b"abcde"]);
    }

    #[test]
    fn absent_and_empty_needles() {
        let rows: &[&[u8]] = &[b"red", b"", b"green", b""];
        check(rows, &[b"yellow", b"re", b"", b"RED"]);
    }

    #[test]
    fn matches_brute_force_on_binary_corpus() {
        use crate::test_corpus::binary_strings;
        let corpus = binary_strings(40, 24, 7);
        let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
        let mut needles = rows.clone();
        needles.push(b"");
        needles.push(b"\x00\x01\x02");
        check(&rows, &needles);
    }
}
