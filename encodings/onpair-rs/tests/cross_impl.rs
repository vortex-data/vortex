// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(clippy::tests_outside_test_module)]
//
// Cross-implementation comparison tests: train + encode the same input with
// both the pure-Rust `onpair_lib::Column` and the C++-FFI
// `vortex_onpair_sys::Column`, then assert that downstream operations
// (decompression by row id, equality search, prefix search, substring
// search) agree.
//
// Bit-exact dictionary equality is NOT asserted: the two implementations
// use different RNGs (`std::mt19937_64` vs `rand`'s `StdRng`), so the
// merge-order of the BPE trainer differs. Equivalence is asserted on
// observable outputs: decode equality, predicate equality on the same
// queries, and the structural invariants the FFI guarantees
// (`bits == cfg.bits`, `len == n`, `dict_size <= 2^bits`,
// `codes_boundaries.len() == n + 1`).
//
// The pure-Rust crate exposes `Column::compress` and `Column::parts` with
// the same shape as `vortex-onpair-sys`. We materialise both columns'
// parts and compare what every downstream Vortex consumer (decode loop,
// predicate kernels) would see.

use onpair_lib::Column as RustColumn;
use onpair_lib::OnPairTrainingConfig as RustConfig;
use onpair_lib::Parts as RustParts;
use onpair_lib::unpack_codes_to_u16;
use vortex_onpair_sys::Column as CppColumn;
use vortex_onpair_sys::OnPairTrainingConfig as CppConfig;
use vortex_onpair_sys::Parts as CppParts;

// ─────────────────────────────────────────────────────────────────────────────
// Common helpers.
// ─────────────────────────────────────────────────────────────────────────────

fn pack<S: AsRef<[u8]>>(strings: &[S]) -> (Vec<u8>, Vec<u64>) {
    let mut bytes = Vec::new();
    let mut offsets = Vec::with_capacity(strings.len() + 1);
    offsets.push(0u64);
    for s in strings {
        bytes.extend_from_slice(s.as_ref());
        offsets.push(bytes.len() as u64);
    }
    (bytes, offsets)
}

fn rust_cfg(bits: u32, threshold: f64, seed: u64) -> RustConfig {
    RustConfig { bits, threshold, seed }
}

fn cpp_cfg(bits: u32, threshold: f64, seed: u64) -> CppConfig {
    CppConfig { bits, threshold, seed }
}

/// Decompress row `row` using the pure-Rust decode loop applied to
/// arbitrary `(dict_bytes, dict_offsets, codes_packed, codes_boundaries,
/// bits)`. This is the same logic `vortex-onpair`'s `DecodeView` runs on
/// the materialised children, so when this decoder agrees on both
/// implementations' parts it proves that `vortex-onpair` downstream
/// (decode, LIKE, EQ) would also agree.
fn decode_row<F>(
    bits: u32,
    dict_bytes: &[u8],
    dict_offsets: F,
    codes_packed: &[u64],
    codes_boundaries: &[u32],
    row: usize,
) -> Vec<u8>
where
    F: Fn(usize) -> (usize, usize),
{
    let begin = codes_boundaries[row] as usize;
    let end = codes_boundaries[row + 1] as usize;
    let codes = unpack_codes_to_u16(codes_packed, end, bits);
    let mut out = Vec::new();
    for &c in &codes[begin..end] {
        let (s, e) = dict_offsets(c as usize);
        out.extend_from_slice(&dict_bytes[s..e]);
    }
    out
}

fn decode_rust(parts: &RustParts<'_>, row: usize) -> Vec<u8> {
    decode_row(
        parts.bits,
        parts.dict_bytes,
        |i| (parts.dict_offsets[i] as usize, parts.dict_offsets[i + 1] as usize),
        parts.codes_packed,
        parts.codes_boundaries,
        row,
    )
}

fn decode_cpp(parts: &CppParts<'_>, row: usize) -> Vec<u8> {
    decode_row(
        parts.bits,
        parts.dict_bytes,
        |i| (parts.dict_offsets[i] as usize, parts.dict_offsets[i + 1] as usize),
        parts.codes_packed,
        parts.codes_boundaries,
        row,
    )
}

/// Naive predicate over decoded strings, used as the source of truth for
/// equality / prefix / substring comparisons.
fn predicate_truth<F>(strings: &[&[u8]], f: F) -> Vec<bool>
where
    F: Fn(&[u8]) -> bool,
{
    strings.iter().map(|s| f(s)).collect()
}

fn rust_predicate<F: Fn(&[u8]) -> bool>(parts: &RustParts<'_>, f: F) -> Vec<bool> {
    (0..parts.num_rows)
        .map(|i| f(&decode_rust(parts, i)))
        .collect()
}

fn cpp_predicate<F: Fn(&[u8]) -> bool>(parts: &CppParts<'_>, f: F) -> Vec<bool> {
    (0..parts.num_rows)
        .map(|i| f(&decode_cpp(parts, i)))
        .collect()
}

/// Corpus that produces lots of repetition so BPE merges fire.
fn corpus_urls() -> Vec<&'static [u8]> {
    vec![
        b"https://www.example.com/page",
        b"https://www.example.com/data",
        b"https://www.example.com/page",
        b"https://www.test.org/page",
        b"ftp://files.example.com/x",
        b"https://docs.example.com/spec",
        b"https://api.example.net/v1",
        b"https://www.example.com/data",
        b"https://docs.example.com/spec",
        b"https://www.example.com/page",
        b"another_unique_row",
        b"yet_another_row",
        b"https://api.example.net/v1",
        b"prefix_admin_001",
        b"prefix_admin_002",
        b"prefix_guest_001",
        b"prefix_user_001",
        b"prefix_user_002",
        b"prefix_user_003",
    ]
}

fn corpus_binary() -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(40);
    for i in 0u8..40 {
        let mut row = Vec::with_capacity(24);
        for j in 0u8..24 {
            row.push(i.wrapping_add(j));
        }
        out.push(row);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Structural parity.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn structural_parity_url_corpus() {
    let strings = corpus_urls();
    let (bytes, offsets) = pack(&strings);

    let cpp = CppColumn::compress(&bytes, &offsets, cpp_cfg(12, 0.5, 42)).expect("cpp compress");
    let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(12, 0.5, 42)).expect("rs compress");

    assert_eq!(cpp.len(), strings.len());
    assert_eq!(rs.len(), strings.len());
    assert_eq!(cpp.bits(), 12);
    assert_eq!(rs.bits(), 12);
    // Both stay under the dict-12 cap of 4096.
    assert!(cpp.dict_size() <= 4096);
    assert!(rs.dict_size() <= 4096);
    let cpp_parts = cpp.parts().expect("cpp parts");
    let rs_parts = rs.parts().expect("rs parts");
    // Number of boundary entries is identical: n + 1 in both.
    assert_eq!(cpp_parts.codes_boundaries.len(), strings.len() + 1);
    assert_eq!(rs_parts.codes_boundaries.len(), strings.len() + 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Decompression equivalence.
//
// For every row in the corpus, both columns must decode back to the original
// bytes, regardless of dictionary divergence.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn decompress_equivalence_url_corpus() {
    let strings = corpus_urls();
    let (bytes, offsets) = pack(&strings);
    for &bits in &[9u32, 10, 11, 12, 13, 14, 15, 16] {
        let cpp = CppColumn::compress(&bytes, &offsets, cpp_cfg(bits, 0.5, 42))
            .expect("cpp compress");
        let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(bits, 0.5, 42))
            .expect("rs compress");
        let cpp_parts = cpp.parts().expect("cpp parts");
        let rs_parts = rs.parts().expect("rs parts");

        for (i, &s) in strings.iter().enumerate() {
            assert_eq!(
                decode_cpp(&cpp_parts, i),
                s,
                "C++ decode bits={bits} row={i}"
            );
            assert_eq!(
                decode_rust(&rs_parts, i),
                s,
                "Rust decode bits={bits} row={i}"
            );
        }
    }
}

#[test]
fn decompress_equivalence_binary_corpus() {
    let strings = corpus_binary();
    let strings_ref: Vec<&[u8]> = strings.iter().map(|s| s.as_slice()).collect();
    let (bytes, offsets) = pack(&strings_ref);
    let cpp = CppColumn::compress(&bytes, &offsets, cpp_cfg(14, 0.5, 7)).expect("cpp");
    let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(14, 0.5, 7)).expect("rs");
    let cpp_parts = cpp.parts().expect("cpp parts");
    let rs_parts = rs.parts().expect("rs parts");
    for (i, s) in strings_ref.iter().enumerate() {
        assert_eq!(decode_cpp(&cpp_parts, i), *s, "cpp binary row {i}");
        assert_eq!(decode_rust(&rs_parts, i), *s, "rust binary row {i}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Predicate equivalence (eq / starts_with / contains).
//
// Run the predicate against the decoded value of every row produced by each
// implementation and confirm both implementations agree with the
// naive-string ground truth.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn equals_equivalence() {
    let strings = corpus_urls();
    let (bytes, offsets) = pack(&strings);
    let cpp = CppColumn::compress(&bytes, &offsets, cpp_cfg(12, 0.5, 42)).expect("cpp");
    let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(12, 0.5, 42)).expect("rs");
    let cpp_parts = cpp.parts().expect("cpp parts");
    let rs_parts = rs.parts().expect("rs parts");

    let needles: Vec<&[u8]> = vec![
        b"https://www.example.com/page",
        b"prefix_user_002",
        b"definitely-not-in-corpus",
        b"another_unique_row",
    ];
    for needle in needles {
        let truth = predicate_truth(&strings, |s| s == needle);
        let cpp_sel = cpp_predicate(&cpp_parts, |s| s == needle);
        let rs_sel = rust_predicate(&rs_parts, |s| s == needle);
        assert_eq!(cpp_sel, truth, "cpp eq for {needle:?}");
        assert_eq!(rs_sel, truth, "rust eq for {needle:?}");
        assert_eq!(cpp_sel, rs_sel, "cpp vs rust eq for {needle:?}");
    }
}

#[test]
fn starts_with_equivalence() {
    let strings = corpus_urls();
    let (bytes, offsets) = pack(&strings);
    let cpp = CppColumn::compress(&bytes, &offsets, cpp_cfg(12, 0.5, 42)).expect("cpp");
    let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(12, 0.5, 42)).expect("rs");
    let cpp_parts = cpp.parts().expect("cpp parts");
    let rs_parts = rs.parts().expect("rs parts");

    let needles: Vec<&[u8]> = vec![
        b"https://",
        b"prefix_user_",
        b"prefix_",
        b"ftp://",
        b"zzz_not_present",
    ];
    for needle in needles {
        let truth = predicate_truth(&strings, |s| s.starts_with(needle));
        let cpp_sel = cpp_predicate(&cpp_parts, |s| s.starts_with(needle));
        let rs_sel = rust_predicate(&rs_parts, |s| s.starts_with(needle));
        assert_eq!(cpp_sel, truth, "cpp starts_with for {needle:?}");
        assert_eq!(rs_sel, truth, "rust starts_with for {needle:?}");
        assert_eq!(cpp_sel, rs_sel);
    }
}

#[test]
fn contains_equivalence() {
    let strings = corpus_urls();
    let (bytes, offsets) = pack(&strings);
    let cpp = CppColumn::compress(&bytes, &offsets, cpp_cfg(12, 0.5, 42)).expect("cpp");
    let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(12, 0.5, 42)).expect("rs");
    let cpp_parts = cpp.parts().expect("cpp parts");
    let rs_parts = rs.parts().expect("rs parts");

    let needles: Vec<&[u8]> = vec![
        b"example",
        b"admin",
        b"docs",
        b"_user_",
        b"never_appears",
    ];
    for needle in needles {
        let truth = predicate_truth(&strings, |s| {
            s.windows(needle.len()).any(|w| w == needle)
        });
        let cpp_sel = cpp_predicate(&cpp_parts, |s| {
            s.windows(needle.len()).any(|w| w == needle)
        });
        let rs_sel = rust_predicate(&rs_parts, |s| {
            s.windows(needle.len()).any(|w| w == needle)
        });
        assert_eq!(cpp_sel, truth, "cpp contains for {needle:?}");
        assert_eq!(rs_sel, truth, "rust contains for {needle:?}");
        assert_eq!(cpp_sel, rs_sel);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dictionary structural invariants.
//
// Both implementations must produce a dictionary that:
//   * begins with all 256 single-byte tokens (in some order) — sufficient to
//     parse every possible byte;
//   * is lexicographically sorted by token byte sequence (required for
//     binary-search prefix lookups in downstream predicates).
// ─────────────────────────────────────────────────────────────────────────────

fn dict_contains_all_single_bytes(dict_bytes: &[u8], dict_offsets: &[u32]) -> bool {
    let mut found = [false; 256];
    for i in 0..dict_offsets.len() - 1 {
        let s = dict_offsets[i] as usize;
        let e = dict_offsets[i + 1] as usize;
        if e - s == 1 {
            found[dict_bytes[s] as usize] = true;
        }
    }
    found.iter().all(|&f| f)
}

fn dict_is_sorted(dict_bytes: &[u8], dict_offsets: &[u32]) -> bool {
    for i in 1..dict_offsets.len() - 1 {
        let a_s = dict_offsets[i - 1] as usize;
        let a_e = dict_offsets[i] as usize;
        let b_s = dict_offsets[i] as usize;
        let b_e = dict_offsets[i + 1] as usize;
        if dict_bytes[a_s..a_e] > dict_bytes[b_s..b_e] {
            return false;
        }
    }
    true
}

#[test]
fn both_dicts_cover_all_single_bytes() {
    let strings = corpus_urls();
    let (bytes, offsets) = pack(&strings);
    let cpp = CppColumn::compress(&bytes, &offsets, cpp_cfg(12, 0.5, 42)).expect("cpp");
    let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(12, 0.5, 42)).expect("rs");
    let cpp_parts = cpp.parts().expect("cpp parts");
    let rs_parts = rs.parts().expect("rs parts");
    assert!(dict_contains_all_single_bytes(cpp_parts.dict_bytes, cpp_parts.dict_offsets));
    assert!(dict_contains_all_single_bytes(rs_parts.dict_bytes, rs_parts.dict_offsets));
}

#[test]
fn both_dicts_are_lex_sorted() {
    let strings = corpus_urls();
    let (bytes, offsets) = pack(&strings);
    let cpp = CppColumn::compress(&bytes, &offsets, cpp_cfg(12, 0.5, 42)).expect("cpp");
    let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(12, 0.5, 42)).expect("rs");
    let cpp_parts = cpp.parts().expect("cpp parts");
    let rs_parts = rs.parts().expect("rs parts");
    assert!(dict_is_sorted(cpp_parts.dict_bytes, cpp_parts.dict_offsets));
    assert!(dict_is_sorted(rs_parts.dict_bytes, rs_parts.dict_offsets));
}

// ─────────────────────────────────────────────────────────────────────────────
// Bitmap-API equivalence.
//
// `onpair-lib`'s predicate API returns the same LSB-first packed bitmap
// layout as `vortex-onpair-sys`'s `*_bitmap` family. Compare the two
// directly against the C++-FFI implementation on identical inputs.
// ─────────────────────────────────────────────────────────────────────────────

mod bitmap_parity {
    use super::*;

    fn corpus_for_predicates() -> Vec<&'static [u8]> {
        vec![
            b"https://www.example.com/page",
            b"https://www.example.com/data",
            b"https://www.test.org/page",
            b"ftp://files.example.com/x",
            b"https://www.example.com/page",
            b"admin_001",
            b"admin_002",
            b"guest_001",
            b"user_007",
            b"no_match_row",
            b"another_no_match",
        ]
    }

    #[test]
    fn equals_bitmap_matches_cpp() {
        let strings = corpus_for_predicates();
        let (bytes, offsets) = pack(&strings);
        let cpp = CppColumn::compress(&bytes, &offsets, cpp_cfg(12, 0.5, 42)).expect("cpp");
        let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(12, 0.5, 42)).expect("rs");
        for needle in [&b"admin_001"[..], b"missing", b"user_007", b""] {
            let cpp_bits = cpp.equals_bitmap(needle).expect("cpp eq");
            let rs_bits = rs.equals_bitmap(needle);
            assert_eq!(cpp_bits, rs_bits, "needle={needle:?}");
        }
    }

    #[test]
    fn starts_with_bitmap_matches_cpp() {
        let strings = corpus_for_predicates();
        let (bytes, offsets) = pack(&strings);
        let cpp = CppColumn::compress(&bytes, &offsets, cpp_cfg(12, 0.5, 42)).expect("cpp");
        let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(12, 0.5, 42)).expect("rs");
        for needle in [&b"https://"[..], b"admin_", b"ftp://", b"zzz", b""] {
            let cpp_bits = cpp.starts_with_bitmap(needle).expect("cpp sw");
            let rs_bits = rs.starts_with_bitmap(needle);
            assert_eq!(cpp_bits, rs_bits, "needle={needle:?}");
        }
    }

    #[test]
    fn contains_bitmap_matches_cpp() {
        let strings = corpus_for_predicates();
        let (bytes, offsets) = pack(&strings);
        let cpp = CppColumn::compress(&bytes, &offsets, cpp_cfg(12, 0.5, 42)).expect("cpp");
        let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(12, 0.5, 42)).expect("rs");
        for needle in [&b"example"[..], b"admin", b"never", b"_00", b""] {
            let cpp_bits = cpp.contains_bitmap(needle).expect("cpp ct");
            let rs_bits = rs.contains_bitmap(needle);
            assert_eq!(cpp_bits, rs_bits, "needle={needle:?}");
        }
    }

    #[test]
    fn decompress_row_matches_cpp() {
        let strings = corpus_for_predicates();
        let (bytes, offsets) = pack(&strings);
        let cpp = CppColumn::compress(&bytes, &offsets, cpp_cfg(12, 0.5, 42)).expect("cpp");
        let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(12, 0.5, 42)).expect("rs");
        let mut cpp_buf = Vec::new();
        let mut rs_buf = Vec::new();
        for (i, s) in strings.iter().enumerate() {
            cpp.decompress_row(i, &mut cpp_buf).expect("cpp dec");
            rs.decompress_row(i, &mut rs_buf).expect("rs dec");
            assert_eq!(cpp_buf, *s, "cpp row {i}");
            assert_eq!(rs_buf, *s, "rs row {i}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-pattern (Aho-Corasick) equivalence against naive truth.
// `vortex-onpair-sys` exposes single-needle predicates only, so we compare
// against the byte-level union-of-substrings predicate.
// ─────────────────────────────────────────────────────────────────────────────

mod multi_pattern {
    use super::*;

    #[test]
    fn multi_pattern_matches_naive_union() {
        let strings: Vec<&[u8]> = vec![
            b"admin_001",
            b"guest_999",
            b"user_007",
            b"no_pattern_here",
            b"some_admin_text",
            b"guest_in_middle",
        ];
        let (bytes, offsets) = pack(&strings);
        let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(12, 0.5, 42)).expect("rs");
        let needles: &[&[u8]] = &[b"admin", b"guest"];
        let rs_bits = rs.multi_pattern_bitmap(needles);
        // Naive truth: row matches iff any needle is a substring of it.
        let mut expected = onpair_lib::empty_bitmap(strings.len());
        for (i, s) in strings.iter().enumerate() {
            if needles
                .iter()
                .any(|n| s.windows(n.len()).any(|w| &w == n))
            {
                expected[i / 8] |= 1u8 << (i % 8);
            }
        }
        assert_eq!(rs_bits, expected);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Combinator algebra: A AND NOT B, A OR B over predicate bitmaps.
// ─────────────────────────────────────────────────────────────────────────────

mod combinators {
    use super::*;
    use onpair_lib::{bitmap_and, bitmap_not, bitmap_or, get_bit};

    #[test]
    fn and_not_matches_set_difference() {
        let strings: Vec<&[u8]> = vec![
            b"user_admin_001",
            b"user_guest_001",
            b"admin_001",
            b"guest_001",
            b"user_007",
        ];
        let (bytes, offsets) = pack(&strings);
        let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(12, 0.5, 42)).expect("rs");

        let users = rs.contains_bitmap(b"user");
        let guests = rs.contains_bitmap(b"guest");
        let users_not_guests = bitmap_and(&users, &bitmap_not(&guests, strings.len()));

        // Truth: rows that contain "user" but not "guest".
        let expected: Vec<bool> = strings
            .iter()
            .map(|s| {
                let has_user = s.windows(4).any(|w| w == b"user");
                let has_guest = s.windows(5).any(|w| w == b"guest");
                has_user && !has_guest
            })
            .collect();
        for (i, &want) in expected.iter().enumerate() {
            assert_eq!(get_bit(&users_not_guests, i), want, "row {i}");
        }
    }

    #[test]
    fn or_matches_union() {
        let strings: Vec<&[u8]> = vec![b"alpha", b"beta", b"gamma", b"delta"];
        let (bytes, offsets) = pack(&strings);
        let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(12, 0.5, 42)).expect("rs");

        let a = rs.starts_with_bitmap(b"alp");
        let b = rs.starts_with_bitmap(b"del");
        let union = bitmap_or(&a, &b);

        for (i, s) in strings.iter().enumerate() {
            let want = s.starts_with(b"alp") || s.starts_with(b"del");
            assert_eq!(get_bit(&union, i), want, "row {i}");
        }
    }

    #[test]
    fn not_matches_complement() {
        let strings: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d", b"e"];
        let (bytes, offsets) = pack(&strings);
        let rs = RustColumn::compress(&bytes, &offsets, rust_cfg(12, 0.5, 42)).expect("rs");
        let b = rs.equals_bitmap(b"b");
        let not_b = bitmap_not(&b, strings.len());
        for (i, s) in strings.iter().enumerate() {
            assert_eq!(get_bit(&not_b, i), s != b"b", "row {i}");
        }
    }
}
