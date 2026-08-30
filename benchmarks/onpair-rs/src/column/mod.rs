// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The compressed column: decodable data (dictionary + code stream) plus the
//! row layer that delimits the original strings within the stream.
//!
//! The code stream is plain [`Token`] data — there is no separate wrapper type.
//! A bulk-only consumer ignores `row_offsets`; the decode kernels never read it.
//! The compressed-domain [`search`](crate::search) predicates
//! ([`ColumnView::rows_equal_to`] and friends), by contrast, read the row layer
//! to delimit rows and match against their codes without decoding.

use std::mem::MaybeUninit;

use crate::core::dictionary::{CompactDictionary, CompactDictionaryView, Dictionary};
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::core::validate::{InvalidColumn, panic_malformed};
use crate::decoding;
use crate::search::{ContainsTable, PrefixQuery, contains, equals, starts_with, tokenize};

/// Owned compressed column, produced by [`Column::compress`] /
/// [`Parser::parse`](crate::Parser::parse). Self-contained: it carries its own
/// dictionary, so it decodes without reference to the training corpus.
#[derive(Debug, Clone)]
pub struct Column<O: Offset> {
    /// Token dictionary, read-padded.
    pub dict: CompactDictionary,
    /// Code stream: one [`Token`] per emitted token, in row-concatenated order.
    /// Every code is `< dict.num_tokens()`.
    pub codes: Vec<Token>,
    /// Row layer: `R + 1` offsets into `codes` delimiting the `R` rows. Row `k`
    /// is `codes[row_offsets[k]..row_offsets[k + 1]]`. `row_offsets[0] == 0`,
    /// non-decreasing, and `row_offsets[R] == codes.len()`.
    pub row_offsets: Vec<O>,
}

impl<O: Offset> Column<O> {
    /// Compress an Arrow `(bytes, offsets)` value pair end-to-end (train a
    /// dictionary, then encode). `offsets` has `n + 1` entries; string `i` is
    /// `bytes[offsets[i]..offsets[i + 1]]`.
    ///
    /// # Errors
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) if `offsets` is empty or
    /// its last entry exceeds `bytes.len()`.
    pub fn compress(bytes: &[u8], offsets: &[O], cfg: crate::Config) -> Result<Self, crate::Error> {
        crate::compress(bytes, offsets, cfg)
    }

    /// Borrow as a [`ColumnView`].
    #[inline]
    pub fn view(&self) -> ColumnView<'_, O> {
        ColumnView {
            dict: self.dict.as_view(),
            codes: &self.codes,
            row_offsets: &self.row_offsets,
        }
    }

    /// Consume the column and return its owned `(dictionary, codes, row_offsets)`
    /// without copying. This is useful for embedders that want OnPair to own
    /// training and parsing, but store the resulting buffers in their own layout.
    #[inline]
    pub fn into_raw(self) -> (CompactDictionary, Vec<Token>, Vec<O>) {
        (self.dict, self.codes, self.row_offsets)
    }
}

/// Borrowed, `Copy` view over a compressed column — obtained from a [`Column`]
/// or built directly from buffers deserialized from storage.
#[derive(Copy, Clone, Debug)]
pub struct ColumnView<'a, O: Offset> {
    /// The token dictionary.
    pub dict: CompactDictionaryView<'a>,
    /// The code stream (see [`Column::codes`]).
    pub codes: &'a [Token],
    /// The row layer (see [`Column::row_offsets`]).
    pub row_offsets: &'a [O],
}

impl<'a, O: Offset> ColumnView<'a, O> {
    /// Number of rows.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.row_offsets.len().saturating_sub(1)
    }

    /// The codes for row `k`. Precondition: `k < num_rows()`.
    ///
    /// Panics with [`InvalidColumn::BadRowOffsets`] if this view's row layer is
    /// malformed (`row_offsets[k] > row_offsets[k + 1]`, or past the code stream)
    /// — never UB. The bound check is the same one slicing would do anyway, so it
    /// only swaps the panic message; the access itself is unchecked.
    #[inline]
    pub fn row_codes(&self, k: usize) -> &'a [Token] {
        let a = self.row_offsets[k].to_usize();
        let b = self.row_offsets[k + 1].to_usize();
        if b < a || b > self.codes.len() {
            panic_malformed(InvalidColumn::BadRowOffsets);
        }
        // SAFETY: just checked `a <= b <= codes.len()`.
        unsafe { self.codes.get_unchecked(a..b) }
    }

    /// Exact decoded byte length of the whole column — sizes a
    /// [`decompress_into`](Self::decompress_into) buffer (plus
    /// [`DECODE_PADDING`](crate::DECODE_PADDING)). `O(codes)`; an out-of-range code
    /// panics with [`InvalidColumn::CodeOutOfRange`].
    #[inline]
    pub fn decoded_len(&self) -> usize {
        decoding::decoded_len(self.codes, self.dict)
    }

    /// Exact decoded byte length of row `k` — sizes a buffer for a
    /// [`decode_into`](crate::decode_into) over [`row_codes`](Self::row_codes).
    /// Precondition: `k < num_rows()`.
    #[inline]
    pub fn row_decoded_len(&self, k: usize) -> usize {
        decoding::decoded_len(self.row_codes(k), self.dict)
    }

    /// Decode the whole column into `out`, returning the bytes written. Expands the
    /// dictionary to its load-free `WideDictionary` form once — the fast layout
    /// for a bulk decode, reached directly per code with no offset indirection —
    /// then over-reads a fixed 16 bytes per token via
    /// [`decode_into`](crate::decode_into). The caller owns buffer sizing: size
    /// `out` from [`decoded_len`](Self::decoded_len) plus
    /// [`DECODE_PADDING`](crate::DECODE_PADDING).
    ///
    /// For repeated decodes, retain the materialized wide form and decode over its
    /// view with [`decode_into`](crate::decode_into), so the wide form is not
    /// rebuilt on every call.
    ///
    /// # Panics
    /// With [`InvalidColumn::CodeOutOfRange`] on an out-of-range code.
    ///
    /// # Safety
    /// `out.len() >= self.decoded_len() + DECODE_PADDING`.
    #[inline]
    pub unsafe fn decompress_into(&self, out: &mut [MaybeUninit<u8>]) -> usize {
        // Expand to the load-free wide form (fast for a bulk decode). The
        // dictionary view guarantees the preconditions of `to_wide`.
        let wide = self.dict.to_wide();
        // SAFETY: the wide form is read-padded by construction (`n` exact 16-byte
        // rows); the only caller precondition is the buffer size.
        unsafe { decoding::decode_into(self.codes, wide.as_view(), out) }
    }

    /// Decode row `k` into `out`, returning the bytes written — the random-access
    /// analog of [`decompress_into`](Self::decompress_into). Same fixed 16-byte
    /// over-copy per token, but decoded directly over the compact dictionary with
    /// no wide-table build (the wide form would cost `O(num_tokens)` to materialize,
    /// dwarfing a single short row). The caller owns buffer sizing and reuse — size
    /// `out` from [`row_decoded_len`](Self::row_decoded_len) plus
    /// [`DECODE_PADDING`](crate::DECODE_PADDING), and reuse it across rows to avoid
    /// per-row allocation. Precondition: `k < num_rows()`.
    ///
    /// Each code is bounds-checked in the loop; an out-of-range code panics with
    /// [`InvalidColumn::CodeOutOfRange`] (never UB).
    ///
    /// # Safety
    /// `out.len() >= self.row_decoded_len(k) + DECODE_PADDING`.
    #[inline]
    pub unsafe fn decompress_row_into(&self, k: usize, out: &mut [MaybeUninit<u8>]) -> usize {
        // SAFETY: `self.dict` is structurally valid and read-padded, so each token's fixed
        // 16-byte over-read stays in bounds; the caller guarantees `out` holds the
        // row's decoded length plus DECODE_PADDING for the final over-store.
        unsafe { decoding::decode_into(self.row_codes(k), self.dict, out) }
    }

    /// Ascending indices of the rows equal to `needle`. The needle is
    /// [`tokenize`]d once, then matched per row without decoding.
    pub fn rows_equal_to(&self, needle: &[u8]) -> Vec<usize> {
        let query = tokenize(needle, self.dict);
        self.select(|codes| equals(codes, &query))
    }

    /// Ascending indices of the rows starting with `prefix`, prepared once as a
    /// [`PrefixQuery`] and matched per row.
    pub fn rows_starting_with(&self, prefix: &[u8]) -> Vec<usize> {
        let query = PrefixQuery::new(prefix, self.dict);
        self.select(|codes| starts_with(codes, &query))
    }

    /// Ascending indices of the rows containing `pattern` as a substring,
    /// prepared once as a [`ContainsTable`] and matched per row. Panics if
    /// `pattern` exceeds 255 bytes.
    pub fn rows_containing(&self, pattern: &[u8]) -> Vec<usize> {
        let table = ContainsTable::new(pattern, self.dict);
        self.select(|codes| contains(codes, &table))
    }

    /// Ascending indices of the rows whose codes satisfy `pred`.
    fn select(&self, pred: impl Fn(&[Token]) -> bool) -> Vec<usize> {
        (0..self.num_rows())
            .filter(|&k| pred(self.row_codes(k)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::{ColumnView, Config, DECODE_PADDING, DEFAULT_CONFIG, MaxDictBits, compress};

    fn pack(rows: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for r in rows {
            bytes.extend_from_slice(r);
            offsets.push(bytes.len() as u32);
        }
        (bytes, offsets)
    }

    /// Decode the whole column into a fresh `Vec` through the caller-buffer API,
    /// sizing from `decoded_len` (test helper; the crate exposes only into-buffer
    /// decode).
    fn decode_all(view: ColumnView<'_, u32>) -> Vec<u8> {
        let mut buf = vec![std::mem::MaybeUninit::uninit(); view.decoded_len() + DECODE_PADDING];
        // SAFETY: view from a trusted column; buffer carries DECODE_PADDING headroom.
        let w = unsafe { view.decompress_into(&mut buf) };
        let got = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), w) };
        got.to_vec()
    }

    /// Decode a single row into a fresh `Vec` through the into-buffer API — the
    /// per-row counterpart of `decode_all`.
    fn decode_row(view: ColumnView<'_, u32>, k: usize) -> Vec<u8> {
        let mut buf =
            vec![std::mem::MaybeUninit::uninit(); view.row_decoded_len(k) + DECODE_PADDING];
        // SAFETY: buffer sized for row `k`; view from a trusted column.
        let w = unsafe { view.decompress_row_into(k, &mut buf) };
        let got = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), w) };
        got.to_vec()
    }

    #[test]
    fn roundtrip_bulk_and_per_row() {
        let rows: &[&[u8]] = &[b"alpha", b"", b"beta beta", b"gamma"];
        let (bytes, offsets) = pack(rows);
        let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
        let view = col.view();

        assert_eq!(view.decoded_len(), bytes.len());
        assert_eq!(decode_all(view), bytes);
        assert_eq!(view.num_rows(), rows.len());
        for (k, row) in rows.iter().enumerate() {
            assert_eq!(decode_row(view, k), *row, "row {k}");
        }
    }

    /// Decoding over the wide form (what `decompress_into` builds) and directly
    /// over the compact dictionary agree, and both reproduce the input.
    #[test]
    fn compact_and_wide_decode_agree() {
        use crate::decode_into;
        let rows: &[&[u8]] = &[b"alpha", b"", b"beta beta", b"gamma", b"alpha"];
        let (bytes, offsets) = pack(rows);
        let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
        let view = col.view();

        // Wide path: `decompress_into` expands to the wide form internally.
        assert_eq!(decode_all(view), bytes);

        // Compact path: decode the same codes directly over the compact view.
        let mut buf = vec![std::mem::MaybeUninit::uninit(); view.decoded_len() + DECODE_PADDING];
        // SAFETY: read-padded compact dict (from a Column); buffer carries headroom.
        let w = unsafe { decode_into(view.codes, view.dict, &mut buf) };
        let got = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), w) };
        assert_eq!(got, bytes.as_slice());
    }

    #[test]
    fn roundtrip_across_bit_widths() {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for i in 0..5000u32 {
            let row = format!("row-{i:04}-https://example.com/path/{}", i % 37);
            bytes.extend_from_slice(row.as_bytes());
            offsets.push(bytes.len() as u32);
        }
        for bits in 9..=16u8 {
            let cfg = Config {
                max_dict_bits: MaxDictBits::new(bits).unwrap(),
                ..DEFAULT_CONFIG
            };
            let col = compress(&bytes, &offsets, cfg).unwrap();
            assert_eq!(decode_all(col.view()), bytes, "bits={bits}");
        }
    }

    #[test]
    fn code_bits_is_within_capacity() {
        let (bytes, offsets) = pack(&[b"hello world", b"hello there", b"world peace"]);
        let cfg = Config {
            max_dict_bits: MaxDictBits::new(12).unwrap(),
            ..DEFAULT_CONFIG
        };
        let col = compress(&bytes, &offsets, cfg).unwrap();
        // Minimal packing width never exceeds the configured capacity.
        assert!(col.dict.code_bits() <= 12);
    }

    /// A malformed row layer surfaces as a typed `BadRowOffsets` panic through the
    /// safe row accessor, not a generic slice-index panic.
    #[test]
    #[should_panic(expected = "row offsets must be non-decreasing")]
    fn row_codes_panics_typed_on_bad_row_offsets() {
        let (bytes, offsets) = pack(&[b"alpha", b"beta"]);
        let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
        let view = col.view();
        // Row 0 spans codes[0..len+1]: past the code stream.
        let ro = vec![0u32, (view.codes.len() + 1) as u32];
        let bad = ColumnView {
            dict: view.dict,
            codes: view.codes,
            row_offsets: &ro,
        };
        let _ = bad.row_codes(0);
    }

    #[test]
    fn search_selects_matching_rows() {
        let rows: &[&[u8]] = &[b"apple", b"banana", b"apricot", b"cherry", b"apple"];
        let (bytes, offsets) = pack(rows);
        let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
        let view = col.view();

        // Duplicates are returned once per row, in ascending row order.
        assert_eq!(view.rows_equal_to(b"apple"), vec![0, 4]);
        assert_eq!(view.rows_starting_with(b"ap"), vec![0, 2, 4]);
        assert_eq!(view.rows_containing(b"an"), vec![1]);
        // Absent needles select nothing.
        assert_eq!(view.rows_equal_to(b"grape"), Vec::<usize>::new());
    }

    #[test]
    fn search_empty_needle_semantics() {
        let rows: &[&[u8]] = &[b"a", b"", b"abc", b""];
        let (bytes, offsets) = pack(rows);
        let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
        let view = col.view();

        // Equality to "" matches only the empty rows; prefix/substring of ""
        // matches every row.
        assert_eq!(view.rows_equal_to(b""), vec![1, 3]);
        assert_eq!(view.rows_starting_with(b""), vec![0, 1, 2, 3]);
        assert_eq!(view.rows_containing(b""), vec![0, 1, 2, 3]);
    }

    /// The column predicates must agree with a brute-force decode-and-match
    /// oracle — the same contract the `search` module checks per free function,
    /// here exercised end-to-end through `ColumnView`.
    #[test]
    fn search_agrees_with_decode_oracle() {
        use crate::test_corpus::user_strings;
        let corpus: Vec<Vec<u8>> = user_strings(60)
            .into_iter()
            .map(String::into_bytes)
            .collect();
        let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
        let (bytes, offsets) = pack(&rows);
        let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
        let view = col.view();

        let needles: &[&[u8]] = &[
            b"",
            b"h",
            b"https",
            b"https://www.example.com/",
            b"example",
            b".com",
            b"://",
            b"zzz",
        ];
        for &needle in needles {
            let eq: Vec<usize> = (0..view.num_rows())
                .filter(|&k| decode_row(view, k).as_slice() == needle)
                .collect();
            assert_eq!(view.rows_equal_to(needle), eq, "equals {needle:?}");

            let pre: Vec<usize> = (0..view.num_rows())
                .filter(|&k| decode_row(view, k).starts_with(needle))
                .collect();
            assert_eq!(
                view.rows_starting_with(needle),
                pre,
                "starts_with {needle:?}"
            );

            let con: Vec<usize> = (0..view.num_rows())
                .filter(|&k| {
                    let r = decode_row(view, k);
                    needle.is_empty() || r.windows(needle.len()).any(|w| w == needle)
                })
                .collect();
            assert_eq!(view.rows_containing(needle), con, "contains {needle:?}");
        }
    }
}
