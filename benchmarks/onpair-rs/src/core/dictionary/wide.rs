// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The wide (decode-optimized) dictionary representation.
//!
//! [`WideDictionary`] stores `num_tokens` rows of [`MAX_TOKEN_SIZE`] bytes — row
//! `id` holds token `id` (zero-padded) — plus a per-token length. Trades space
//! for a **load-free** token address: token `id`'s bytes are at `data + id*16`,
//! with no `code → offset → bytes` indirection, so a decode is one independent
//! load.
//!
//! Build it from the compact form with
//! [`CompactDictionaryView::to_wide`](super::CompactDictionaryView::to_wide), or
//! materialize it directly from [`DictionaryStorage`] with
//! [`WideDictionary::validate_safety`]. The compact form is the serialized one;
//! the wide form exists only to accelerate decoding, so the conversion is
//! one-way. Both representations implement [`DictionaryView`] (via their
//! views), so the decode kernels treat them uniformly.
//!
//! # Invariants
//! Upheld by [`CompactDictionaryView::to_wide`](super::CompactDictionaryView::to_wide)
//! and [`WideDictionary::validate_safety`]. They either receive a compact view
//! that already satisfies the structural invariants or establish those
//! invariants before materializing the wide form. The wide form is never
//! deserialized, so there is no untrusted wide type.
//! - `lens.len() == num_tokens` and `data.len() == num_tokens * MAX_TOKEN_SIZE`.
//! - **Bounded lengths** — every `lens[id]` is in `1..=MAX_TOKEN_SIZE`, so every
//!   token is non-empty; row `id`'s first `lens[id]` bytes are token `id`, and the
//!   rest of the row is unused.
//!
use super::{Dictionary, DictionaryStorage, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token};
use crate::core::validate::InvalidColumn;

/// Owned wide dictionary: `num_tokens` rows of [`MAX_TOKEN_SIZE`] bytes plus
/// per-token lengths.
///
/// Like [`CompactDictionary`](super::CompactDictionary), the fields are private.
/// Obtain one from [`CompactDictionaryView::to_wide`](super::CompactDictionaryView::to_wide)
/// or directly from [`DictionaryStorage`] with [`validate_safety`](Self::validate_safety).
/// There is no untrusted wide type — the wide form is materialized in memory and
/// never deserialized directly.
#[derive(Default, Debug, Clone)]
pub struct WideDictionary {
    /// `num_tokens * MAX_TOKEN_SIZE` row-major token bytes; row `id` holds token
    /// `id`, zero-padded to the row width.
    data: Vec<u8>,
    /// `num_tokens` true token lengths.
    lens: Vec<u8>,
}

impl WideDictionary {
    /// Number of tokens.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.lens.len()
    }

    /// Validate compact dictionary storage for safe decoding and materialize the
    /// wide representation without constructing an intermediate
    /// [`CompactDictionary`](super::CompactDictionary).
    ///
    /// This checks only the structural invariants required for bounded token
    /// access and decoding. It deliberately does not inspect token contents, so
    /// sortedness, uniqueness, and alphabet completeness are not established.
    /// The storage is copied into the wide representation and may be dropped
    /// after this call returns.
    ///
    /// The hot loop accumulates validation failures instead of returning early.
    /// On valid input it performs one offset traversal and one fixed-width copy
    /// per token; the fixed-width copies are independent and easy for the
    /// optimizer to lower to load/store pairs.
    pub fn validate_safety<S>(storage: S) -> Result<Self, InvalidColumn>
    where
        S: DictionaryStorage<u32>,
    {
        let bytes = storage.bytes();
        let offsets = storage.offsets();
        let Some(num_tokens) = offsets.len().checked_sub(1) else {
            return Err(InvalidColumn::EmptyDictionary);
        };
        if num_tokens == 0 {
            return Err(InvalidColumn::EmptyDictionary);
        }
        if num_tokens > (Token::MAX as usize + 1) {
            return Err(InvalidColumn::CodeOutOfRange);
        }
        if offsets.first().copied() != Some(0) {
            return Err(InvalidColumn::FirstOffsetNotZero);
        }

        let mut data = vec![0u8; num_tokens * MAX_TOKEN_SIZE];
        let mut lens = vec![0u8; num_tokens];
        let source = bytes.as_ptr();
        let destination = data.as_mut_ptr();

        // If the offsets are valid and non-decreasing, the last token start is
        // the greatest source address. This one check proves the fixed-width
        // read for every token. During the loop we additionally require each
        // start to be at most `last_start` before copying, because monotonicity
        // has not been established yet.
        let last_start = offsets[num_tokens - 1];
        let last_source_is_readable =
            u64::from(last_start) + MAX_TOKEN_SIZE as u64 <= bytes.len() as u64;

        let mut bad_decreasing = 0u32;
        let mut bad_empty = 0u32;
        let mut bad_length = 0u32;
        let mut bad_padding = (!last_source_is_readable) as u32;

        for id in 0..num_tokens {
            let start = offsets[id];
            let end = offsets[id + 1];
            let length = end.wrapping_sub(start);

            bad_decreasing |= (end < start) as u32;
            bad_empty |= (end == start) as u32;
            bad_length |= (length > MAX_TOKEN_SIZE as u32) as u32;

            // A final-offset check is insufficient until monotonicity has been
            // established: malformed input could put an earlier start past the
            // final start. This one integer comparison prevents that source read.
            let source_is_readable = last_source_is_readable && start <= last_start;
            bad_padding |= (!source_is_readable) as u32;

            if end > start && length <= MAX_TOKEN_SIZE as u32 && source_is_readable {
                let source_offset = start as usize;
                let destination_offset = id * MAX_TOKEN_SIZE;
                // SAFETY: `source_is_readable` proves that the fixed-width source
                // read is in bounds. `end > start` and the length bound make the
                // stored length exact. The destination row is within the vectors
                // allocated from `num_tokens`.
                unsafe {
                    *lens.get_unchecked_mut(id) = length as u8;
                    std::ptr::copy_nonoverlapping(
                        source.add(source_offset),
                        destination.add(destination_offset),
                        MAX_TOKEN_SIZE,
                    );
                }
            }
        }

        if bad_decreasing != 0 {
            return Err(InvalidColumn::DecreasingOffsets);
        }
        if bad_empty != 0 {
            return Err(InvalidColumn::EmptyToken);
        }
        if bad_length != 0 {
            return Err(InvalidColumn::TokenTooLarge);
        }
        if bad_padding != 0 {
            return Err(InvalidColumn::MissingPadding);
        }

        Ok(Self::from_raw(data, lens))
    }

    /// Seal row-major `data` and `lens` into a wide dictionary. The caller must
    /// guarantee the physical layout invariants.
    #[inline]
    pub(crate) fn from_raw(data: Vec<u8>, lens: Vec<u8>) -> Self {
        Self { data, lens }
    }
}

impl Dictionary for WideDictionary {
    type View<'a> = WideDictionaryView<'a>;
    #[inline]
    fn as_view(&self) -> WideDictionaryView<'_> {
        WideDictionaryView {
            data: &self.data,
            lens: &self.lens,
        }
    }
}

/// Borrowed, `Copy` view over a wide dictionary's buffers.
///
/// Private fields: obtained only from a [`WideDictionary`] via [`Dictionary::as_view`].
#[derive(Copy, Clone, Debug)]
pub struct WideDictionaryView<'a> {
    /// `num_tokens * MAX_TOKEN_SIZE` row-major token bytes.
    data: &'a [u8],
    /// `num_tokens` true token lengths.
    lens: &'a [u8],
}

impl DictionaryView for WideDictionaryView<'_> {
    #[inline]
    fn num_tokens(&self) -> usize {
        self.lens.len()
    }

    #[inline]
    fn token(&self, id: Token) -> &[u8] {
        let row = id as usize * MAX_TOKEN_SIZE;
        &self.data[row..row + self.lens[id as usize] as usize]
    }

    #[inline]
    fn token_len(&self, id: Token) -> usize {
        self.lens[id as usize] as usize
    }

    #[inline]
    unsafe fn token_ptr(&self, id: Token) -> *const u8 {
        // SAFETY: id < num_tokens ⇒ row id is within data; the last row ends
        // exactly at data.len() (= n*16), so 16 bytes are readable. No load.
        unsafe { self.data.as_ptr().add(id as usize * MAX_TOKEN_SIZE) }
    }

    #[inline]
    unsafe fn token_len_unchecked(&self, id: Token) -> usize {
        // SAFETY: id < num_tokens ⇒ lens[id] is in bounds.
        unsafe { *self.lens.get_unchecked(id as usize) as usize }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionary, OwnedDictionaryStorage, pad_raw};
    use crate::search::{ContainsTable, PrefixQuery, contains, starts_with, tokenize};
    use crate::{DECODE_PADDING, decode_into, decoded_len, try_decode_into};
    use std::mem::MaybeUninit;

    fn padded_storage(tokens: &[&[u8]]) -> OwnedDictionaryStorage {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for t in tokens {
            bytes.extend_from_slice(t);
            offsets.push(bytes.len() as u32);
        }
        pad_raw(&mut bytes, &offsets);
        OwnedDictionaryStorage::new(bytes, offsets)
    }

    fn padded_compact(tokens: &[&[u8]]) -> CompactDictionary {
        let (bytes, offsets) = padded_storage(tokens).into_raw();
        CompactDictionary::from_raw(bytes, offsets)
    }

    /// Exercise decoding and search with a dictionary that passed only the
    /// structural safety boundary. Correctness properties may be violated, but
    /// every token remains bounded and non-empty.
    fn assert_safe_use(tokens: &[&[u8]], text: &[u8], codes: &[Token]) {
        let dictionary = WideDictionary::validate_safety(padded_storage(tokens)).unwrap();
        let view = dictionary.as_view();

        // Tokenization can return the wrong codes without a sorted/complete
        // dictionary, but it must always make progress and consume the input.
        let tokenized = tokenize(text, view);
        assert_eq!(
            tokenized
                .iter()
                .map(|&code| view.token_len(code))
                .sum::<usize>(),
            text.len()
        );

        let prefix = PrefixQuery::new(text, view);
        let _ = starts_with(codes, &prefix);
        let table = ContainsTable::new(text, view);
        let _ = contains(codes, &table);

        let expected: Vec<u8> = codes
            .iter()
            .flat_map(|&code| view.token(code).iter().copied())
            .collect();
        let decoded_len = decoded_len(codes, view);
        assert_eq!(decoded_len, expected.len());

        let mut padded_out = Vec::with_capacity(decoded_len + DECODE_PADDING);
        let written = unsafe { decode_into(codes, view, padded_out.spare_capacity_mut()) };
        unsafe { padded_out.set_len(written) };
        assert_eq!(padded_out, expected);

        let mut exact_out = vec![MaybeUninit::uninit(); decoded_len];
        let written = try_decode_into(codes, view, &mut exact_out).unwrap();
        let exact_bytes =
            unsafe { std::slice::from_raw_parts(exact_out.as_ptr().cast::<u8>(), written) };
        assert_eq!(exact_bytes, expected.as_slice());
    }

    #[test]
    fn num_tokens_counts_rows() {
        let wide = padded_compact(&[b"a", b"bc", b"def"]).to_wide();
        assert_eq!(wide.num_tokens(), 3);
        assert_eq!(wide.as_view().num_tokens(), 3);
    }

    #[test]
    fn to_wide_rows_and_lens_match_tokens() {
        // `to_wide` lays each token in its own zero-padded MAX_TOKEN_SIZE row.
        let tokens: &[&[u8]] = &[b"a", b"bc", b"def", b"ghij"];
        let wide = padded_compact(tokens).to_wide();
        assert_eq!(wide.num_tokens(), tokens.len());
        for (id, tok) in tokens.iter().enumerate() {
            assert_eq!(wide.lens[id] as usize, tok.len());
            assert_eq!(
                &wide.data[id * MAX_TOKEN_SIZE..id * MAX_TOKEN_SIZE + tok.len()],
                *tok
            );
        }
    }

    #[test]
    fn validate_safety_materializes_storage_directly() {
        let storage = padded_storage(&[b"a", b"bc", b"def"]);
        let expected = padded_compact(&[b"a", b"bc", b"def"]).to_wide();
        let wide = WideDictionary::validate_safety(storage).unwrap();

        assert_eq!(wide.num_tokens(), expected.num_tokens());
        for id in 0..wide.num_tokens() {
            assert_eq!(
                wide.as_view().token(id as Token),
                expected.as_view().token(id as Token)
            );
            assert_eq!(
                wide.as_view().token_len(id as Token),
                expected.as_view().token_len(id as Token)
            );
        }
    }

    #[test]
    fn validate_safety_accepts_semantically_malformed_storage() {
        // Sortedness and alphabet completeness are correctness properties, not
        // prerequisites for safely materializing the wide representation.
        let storage = padded_storage(&[b"b", b"a"]);
        let wide = WideDictionary::validate_safety(storage).unwrap();

        assert_eq!(wide.as_view().token(0), b"b");
        assert_eq!(wide.as_view().token(1), b"a");
    }

    #[test]
    fn safety_valid_semantically_malformed_dictionary_remains_safe_to_use() {
        // The compact and wide dictionaries expose the same structural safety
        // guarantee even when correctness-only properties are violated.
        assert_safe_use(&[&[1u8], &[0u8]], b"xyz", &[0, 1, 0, 1]);
        assert_safe_use(&[&[0u8], &[1u8], &[2u8]], b"xyz", &[2, 0, 1, 2]);
        assert_safe_use(&[&[0u8], &[0u8]], b"xyz", &[0, 1, 0]);
    }

    #[test]
    fn validate_safety_rejects_malformed_storage() {
        let cases = [
            (
                OwnedDictionaryStorage::new(vec![0; MAX_TOKEN_SIZE], vec![0]),
                InvalidColumn::EmptyDictionary,
            ),
            (
                OwnedDictionaryStorage::new(vec![0; MAX_TOKEN_SIZE + 1], vec![1, 2]),
                InvalidColumn::FirstOffsetNotZero,
            ),
            (
                OwnedDictionaryStorage::new(vec![0; MAX_TOKEN_SIZE + 2], vec![0, 2, 1]),
                InvalidColumn::DecreasingOffsets,
            ),
            (
                OwnedDictionaryStorage::new(vec![0; MAX_TOKEN_SIZE + 17], vec![0, 17]),
                InvalidColumn::TokenTooLarge,
            ),
            (
                OwnedDictionaryStorage::new(b"abc".to_vec(), vec![0, 1, 3]),
                InvalidColumn::MissingPadding,
            ),
        ];

        for (storage, expected) in cases {
            assert_eq!(
                WideDictionary::validate_safety(storage).map(|_| ()),
                Err(expected)
            );
        }
    }
}
