// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The compact dictionary: Arrow binary `bytes` + `offsets`.
//!
//! Layout is Arrow binary — a flat `bytes` buffer plus an `offsets` index of
//! length `num_tokens + 1`; token `i` is `bytes[offsets[i]..offsets[i + 1]]`.
//!
//! # Invariants
//! A safety-validated dictionary satisfies the structural invariants required by
//! every accessor and by decoding:
//! - `offsets[0] == 0` and `offsets.len() == num_tokens + 1`.
//! - **Strictly increasing** offsets — every token is non-empty, with length in
//!   `1..=MAX_TOKEN_SIZE`.
//! - **Addressable** — at most `2^16` tokens, so every token id fits in
//!   [`Token`].
//! - **Read-padded** — `bytes` is readable for [`MAX_TOKEN_SIZE`] bytes past the
//!   highest token offset. `offsets.last()` is the logical length; `bytes.len()`
//!   may exceed it by the padding.
//!
//! A correctness-validated dictionary, and every dictionary produced by the
//! trainer, additionally satisfies the semantic conformance invariants:
//! - **Sorted** — tokens are in strictly ascending bytewise-lexicographic order.
//! - **Complete** — all 256 single-byte tokens are present, so any byte string
//!   is encodable.
//! - **Unique** — no two tokens are equal.
//!
//! Safety validation establishes only the first group. Search and tokenization
//! require the second group as well; a safety-valid dictionary can be decoded
//! safely while still producing incorrect search results.

use super::{Dictionary, DictionaryView, WideDictionary};
use crate::core::types::{MAX_TOKEN_SIZE, Token};
use crate::core::validate::InvalidColumn;

/// Storage for a compact dictionary's serialized buffers.
///
/// Implementations must keep the returned slices immutable and stable for as
/// long as the storage value is alive. In particular, repeated calls must
/// refer to the same logical buffers, and an implementation must not expose a
/// mutable alias that can change either buffer while the storage is alive.
///
/// This is what lets a validated [`CompactDictionary`] retain `S` and lend the
/// same zero-copy view as the default owned representation. The storage itself
/// is only a raw buffer carrier; [`CompactDictionary::validate_safety`]
/// is the safe boundary that establishes the structural decoder invariants.
/// Validation does not freeze, copy, or snapshot the buffers. Violating this
/// contract after validation can cause undefined behavior in the unchecked
/// decoder, even if validation initially succeeds.
pub trait DictionaryStorage<D> {
    /// The concatenated, read-padded token bytes.
    fn bytes(&self) -> &[u8];

    /// The `num_tokens + 1` offsets delimiting the token bytes.
    fn offsets(&self) -> &[D];
}

/// The default owned storage used by [`CompactDictionary`].
#[derive(Debug, Clone)]
pub struct OwnedDictionaryStorage {
    bytes: Vec<u8>,
    offsets: Vec<u32>,
}

impl OwnedDictionaryStorage {
    /// Build owned dictionary storage from its serialized buffers.
    pub fn new(bytes: Vec<u8>, offsets: Vec<u32>) -> Self {
        Self { bytes, offsets }
    }

    /// Consume the storage and return its serialized buffers without copying.
    pub fn into_raw(self) -> (Vec<u8>, Vec<u32>) {
        (self.bytes, self.offsets)
    }
}

impl DictionaryStorage<u32> for OwnedDictionaryStorage {
    #[inline]
    fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[inline]
    fn offsets(&self) -> &[u32] {
        &self.offsets
    }
}

/// Maximum number of dictionary entries addressable by a [`Token`].
const MAX_NUM_TOKENS: usize = Token::MAX as usize + 1;

/// Append `MAX_TOKEN_SIZE - len(last token)` zero bytes to `bytes` so the decoder's
/// fixed-width read from any token offset stays in bounds — the read-padding
/// invariant. Applied once, on the raw buffers, just before sealing a
/// [`CompactDictionary`]. Idempotent: a no-op once the padding is present or when
/// the last token is already `MAX_TOKEN_SIZE` wide.
pub(crate) fn pad_raw(bytes: &mut Vec<u8>, offsets: &[u32]) {
    if offsets.len() < 2 {
        return;
    }
    let last_token_start = offsets[offsets.len() - 2] as usize;
    let required = last_token_start
        .checked_add(MAX_TOKEN_SIZE)
        .expect("dictionary padding length must fit in usize");
    if bytes.len() < required {
        bytes.resize(required, 0);
    }
}

/// Validate the structural invariants required by the unchecked decoder and the
/// safe dictionary accessors.
///
/// The safety boundary intentionally checks only the following properties:
///
/// * The dictionary contains at least one token. The search tokenizer constructs
///   an inclusive token range ending at `num_tokens - 1`, so an empty dictionary
///   cannot be used by that API.
/// * The number of tokens fits the `u16` token-code address space. This is partly
///   an addressability invariant rather than a raw pointer-safety requirement:
///   the decoder checks each code, but search ranges are represented as `u16`.
/// * The first offset is zero, preserving the compact dictionary's Arrow-style
///   layout. This is a cheap structural check; the final padding check below is
///   what proves the token reads themselves are in bounds.
/// * Every adjacent offset pair is strictly increasing. This prevents the
///   unchecked length subtraction from underflowing and ensures search advances
///   past every token instead of looping on an empty token.
/// * Every token length is at most [`MAX_TOKEN_SIZE`]. The current decoder copies
///   a fixed-width `MAX_TOKEN_SIZE`-byte chunk while advancing its output cursor
///   by the logical token length; this bound is part of that output-safety proof.
/// * The first byte of the last token has at least `MAX_TOKEN_SIZE` readable bytes
///   after it. The last token start is the highest token start, so this one check
///   proves every fixed-width dictionary read is in bounds, including reads into
///   the trailing padding.
///
/// Sortedness, uniqueness, and alphabet completeness are deliberately not checked
/// here. They affect search and tokenization results, but not memory safety; they
/// are checked by [`validate_conformance`] for callers that require correctness.
///
/// This is deliberately an offset-only pass. It does not read token bytes, so it
/// is suitable for the cheap validation path used by consumers that only need
/// memory safety. The loop scans the complete offset pairs and accumulates error
/// flags instead of returning early, keeping its hot path simple enough for the
/// optimizer to vectorize.
fn validate_safety(bytes: &[u8], offsets: &[u32]) -> Result<(), InvalidColumn> {
    let Some(num_tokens) = offsets.len().checked_sub(1) else {
        return Err(InvalidColumn::EmptyDictionary);
    };
    if num_tokens == 0 {
        return Err(InvalidColumn::EmptyDictionary);
    }
    if num_tokens > MAX_NUM_TOKENS {
        return Err(InvalidColumn::CodeOutOfRange);
    }
    if offsets.first().copied() != Some(0) {
        return Err(InvalidColumn::FirstOffsetNotZero);
    }

    let starts = &offsets[..num_tokens];
    let ends = &offsets[1..];
    let mut bad_decreasing = 0u32;
    let mut bad_empty = 0u32;
    let mut bad_length = 0u32;

    for (&start, &end) in starts.iter().zip(ends) {
        // Keep the subtraction defined even when an attacker supplies decreasing
        // offsets. `bad_order` is tracked independently because a wrapping
        // difference can happen to look small for some malformed pairs.
        let length = end.wrapping_sub(start);
        bad_decreasing |= (end < start) as u32;
        bad_empty |= (end == start) as u32;
        bad_length |= (length > MAX_TOKEN_SIZE as u32) as u32;
    }

    if bad_decreasing != 0 {
        return Err(InvalidColumn::DecreasingOffsets);
    }
    if bad_empty != 0 {
        // Empty tokens are not safe for the search tokenizer because they would
        // make it fail to advance.
        return Err(InvalidColumn::EmptyToken);
    }
    if bad_length != 0 {
        return Err(InvalidColumn::TokenTooLarge);
    }

    // The start of the last token is the highest token start. Since every token
    // is at most MAX_TOKEN_SIZE bytes, one final padding check covers every fixed
    // width source read and implies that all logical token ends are in bounds.
    let last_start =
        usize::try_from(offsets[num_tokens - 1]).map_err(|_| InvalidColumn::MissingPadding)?;
    let Some(last_read_end) = last_start.checked_add(MAX_TOKEN_SIZE) else {
        return Err(InvalidColumn::MissingPadding);
    };
    if last_read_end > bytes.len() {
        return Err(InvalidColumn::MissingPadding);
    }

    Ok(())
}

/// Validate the dictionary-content invariants required for correct search and
/// tokenization, after [`validate_safety`] has established that every token slice
/// is safe to access.
///
/// The correctness boundary checks three properties:
///
/// * Tokens are in strictly increasing bytewise-lexicographic order. The search
///   tokenizer narrows candidate tokens with binary search, so this ordering is
///   required for it to find the correct longest match.
/// * Tokens are unique. This is checked together with ordering: `prev >= token`
///   rejects both a token that sorts before its predecessor and a token equal to
///   its predecessor. Uniqueness is required so different code sequences cannot
///   decode to the same string and make code-domain equality search incorrect.
/// * All 256 single-byte tokens are present. This guarantees that every byte can
///   be tokenized, including bytes not covered by a multi-byte token. Without the
///   complete alphabet, tokenization can silently fall back to an unrelated token
///   and produce an incorrect query or encoded string.
///
/// None of these checks is required to prevent out-of-bounds access; malformed
/// content remains structurally safe but can produce wrong search results. The
/// pass is therefore deliberately kept off the safety-only path. It reads token
/// bytes because lexicographic comparison is inherently variable-length; the
/// four-word alphabet bitset avoids a separate scan over 256 boolean entries.
fn validate_conformance(bytes: &[u8], offsets: &[u32]) -> Result<(), InvalidColumn> {
    // Sortedness implies uniqueness. A four-word bitset makes the completeness
    // check a small integer reduction rather than a second scan over 256 booleans.
    let mut seen = [0u64; 4];
    let mut prev: &[u8] = &[];
    for (&start, &end) in offsets[..offsets.len() - 1].iter().zip(&offsets[1..]) {
        let token = &bytes[start as usize..end as usize];
        if prev >= token {
            return Err(InvalidColumn::UnsortedTokens);
        }
        if token.len() == 1 {
            let byte = token[0] as usize;
            seen[byte >> 6] |= 1u64 << (byte & 63);
        }
        prev = token;
    }
    if seen != [u64::MAX; 4] {
        return Err(InvalidColumn::IncompleteAlphabet);
    }
    Ok(())
}

/// Minimum code width needed to address `num_tokens` distinct tokens:
/// `ceil(log2(num_tokens))`.
///
/// This is the runtime width of OnPair token codes for a dictionary with
/// `num_tokens` entries. It is independent from the training-time
/// [`MaxDictBits`](crate::MaxDictBits) budget.
pub fn code_bits_for_num_tokens(num_tokens: usize) -> u8 {
    debug_assert!(
        num_tokens >= 1,
        "log2(0) is undefined; num_tokens must be >= 1"
    );
    if num_tokens <= 1 {
        1
    } else {
        ((num_tokens as u32 - 1).ilog2() + 1) as u8
    }
}

/// Owned compact dictionary. Holding one is a proof that its buffers satisfy the
/// structural invariants in this module's documentation.
///
/// The fields are private, so a value can only be obtained through a door that
/// establishes that proof: the trainer, either validation method, or an unsafe
/// unchecked constructor applied to deserialized buffers. Semantic conformance
/// is established separately by [`validate`](Self::validate)
/// when search correctness is required. Read the buffers back with
/// [`bytes`](Self::bytes) / [`offsets`](Self::offsets), or move them out with
/// [`into_raw`](Self::into_raw) (e.g. to serialize).
#[derive(Debug, Clone)]
pub struct CompactDictionary<S = OwnedDictionaryStorage> {
    /// Storage whose buffers satisfy the structural dictionary invariants.
    storage: S,
}

impl<S> CompactDictionary<S>
where
    S: DictionaryStorage<u32>,
{
    /// Number of tokens.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.storage.offsets().len().saturating_sub(1)
    }

    /// The token bytes, including trailing read-padding (the serialized
    /// `dict_bytes`; see `docs/interchange-format.md`).
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        self.storage.bytes()
    }

    /// The `num_tokens + 1` token offsets (the serialized `dict_offsets`).
    #[inline]
    pub fn offsets(&self) -> &[u32] {
        self.storage.offsets()
    }

    /// Borrow the storage backing this dictionary.
    #[inline]
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Consume the dictionary and return its backing storage without copying.
    #[inline]
    pub fn into_storage(self) -> S {
        self.storage
    }

    /// Logical byte length — token bytes only, excluding read-padding.
    #[inline]
    pub fn logical_len(&self) -> usize {
        self.storage.offsets().last().copied().unwrap_or(0) as usize
    }

    /// Minimum bits per code needed to address this dictionary,
    /// `ceil(log2(num_tokens))`. A consumer that bit-packs the code stream packs
    /// each code in this many bits. For a dictionary with at most `2^16` tokens,
    /// the result is in `1..=16`.
    #[inline]
    pub fn code_bits(&self) -> u8 {
        code_bits_for_num_tokens(self.num_tokens())
    }

    /// Materialize the [`WideDictionary`] form (see
    /// [`CompactDictionaryView::to_wide`]). Borrow as a view first with
    /// [`Dictionary::as_view`].
    #[inline]
    pub fn to_wide(&self) -> WideDictionary {
        self.as_view().to_wide()
    }

    /// Validate storage against the structural dictionary invariants without
    /// copying either serialized buffer.
    ///
    /// This is the safety boundary: it checks offsets, token lengths,
    /// addressability, and trailing read-padding, but does not inspect token
    /// contents.
    pub fn validate_safety(storage: S) -> Result<Self, InvalidColumn> {
        validate_safety(storage.bytes(), storage.offsets())?;
        Ok(Self { storage })
    }

    /// Validate storage against the structural and semantic dictionary
    /// invariants without copying either serialized buffer.
    pub fn validate(storage: S) -> Result<Self, InvalidColumn> {
        let dictionary = Self::validate_safety(storage)?;
        dictionary.check_correctness()?;
        Ok(dictionary)
    }

    /// Check the correctness-only invariants of an already safety-validated
    /// dictionary. This avoids repeating the offset scan performed by
    /// [`validate_safety`](Self::validate_safety).
    pub fn check_correctness(&self) -> Result<(), InvalidColumn> {
        validate_conformance(self.storage.bytes(), self.storage.offsets())
    }

    /// Seal storage into a dictionary without checking.
    ///
    /// # Safety
    /// `storage` must satisfy the structural invariants checked by
    /// [`validate_safety`](Self::validate_safety). In particular,
    /// it must be safe for the unchecked decoder, and its returned slices must
    /// remain immutable and stable for the lifetime of the dictionary.
    pub unsafe fn new_unchecked(storage: S) -> Self {
        Self { storage }
    }
}

impl CompactDictionary<OwnedDictionaryStorage> {
    /// Consume the dictionary, returning its owned buffers without copying.
    #[inline]
    pub fn into_raw(self) -> (Vec<u8>, Vec<u32>) {
        self.storage.into_raw()
    }

    /// Seal raw buffers into a safety-trusted dictionary. The crate-internal
    /// trust mint: the caller (trainer, or a validation door) guarantees the
    /// structural safety invariants; the trainer also guarantees conformance.
    #[inline]
    pub(crate) fn from_raw(bytes: Vec<u8>, offsets: Vec<u32>) -> Self {
        Self {
            storage: OwnedDictionaryStorage::new(bytes, offsets),
        }
    }
}

impl<S> Dictionary for CompactDictionary<S>
where
    S: DictionaryStorage<u32>,
{
    type View<'a>
        = CompactDictionaryView<'a>
    where
        S: 'a;
    #[inline]
    fn as_view(&self) -> CompactDictionaryView<'_> {
        CompactDictionaryView {
            bytes: self.storage.bytes(),
            offsets: self.storage.offsets(),
        }
    }
}

/// Borrowed, `Copy` view over a compact dictionary's buffers.
///
/// Like [`CompactDictionary`] its fields are private: a value can only be obtained
/// from an owned dictionary ([`Dictionary::as_view`]) or by validating raw borrowed
/// buffers with [`validate_safety`](Self::validate_safety),
/// [`validate`](Self::validate), or
/// [`new_unchecked`](Self::new_unchecked).
#[derive(Copy, Clone, Debug)]
pub struct CompactDictionaryView<'a> {
    /// Read-padded token bytes.
    bytes: &'a [u8],
    /// `num_tokens + 1` offsets into `bytes`.
    offsets: &'a [u32],
}

impl<'a> CompactDictionaryView<'a> {
    /// Seal raw borrowed buffers into a view (crate-internal trust mint; the
    /// caller guarantees the structural dictionary invariants).
    #[inline]
    pub(crate) fn from_raw(bytes: &'a [u8], offsets: &'a [u32]) -> Self {
        Self { bytes, offsets }
    }

    /// Validate raw borrowed `(bytes, offsets)` for safe decoding over the same
    /// slices (no copy) — the checked door across the safety boundary. The
    /// borrowed bytes must already be read-padded (a borrow cannot be extended).
    pub fn validate_safety(bytes: &'a [u8], offsets: &'a [u32]) -> Result<Self, InvalidColumn> {
        validate_safety(bytes, offsets)?;
        Ok(Self::from_raw(bytes, offsets))
    }

    /// Validate raw borrowed `(bytes, offsets)` for safe decoding and correct
    /// search over the same slices (no copy) — the checked door across the trust
    /// boundary. The borrowed bytes must already be read-padded (a borrow cannot
    /// be extended).
    ///
    /// # Errors
    /// As [`CompactDictionary::validate`].
    pub fn validate(bytes: &'a [u8], offsets: &'a [u32]) -> Result<Self, InvalidColumn> {
        validate_safety(bytes, offsets)?;
        validate_conformance(bytes, offsets)?;
        Ok(Self::from_raw(bytes, offsets))
    }

    /// Check correctness-only invariants on an already safety-validated view.
    pub fn check_correctness(&self) -> Result<(), InvalidColumn> {
        validate_conformance(self.bytes, self.offsets)
    }

    /// Seal raw borrowed `(bytes, offsets)` into a view without checking.
    ///
    /// # Safety
    /// The slices must satisfy the structural invariants checked by
    /// [`validate_safety`](Self::validate_safety), and must remain immutable and
    /// stable for the lifetime of the returned view.
    pub unsafe fn new_unchecked(bytes: &'a [u8], offsets: &'a [u32]) -> Self {
        Self::from_raw(bytes, offsets)
    }

    /// Minimum bits per code needed to address this dictionary,
    /// `ceil(log2(num_tokens))`. See [`CompactDictionary::code_bits`].
    #[inline]
    pub fn code_bits(&self) -> u8 {
        code_bits_for_num_tokens(self.num_tokens())
    }

    /// Materialize the [`WideDictionary`] form: every token laid out in its own
    /// fixed [`MAX_TOKEN_SIZE`]-byte row, so a decode reaches a token at
    /// `code * MAX_TOKEN_SIZE` with no `code → offset → bytes` indirection. Worth
    /// building once to amortize over a bulk or repeated decode; see
    /// [`WideDictionary`] for the space/speed trade-off.
    ///
    /// The source view already satisfies the structural invariants, so this never
    /// validates and never fails. Two invariants carry the build:
    ///
    /// * read-padding lets each row be filled with one fixed 16-byte copy from the
    ///   token's offset — an over-read past the token into neighbouring or padding
    ///   bytes (harmless: decode only ever reads a row's first `lens[id]` bytes),
    ///   kept in bounds by the padding;
    /// * the `≤ MAX_TOKEN_SIZE` length bound makes `lens[id] = len as u8` exact,
    ///   not a silent truncation.
    ///
    /// `O(num_tokens)`, dominated by the row copy.
    ///
    /// When starting from raw [`DictionaryStorage`] rather than an existing
    /// compact view, [`WideDictionary::validate_safety`](super::WideDictionary::validate_safety)
    /// materializes the wide form directly and avoids constructing an intermediate
    /// [`CompactDictionary`].
    pub fn to_wide(&self) -> WideDictionary {
        let n = self.num_tokens();
        let mut data = vec![0u8; n * MAX_TOKEN_SIZE];
        let mut lens = vec![0u8; n];
        let src = self.bytes.as_ptr();
        let dst = data.as_mut_ptr();
        for id in 0..n {
            // SAFETY: the view has `offsets.len() == n + 1`, so `id` and
            // `id + 1` index it in bounds. Offsets are strictly increasing with
            // every token length in `1..=MAX_TOKEN_SIZE`, so `end - off` neither
            // wraps nor overflows the `u8` length.
            let (off, end) = unsafe {
                (
                    *self.offsets.get_unchecked(id) as usize,
                    *self.offsets.get_unchecked(id + 1) as usize,
                )
            };
            // SAFETY: `id < n == lens.len()`.
            unsafe { *lens.get_unchecked_mut(id) = (end - off) as u8 };
            // SAFETY: dst — `(id + 1) * MAX_TOKEN_SIZE <= n * MAX_TOKEN_SIZE == data.len()`.
            // src — the read-padding invariant guarantees `MAX_TOKEN_SIZE` readable
            // bytes at `off`; `src` (borrowed dictionary) and the freshly-allocated
            // `dst` are distinct allocations, so the copy cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.add(off),
                    dst.add(id * MAX_TOKEN_SIZE),
                    MAX_TOKEN_SIZE,
                );
            }
        }
        WideDictionary::from_raw(data, lens)
    }
}

impl DictionaryView for CompactDictionaryView<'_> {
    #[inline]
    fn num_tokens(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    fn token(&self, id: Token) -> &[u8] {
        let begin = self.offsets[id as usize] as usize;
        let end = self.offsets[id as usize + 1] as usize;
        &self.bytes[begin..end]
    }

    #[inline]
    fn token_len(&self, id: Token) -> usize {
        (self.offsets[id as usize + 1] - self.offsets[id as usize]) as usize
    }

    #[inline]
    unsafe fn token_ptr(&self, id: Token) -> *const u8 {
        // SAFETY: id < num_tokens ⇒ offsets[id] is in bounds; the read-padding
        // invariant guarantees MAX_TOKEN_SIZE readable bytes at the offset.
        unsafe {
            self.bytes
                .as_ptr()
                .add(*self.offsets.get_unchecked(id as usize) as usize)
        }
    }

    #[inline]
    unsafe fn token_len_unchecked(&self, id: Token) -> usize {
        // SAFETY: id < num_tokens ⇒ offsets[id] and offsets[id + 1] are in bounds.
        unsafe {
            (*self.offsets.get_unchecked(id as usize + 1)
                - *self.offsets.get_unchecked(id as usize)) as usize
        }
    }
}

impl<'a> From<&'a CompactDictionary> for CompactDictionaryView<'a> {
    #[inline]
    fn from(d: &'a CompactDictionary) -> Self {
        d.as_view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{ContainsTable, PrefixQuery, contains, starts_with, tokenize};
    use crate::{DECODE_PADDING, decode_into, decoded_len, try_decode_into};
    use std::mem::MaybeUninit;
    use std::sync::Arc;

    #[derive(Clone, Debug)]
    struct SharedStorage {
        bytes: Arc<[u8]>,
        offsets: Arc<[u32]>,
    }

    impl DictionaryStorage<u32> for SharedStorage {
        fn bytes(&self) -> &[u8] {
            &self.bytes
        }

        fn offsets(&self) -> &[u32] {
            &self.offsets
        }
    }

    fn dict(offsets: Vec<u32>, bytes: &[u8]) -> CompactDictionary {
        CompactDictionary::from_raw(bytes.to_vec(), offsets)
    }

    #[test]
    fn num_tokens_is_offsets_len_minus_one() {
        assert_eq!(dict(vec![0, 3, 5, 8], b"").num_tokens(), 3);
    }

    #[test]
    fn token_returns_correct_slice() {
        let d = dict(vec![0, 1, 3, 6], b"abcdef");
        let v = d.as_view();
        assert_eq!(v.token(0), b"a");
        assert_eq!(v.token(1), b"bc");
        assert_eq!(v.token(2), b"def");
        assert_eq!(v.token_len(2), 3);
    }

    #[test]
    fn storage_backed_dictionary_does_not_copy_buffers() {
        let (bytes, offsets) = conformant(&[b"bc", b"def"]);
        let bytes: Arc<[u8]> = bytes.into();
        let offsets: Arc<[u32]> = offsets.into();
        let storage = SharedStorage {
            bytes: bytes.clone(),
            offsets: offsets.clone(),
        };

        let dictionary = CompactDictionary::<SharedStorage>::validate(storage).unwrap();
        assert_eq!(dictionary.bytes().as_ptr(), bytes.as_ptr());
        assert_eq!(dictionary.offsets().as_ptr(), offsets.as_ptr());
        assert_eq!(dictionary.as_view().token(0), &[0]);

        let storage = dictionary.into_storage();
        assert!(Arc::ptr_eq(&storage.bytes, &bytes));
        assert!(Arc::ptr_eq(&storage.offsets, &offsets));
    }

    #[test]
    fn code_bits_is_ceil_log2_num_tokens() {
        assert_eq!(dict(vec![0; 257], b"").code_bits(), 8); // 256 tokens -> 8 bits
        assert_eq!(dict(vec![0; 258], b"").code_bits(), 9); // 257 tokens -> 9 bits
        assert_eq!(dict(vec![0; 513], b"").code_bits(), 9); // 512 tokens -> 9 bits
        assert_eq!(dict(vec![0; 514], b"").code_bits(), 10); // 513 tokens -> 10 bits
        assert_eq!(dict(vec![0; 65_537], b"").code_bits(), 16); // 65536 tokens -> 16 bits
    }

    #[test]
    fn pad_raw_extends_to_max_token_read() {
        // Last token "bc" is 2 bytes; padding fills to offset(last) + MAX_TOKEN_SIZE.
        let mut bytes = b"abc".to_vec();
        pad_raw(&mut bytes, &[0, 1, 3]);
        assert_eq!(bytes.len(), 1 + MAX_TOKEN_SIZE); // offset(last)=1, +16
    }

    #[test]
    fn pad_raw_is_idempotent() {
        let mut bytes = b"abc".to_vec();
        let offsets = [0u32, 1, 3];
        pad_raw(&mut bytes, &offsets);
        let len = bytes.len();
        pad_raw(&mut bytes, &offsets);
        assert_eq!(bytes.len(), len);
    }

    #[test]
    fn pad_raw_tops_up_insufficient_trailing_bytes() {
        // bytes already exceed logical_len (3) but lack room for a full
        // MAX_TOKEN_SIZE read from the last token's start (offset 1).
        let mut bytes = vec![b'a', b'b', b'c', 0];
        pad_raw(&mut bytes, &[0, 1, 3]);
        assert_eq!(bytes.len(), 1 + MAX_TOKEN_SIZE);
    }

    #[test]
    fn pad_raw_noop_for_full_width_last_token() {
        let mut bytes = vec![b'z'; MAX_TOKEN_SIZE];
        pad_raw(&mut bytes, &[0, MAX_TOKEN_SIZE as u32]);
        assert_eq!(bytes.len(), MAX_TOKEN_SIZE);
    }

    /// Read-padded `(bytes, offsets)` from an explicit token list — the caller
    /// controls exactly which conformance property is under test.
    fn padded(tokens: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for t in tokens {
            bytes.extend_from_slice(t);
            offsets.push(bytes.len() as u32);
        }
        bytes.resize(bytes.len() + MAX_TOKEN_SIZE, 0); // worst-case read padding
        (bytes, offsets)
    }

    /// Exercise every operation that a safety-valid dictionary must support
    /// without relying on semantic conformance. The expected bytes come from
    /// the dictionary itself: malformed ordering or alphabet coverage may make
    /// search results wrong, but must not make any access leave its bounds.
    fn assert_safe_use(tokens: &[&[u8]], text: &[u8], codes: &[Token]) {
        let (bytes, offsets) = padded(tokens);
        let dictionary =
            CompactDictionary::validate_safety(OwnedDictionaryStorage::new(bytes, offsets))
                .unwrap();
        let view = dictionary.as_view();

        // An incomplete or unsorted dictionary may choose the wrong token, but
        // every chosen token is non-empty, so tokenization still makes progress
        // and consumes exactly the input rather than looping on an empty token.
        let tokenized = tokenize(text, view);
        assert_eq!(
            tokenized
                .iter()
                .map(|&code| view.token_len(code))
                .sum::<usize>(),
            text.len()
        );

        // Exercise search construction and execution too. These operations have
        // a conformance precondition for correct answers, but safety validation
        // must still make malformed contents bounded to inspect.
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

        // The unchecked fast path performs its fixed-width source reads and
        // destination over-stores. The validated dictionary and this padding
        // together prove that those accesses stay in bounds.
        let mut padded_out = Vec::with_capacity(decoded_len + DECODE_PADDING);
        let written = unsafe { decode_into(codes, view, padded_out.spare_capacity_mut()) };
        unsafe { padded_out.set_len(written) };
        assert_eq!(padded_out, expected);

        // The checked path must also succeed with an exactly-sized destination,
        // proving that its exact-copy tail never writes beyond the caller's
        // buffer for a safety-valid dictionary.
        let mut exact_out = vec![MaybeUninit::uninit(); decoded_len];
        let written = try_decode_into(codes, view, &mut exact_out).unwrap();
        let exact_bytes =
            unsafe { std::slice::from_raw_parts(exact_out.as_ptr().cast::<u8>(), written) };
        assert_eq!(exact_bytes, expected.as_slice());
    }

    /// A conformant dictionary: all 256 single-byte tokens plus `extra`, sorted and
    /// deduped (so strictly increasing, sorted, unique, complete, and padded).
    fn conformant(extra: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut toks: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
        for &t in extra {
            toks.push(t.to_vec());
        }
        toks.sort();
        toks.dedup();
        let refs: Vec<&[u8]> = toks.iter().map(Vec::as_slice).collect();
        padded(&refs)
    }

    /// A conformant dictionary with exactly `num_tokens` entries. Multi-byte
    /// entries are generated as unique two-byte sequences, then sorted together
    /// with the complete single-byte alphabet.
    fn conformant_with_num_tokens(num_tokens: usize) -> (Vec<u8>, Vec<u32>) {
        assert!((256..=256 + u16::MAX as usize + 1).contains(&num_tokens));
        let mut toks: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
        toks.extend((0..num_tokens - 256).map(|value| (value as u16).to_be_bytes().to_vec()));
        toks.sort();
        let refs: Vec<&[u8]> = toks.iter().map(Vec::as_slice).collect();
        padded(&refs)
    }

    /// Map to `Result<(), _>` so we can compare (`CompactDictionary` isn't `Eq`).
    fn check(bytes: Vec<u8>, offsets: Vec<u32>) -> Result<(), InvalidColumn> {
        CompactDictionary::validate(OwnedDictionaryStorage::new(bytes, offsets)).map(|_| ())
    }

    #[test]
    fn validate_accepts_conformant() {
        let (bytes, offsets) = conformant(&[b"bc", b"def"]);
        assert_eq!(check(bytes, offsets), Ok(()));
    }

    #[test]
    fn validate_enforces_token_address_space() {
        let (bytes, offsets) = conformant_with_num_tokens(MAX_NUM_TOKENS);
        assert_eq!(check(bytes, offsets), Ok(()));

        let (bytes, offsets) = conformant_with_num_tokens(MAX_NUM_TOKENS + 1);
        assert_eq!(check(bytes, offsets), Err(InvalidColumn::CodeOutOfRange));
    }

    #[test]
    fn validate_classifies_safety_corruption() {
        // The first offset must point at the beginning of the byte buffer.
        assert_eq!(
            check(vec![0u8; MAX_TOKEN_SIZE + 1], vec![1, 2]),
            Err(InvalidColumn::FirstOffsetNotZero)
        );

        // Decreasing offsets (would underflow the length subtraction).
        let mut bytes = b"ab".to_vec();
        bytes.resize(2 + MAX_TOKEN_SIZE, 0);
        assert_eq!(
            check(bytes, vec![0, 2, 1]),
            Err(InvalidColumn::DecreasingOffsets)
        );

        // Zero-length token (`e == s`).
        assert_eq!(
            check(vec![0u8; MAX_TOKEN_SIZE], vec![0, 0]),
            Err(InvalidColumn::EmptyToken)
        );

        // Token longer than MAX_TOKEN_SIZE.
        assert_eq!(
            check(vec![b'x'; 20 + MAX_TOKEN_SIZE], vec![0, 20]),
            Err(InvalidColumn::TokenTooLarge)
        );

        // Missing the trailing read-padding.
        assert_eq!(
            check(b"abc".to_vec(), vec![0, 1, 3]),
            Err(InvalidColumn::MissingPadding)
        );
    }

    #[test]
    fn validate_classifies_conformance_corruption() {
        // Safe + padded but out of order (also how a duplicate would surface).
        let (bytes, offsets) = padded(&[&[1u8], &[0u8]]);
        assert_eq!(check(bytes, offsets), Err(InvalidColumn::UnsortedTokens));

        // Sorted + safe but missing all but three single-byte tokens.
        let (bytes, offsets) = padded(&[&[0u8], &[1u8], &[2u8]]);
        assert_eq!(
            check(bytes, offsets),
            Err(InvalidColumn::IncompleteAlphabet)
        );
    }

    #[test]
    fn validate_safety_skips_conformance_checks() {
        // Safe + padded, but unsorted and incomplete. The safety path intentionally
        // accepts both because neither property is needed for bounded access.
        let (bytes, offsets) = padded(&[&[1u8], &[0u8]]);
        let dictionary =
            CompactDictionary::validate_safety(OwnedDictionaryStorage::new(bytes, offsets))
                .unwrap();
        assert_eq!(
            dictionary.check_correctness(),
            Err(InvalidColumn::UnsortedTokens)
        );

        let (bytes, offsets) = padded(&[&[0u8], &[1u8], &[2u8]]);
        let dictionary =
            CompactDictionary::validate_safety(OwnedDictionaryStorage::new(bytes, offsets))
                .unwrap();
        assert_eq!(
            dictionary.check_correctness(),
            Err(InvalidColumn::IncompleteAlphabet)
        );
    }

    #[test]
    fn validate_safety_rejects_empty_dictionary() {
        assert_eq!(
            CompactDictionary::validate_safety(OwnedDictionaryStorage::new(
                vec![0; MAX_TOKEN_SIZE],
                vec![0],
            ))
            .map(|_| ()),
            Err(InvalidColumn::EmptyDictionary)
        );
    }

    #[test]
    fn validate_safety_rejects_structural_corruption() {
        let cases = [
            (
                vec![0u8; MAX_TOKEN_SIZE + 1],
                vec![1, 2],
                InvalidColumn::FirstOffsetNotZero,
            ),
            (
                vec![0u8; MAX_TOKEN_SIZE + 2],
                vec![0, 2, 1],
                InvalidColumn::DecreasingOffsets,
            ),
            (
                vec![0u8; MAX_TOKEN_SIZE],
                vec![0, 0],
                InvalidColumn::EmptyToken,
            ),
            (
                vec![0u8; MAX_TOKEN_SIZE + 17],
                vec![0, 17],
                InvalidColumn::TokenTooLarge,
            ),
            (
                b"abc".to_vec(),
                vec![0, 1, 3],
                InvalidColumn::MissingPadding,
            ),
        ];

        for (bytes, offsets, expected) in cases {
            assert_eq!(
                CompactDictionary::validate_safety(OwnedDictionaryStorage::new(bytes, offsets))
                    .map(|_| ()),
                Err(expected)
            );
        }
    }

    #[test]
    fn safety_valid_semantically_malformed_dictionary_remains_safe_to_use() {
        // Sortedness, uniqueness, and alphabet completeness are correctness
        // guarantees. Their absence is intentionally observable as an error from
        // `check_correctness`, but it must not make decoding or search unsafe.
        let (bytes, offsets) = padded(&[&[1u8], &[0u8]]);
        let dictionary =
            CompactDictionary::validate_safety(OwnedDictionaryStorage::new(bytes, offsets))
                .unwrap();
        assert_eq!(
            dictionary.check_correctness(),
            Err(InvalidColumn::UnsortedTokens)
        );
        drop(dictionary);
        assert_safe_use(&[&[1u8], &[0u8]], b"xyz", &[0, 1, 0, 1]);

        let (bytes, offsets) = padded(&[&[0u8], &[1u8], &[2u8]]);
        let dictionary =
            CompactDictionary::validate_safety(OwnedDictionaryStorage::new(bytes, offsets))
                .unwrap();
        assert_eq!(
            dictionary.check_correctness(),
            Err(InvalidColumn::IncompleteAlphabet)
        );
        drop(dictionary);
        assert_safe_use(&[&[0u8], &[1u8], &[2u8]], b"xyz", &[2, 0, 1, 2]);

        let (bytes, offsets) = padded(&[&[0u8], &[0u8]]);
        let dictionary =
            CompactDictionary::validate_safety(OwnedDictionaryStorage::new(bytes, offsets))
                .unwrap();
        assert_eq!(
            dictionary.check_correctness(),
            Err(InvalidColumn::UnsortedTokens)
        );
        drop(dictionary);
        assert_safe_use(&[&[0u8], &[0u8]], b"xyz", &[0, 1, 0]);
    }

    #[test]
    fn new_unchecked_matches_validate() {
        let (bytes, offsets) = conformant(&[b"bc"]);
        let checked = CompactDictionary::validate(OwnedDictionaryStorage::new(
            bytes.clone(),
            offsets.clone(),
        ))
        .unwrap();
        // SAFETY: `conformant` produces a conformant dictionary.
        let trusted = unsafe {
            CompactDictionary::new_unchecked(OwnedDictionaryStorage::new(bytes, offsets))
        };
        assert_eq!(checked.bytes(), trusted.bytes());
        assert_eq!(checked.offsets(), trusted.offsets());
    }

    #[test]
    fn into_raw_returns_buffers_and_round_trips() {
        let (bytes, offsets) = conformant(&[b"bc", b"def"]);
        let num_tokens = offsets.len() - 1;
        let dict = CompactDictionary::validate(OwnedDictionaryStorage::new(
            bytes.clone(),
            offsets.clone(),
        ))
        .unwrap();

        // The owned buffers come back byte-for-byte, read-padding included.
        let (raw_bytes, raw_offsets) = dict.into_raw();
        assert_eq!(raw_bytes, bytes);
        assert_eq!(raw_offsets, offsets);

        // ...and rebuild into an equivalent trusted dictionary.
        let rebuilt =
            CompactDictionary::validate(OwnedDictionaryStorage::new(raw_bytes, raw_offsets))
                .unwrap();
        assert_eq!(rebuilt.num_tokens(), num_tokens);
    }

    #[test]
    fn view_validate_yields_usable_view() {
        let (bytes, offsets) = conformant(&[b"bc"]);
        let view = CompactDictionaryView::validate(&bytes, &offsets).unwrap();
        assert_eq!(view.num_tokens(), 257); // 256 single bytes + "bc"
        assert_eq!(view.token(0), &[0u8]); // byte 0 sorts first

        // A non-conformant borrow is rejected.
        let raw: &[u8] = b"abc";
        assert!(CompactDictionaryView::validate(raw, &[0, 1, 3]).is_err());
    }
}
