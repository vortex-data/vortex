// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The one vocabulary for "these compressed buffers are malformed", plus the
//! panic helper that surfaces it.
//!
//! A [`ColumnView`](crate::ColumnView) / dictionary view is built from buffers a
//! consumer deserialized from storage, so its arrays may be corrupt.
//! [`InvalidColumn`] enumerates safety violations that would otherwise make a
//! decoder read or write out of bounds, plus conformance violations that can make
//! search or tokenization return wrong answers. It is surfaced through
//! recoverable validation and infallible operations:
//!
//! * **Recoverable validation** — the validation family returns a `Result` for
//!   buffers a consumer deserialized from storage.
//!   * Dictionary validation has two levels:
//!     * [`CompactDictionary::validate_safety`](crate::CompactDictionary::validate_safety)
//!       is the cheap safety trust boundary: it seals raw `(bytes, offsets)` into
//!       a dictionary that the unchecked decoder can read safely, without
//!       inspecting token contents.
//!     * [`CompactDictionary::validate`](crate::CompactDictionary::validate)
//!       additionally checks sortedness, uniqueness, and alphabet completeness for
//!       semantically correct search and tokenization.
//! * **Infallible operations** — operations that are infallible by signature (the
//!   convenience
//!   decoders, the per-row/per-code guards) panic on malformed data through
//!   [`panic_malformed`], with a message derived from `InvalidColumn`'s `Display`.
//!
//! Bad *arguments* to the fallible encoding API (`compress`/`train`/`parse`) are a
//! separate domain ([`Error`](crate::Error)).

/// A violation found while validating compressed buffers.
///
/// Two kinds. **Safety** violations would let an unchecked decoder read or write
/// out of bounds, or prevent a search tokenizer from making progress — these are
/// exactly the obligations an unsafe safety-level constructor must uphold. A
/// safety-valid dictionary can still be semantically malformed. **Conformance**
/// violations decode safely but make search/tokenize give wrong answers.
/// `validate_safety` checks only the former; `validate` checks both.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum InvalidColumn {
    // ── Safety / addressability ──────────────────────────────────────────────
    /// The dictionary has no token, so it cannot be used by the search tokenizer.
    EmptyDictionary,
    /// The first dictionary offset is not zero.
    FirstOffsetNotZero,
    /// Dictionary offsets decrease (`offsets[i] > offsets[i + 1]`), which would
    /// underflow the unchecked token-length subtraction.
    DecreasingOffsets,
    /// A dictionary token is longer than [`MAX_TOKEN_SIZE`](crate::MAX_TOKEN_SIZE).
    TokenTooLarge,
    /// A token offset has fewer than [`MAX_TOKEN_SIZE`](crate::MAX_TOKEN_SIZE)
    /// readable bytes after it, so the decoder's fixed-width read runs off the end.
    MissingPadding,
    /// The dictionary has more than `2^16` entries, or a code does not index the
    /// dictionary (`code >= num_tokens`). In either case, the `u16` token/code
    /// type cannot address the requested entry.
    CodeOutOfRange,
    /// Row offsets are not non-decreasing, or the last exceeds the code count.
    BadRowOffsets,
    /// The column's tokens sum to more than `usize::MAX` decoded bytes, so the
    /// decoded-length computation overflows and would under-size the output buffer.
    DecodedLenOverflow,
    /// A dictionary token has zero length (offsets are not strictly increasing),
    /// so the search tokenizer would not make progress.
    EmptyToken,
    // ── Conformance: decodes safely, but search / tokenize give wrong answers ──
    /// Dictionary tokens are not in strictly ascending bytewise order, so they are
    /// not sorted (binary search breaks) or not unique.
    UnsortedTokens,
    /// The dictionary lacks one or more of the 256 single-byte tokens, so some
    /// inputs are not encodable.
    IncompleteAlphabet,
}

impl std::fmt::Display for InvalidColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::EmptyDictionary => "dictionary must contain at least one token",
            Self::FirstOffsetNotZero => "dictionary offsets must start at zero",
            Self::DecreasingOffsets => "dictionary offsets must be strictly increasing",
            Self::TokenTooLarge => "dictionary token exceeds MAX_TOKEN_SIZE",
            Self::MissingPadding => "dictionary bytes lack the required trailing decoder padding",
            Self::CodeOutOfRange => "code index out of range for dictionary",
            Self::BadRowOffsets => "row offsets must be non-decreasing and within the code stream",
            Self::DecodedLenOverflow => "column decodes to more than usize::MAX bytes",
            Self::EmptyToken => "dictionary contains an empty token",
            Self::UnsortedTokens => "dictionary tokens must be sorted and unique",
            Self::IncompleteAlphabet => "dictionary is missing one or more single-byte tokens",
        })
    }
}

impl std::error::Error for InvalidColumn {}

/// Panic for a malformed column/dictionary, message derived from
/// `InvalidColumn`'s `Display`. `#[cold]` + `#[inline(never)]` so a caller's
/// guard is laid out as a never-taken branch.
#[cold]
#[inline(never)]
pub(crate) fn panic_malformed(e: InvalidColumn) -> ! {
    panic!("onpair: {e}")
}
