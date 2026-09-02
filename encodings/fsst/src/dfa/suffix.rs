// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tail matcher for suffix matching (`LIKE '%suffix'`) on FSST codes.
//!
//! Prefix and contains matching win by exiting early: a prefix decides within the first
//! few codes, and contains stops at the first hit. A forward suffix scan has no such exit
//! — the answer depends on the last byte, so every code of every row gets touched. That
//! is the wrong asymptotics to bring against a vectorized decompress-and-compare, and it
//! measured slower.
//!
//! So this walks the code stream *backward* from the end. The array stores per-row
//! offsets, so each row's end is already known, and the first byte compared is the row's
//! last byte — a non-matching row is rejected after a single symbol.
//!
//! Backward walking needs token boundaries, which the forward escape mechanism appears to
//! hide. It does not, and only locally: ambiguity propagates through a run of consecutive
//! [`ESCAPE_CODE`] bytes and stops at the first byte that is not one. Given a known token
//! boundary `p`, let `r` be the number of consecutive `ESCAPE_CODE` bytes immediately left
//! of `codes[p - 1]`:
//!
//! * `r` odd — `codes[p - 2]` is an escape marker, so `codes[p - 1]` is a literal byte.
//! * `r` even — `codes[p - 1]` is itself a symbol code.
//!
//! This holds because the first non-`ESCAPE_CODE` byte left of the run is either a symbol
//! code or an escaped literal, and in both cases the position after it is a token
//! boundary; the run in between pairs off into two-byte escaped-`0xFF` tokens. Real text
//! contains no `0xFF`, so `r` is 0 or 1 and each step is `O(1)`.
//!
//! Two precomputations then remove the per-row byte work:
//!
//! * `tail_step` answers "can a row ending in this symbol match, and how much of the
//!   suffix does it cover" per code, so the common rejection is one byte lookup that never
//!   reads the symbol's bytes.
//! * A symbol holds at most 8 bytes, so a whole symbol fits in a `u64`. Storing both the
//!   symbols and the suffix *reversed* aligns the bytes each step has to compare at the low
//!   end of a word, turning the comparison into `(a ^ b) & mask`. That replaces a
//!   variable-length slice comparison with three ALU ops, which is what the walk past the
//!   first symbol costs.

use fsst::ESCAPE_CODE;
use fsst::Symbol;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

/// A `tail_step` entry meaning no string ending in that symbol can match.
///
/// Real entries are `min(symbol_len, suffix_len)`, which a symbol's 8-byte cap keeps
/// well below this.
const REJECT: u8 = u8::MAX;

/// Number of byte values that can be a symbol code: [`ESCAPE_CODE`] is the one that cannot.
///
/// Every per-code table is sized to this so a code past the symbol table, which only a
/// corrupt file produces, reads padding instead of indexing out of bounds — the same way
/// the prefix and contains DFAs absorb it in their 256-wide tables.
const CODE_SPACE: usize = ESCAPE_CODE as usize;

/// Low-`n`-byte masks, indexed by how many bytes a step compares.
const TAKE_MASK: [u64; 9] = [
    0x0000_0000_0000_0000,
    0x0000_0000_0000_00ff,
    0x0000_0000_0000_ffff,
    0x0000_0000_00ff_ffff,
    0x0000_0000_ffff_ffff,
    0x0000_00ff_ffff_ffff,
    0x0000_ffff_ffff_ffff,
    0x00ff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
];

/// One FSST token, resolved backward from the end of a code stream.
enum Token {
    /// An escaped byte, standing for itself.
    Literal(u8),
    /// A symbol table code.
    Code(u8),
}

/// Matches FSST-compressed strings that end with a fixed byte string.
pub(crate) struct SuffixMatcher {
    suffix_len: usize,
    /// Decoded byte length of each symbol, indexed by code, `0` past the table.
    symbol_lengths: Vec<u8>,
    /// Each symbol's decoded bytes, last byte first, packed little-endian.
    symbol_rev: Vec<u64>,
    /// For each count of still-unmatched suffix bytes, those bytes last-first.
    suffix_rev: Vec<u64>,
    /// Suffix bytes accounted for by a row's final symbol, or [`REJECT`].
    tail_step: Vec<u8>,
}

/// Packs up to the last 8 of `bytes[..len]` into a `u64`, last byte in the low position.
fn pack_rev(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .rev()
        .take(8)
        .enumerate()
        .fold(0u64, |w, (j, &b)| w | (u64::from(b) << (8 * j)))
}

impl SuffixMatcher {
    /// The needle is compared byte-wise; this bound only keeps it in step with the
    /// other matchers.
    pub(crate) const MAX_SUFFIX_LEN: usize = u8::MAX as usize - 1;

    pub(crate) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        suffix: &[u8],
    ) -> VortexResult<Self> {
        if suffix.len() > Self::MAX_SUFFIX_LEN {
            vortex_bail!(
                "suffix length {} exceeds maximum {} for suffix matching",
                suffix.len(),
                Self::MAX_SUFFIX_LEN
            );
        }

        let decoded: Vec<[u8; 8]> = symbols.iter().map(|s| s.to_u64().to_le_bytes()).collect();
        let bytes_of = |code: usize| match (decoded.get(code), symbol_lengths.get(code)) {
            // A length over 8 cannot describe a symbol, so clamp rather than slice past it.
            (Some(bytes), Some(&len)) => &bytes[..usize::from(len).min(8)],
            _ => &[][..],
        };

        let mut symbol_lengths = symbol_lengths.to_vec();
        symbol_lengths.resize(CODE_SPACE, 0);

        Ok(Self {
            suffix_len: suffix.len(),
            symbol_lengths,
            symbol_rev: (0..CODE_SPACE).map(|c| pack_rev(bytes_of(c))).collect(),
            suffix_rev: (0..=suffix.len()).map(|r| pack_rev(&suffix[..r])).collect(),
            tail_step: (0..CODE_SPACE)
                .map(|c| Self::tail_step_for(bytes_of(c), suffix))
                .collect(),
        })
    }

    /// How much of `suffix` a row ending in `symbol` accounts for, or [`REJECT`].
    ///
    /// A symbol at least as long as the suffix settles the match on its own; a shorter
    /// one has to be the suffix's own tail, and leaves the rest to earlier tokens.
    fn tail_step_for(symbol: &[u8], suffix: &[u8]) -> u8 {
        if symbol.is_empty() {
            // Padding past the symbol table, or a zero-length symbol a valid file cannot
            // hold. Either way no row ending there matches.
            return REJECT;
        }
        // A symbol holds at most 8 bytes, so the overlap is always well below `REJECT`
        // and this conversion cannot fail.
        let Ok(overlap) = u8::try_from(symbol.len().min(suffix.len())) else {
            return REJECT;
        };
        let n = usize::from(overlap);
        if symbol[symbol.len() - n..] == suffix[suffix.len() - n..] {
            overlap
        } else {
            REJECT
        }
    }

    /// The token ending at `p`, and the boundary that precedes it.
    ///
    /// `p` must be a token boundary. `None` means the stream is truncated: an escape
    /// marker with no byte after it.
    ///
    /// The escape-run scan is `O(r)` for a run of `r` consecutive `ESCAPE_CODE` bytes,
    /// which only a row built from literal `0xFF` bytes can make long.
    #[inline]
    fn token_before(codes: &[u8], p: usize) -> Option<(Token, usize)> {
        let last = codes[p - 1];

        let mut escapes = 0usize;
        let mut q = p - 1;
        while q > 0 && codes[q - 1] == ESCAPE_CODE {
            escapes += 1;
            q -= 1;
        }

        if escapes % 2 == 1 {
            // `codes[p - 2]` escapes it, so the byte stands for itself.
            Some((Token::Literal(last), p - 2))
        } else if last == ESCAPE_CODE {
            None
        } else {
            Some((Token::Code(last), p - 1))
        }
    }

    /// A token's decoded bytes, last byte first, and how many of them there are.
    #[inline]
    fn rev_word(&self, token: &Token) -> (u64, usize) {
        match *token {
            Token::Literal(byte) => (u64::from(byte), 1),
            Token::Code(code) => (
                self.symbol_rev[usize::from(code)],
                usize::from(self.symbol_lengths[usize::from(code)]),
            ),
        }
    }

    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        let k = self.suffix_len;
        if k == 0 {
            return true;
        }
        if codes.is_empty() {
            return false;
        }

        // The final token decides most rows, and for a symbol `tail_step` decides it from
        // the code alone.
        let Some((token, next)) = Self::token_before(codes, codes.len()) else {
            return false;
        };
        let step = match token {
            Token::Code(code) => self.tail_step[usize::from(code)],
            Token::Literal(byte) if u64::from(byte) == (self.suffix_rev[k] & 0xff) => 1,
            Token::Literal(_) => REJECT,
        };
        if step == REJECT {
            return false;
        }

        // Anything left of the suffix reaches back over earlier tokens. A symbol may
        // reach past the suffix's start, in which case only its last `take` bytes are
        // compared and the walk is done.
        let mut remaining = k - usize::from(step);
        let mut pos = next;
        while remaining > 0 {
            if pos == 0 {
                // The decoded string is shorter than the suffix.
                return false;
            }
            let Some((token, next)) = Self::token_before(codes, pos) else {
                return false;
            };
            let (word, len) = self.rev_word(&token);
            let take = len.min(remaining);
            if take == 0 {
                // Only a padded code past the symbol table is zero-length, and it carries
                // none of the suffix.
                return false;
            }
            if (word ^ self.suffix_rev[remaining]) & TAKE_MASK[take] != 0 {
                return false;
            }
            remaining -= take;
            pos = next;
        }

        true
    }
}
