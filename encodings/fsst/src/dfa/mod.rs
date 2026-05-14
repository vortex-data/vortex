// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! # FSST LIKE Pushdown via DFA Construction
//!
//! This module implements DFA-based pattern matching directly on FSST-compressed
//! strings, without decompressing them. It handles three pattern shapes:
//!
//! - **Prefix**: `'prefix%'`  — matches strings starting with a literal prefix.
//! - **Suffix**: `'%suffix'`  — matches strings ending with a literal suffix.
//! - **Contains**: `'%needle%'` — matches strings containing a literal substring.
//! - **Multi-Contains**: `'%seg1%seg2%...%segN%'` — matches strings containing
//!   multiple literal substrings in order (see [`multi_contains`]).
//!
//! Pushdown is intentionally conservative. If the pattern shape is unsupported,
//! or if the pattern exceeds the DFA's representable state space, construction
//! returns `None` and the caller must fall back to ordinary decompression-based
//! LIKE evaluation.
//!
//! ## Background: FSST Encoding
//!
//! [FSST](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf) compresses strings by
//! replacing frequent byte sequences with single-byte **symbol codes** (0–254). Code
//! byte 255 is reserved as the **escape code**: the next byte is a literal (uncompressed)
//! byte. So a compressed string is a stream of:
//!
//! ```text
//! [symbol_code] ... [symbol_code] [ESCAPE literal_byte] [symbol_code] ...
//! ```
//!
//! A single symbol can expand to 1–8 bytes. Matching on compressed codes requires
//! the DFA to handle multi-byte symbol expansions and the escape mechanism.
//!
//! ## The Algorithm: KMP → Byte Table → Symbol Table → Flat DFA
//!
//! Construction proceeds through four stages:
//!
//! ### Stage 1: KMP Failure Function
//!
//! We compute the standard [KMP](https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm)
//! failure function for the needle bytes. This tells us, on a mismatch at
//! position `i`, the longest proper prefix of `needle[0..i]` that is also a
//! suffix — i.e., where to resume matching instead of starting over.
//!
//! ```text
//! Needle: "abcabd"
//! Failure: [0, 0, 0, 1, 2, 0]
//!                      ^  ^
//!                      At position 3 ('a'), the prefix "a" matches suffix "a"
//!                      At position 4 ('b'), the prefix "ab" matches suffix "ab"
//! ```
//!
//! ### Stage 2: Byte-Level Transition Table
//!
//! From the failure function, we build a full `(state × byte) → state` transition
//! table. State `i` means "we have matched `needle[0..i]`". State `n` (= needle
//! length) is the **accept** state.
//!
//! ```text
//! Needle: "aba"  (3 states + accept)
//!
//!         Input byte
//! State   'a'    'b'    other
//! ─────   ────   ────   ─────
//!   0      1      0      0      ← looking for first 'a'
//!   1      1      2      0      ← matched "a", want 'b'
//!   2      3✓     0      0      ← matched "ab", want 'a'
//!   3✓     3✓     3✓     3✓     ← accept (sticky)
//! ```
//!
//! For prefix matching, a mismatch at any state goes to a **fail** state (no
//! fallback). For contains matching, mismatches follow KMP fallback transitions
//! so we can find the needle anywhere in the string.
//!
//! ### Stage 3: Symbol-Level Transition Table
//!
//! FSST symbols can be multi-byte. To compute the transition for symbol code `c`
//! in state `s`, we simulate feeding each byte of the symbol through the byte
//! table:
//!
//! ```text
//! Symbol #42 = "the" (3 bytes)
//! State 0 + 't' → 0, + 'h' → 0, + 'e' → 0  ⟹ sym_trans[0][42] = 0
//!
//! If needle = "them":
//! State 0 + 't' → 1, + 'h' → 2, + 'e' → 3  ⟹ sym_trans[0][42] = 3
//! ```
//!
//! We then build a **fused 256-wide table**: for code bytes 0–254, use the
//! symbol transition; for code byte 255 (ESCAPE_CODE), transition to a
//! special sentinel that tells the scanner to read the next literal byte.
//!
//! ### Stage 4: Flat `u8` Table
//!
//! The fused table is stored as a flat `Vec<u8>` indexed as
//! `transitions[state * 256 + byte]`. Both the prefix and contains DFAs use
//! escape-sentinel handling: when the scanner sees the sentinel value, it reads
//! the next byte from a separate byte-level escape table.
//!
//! For needles ≤ 127 bytes the contains DFA uses an **escape-folded** variant
//! (see [`folded_contains`]) that encodes the post-escape "expecting a literal
//! byte" status into the state space, removing the sentinel branch from the
//! inner loop entirely. Longer needles (128–254 bytes) fall back to the plain
//! [`flat_contains`] DFA.
//!
//! ## State-Space Limits
//!
//! The public behavior is shaped by two implementation limits, both measured in
//! pattern **bytes** rather than Unicode scalar values:
//!
//! - `prefix%` pushdown is limited to **253 bytes**. The flat prefix DFA uses
//!   `u8` state ids and needs room for progress states, an accept state, a
//!   fail state, and one escape sentinel (N+3 ≤ 256).
//! - `%suffix` pushdown is limited to **254 bytes**. The suffix DFA stores
//!   states in `u8`, needing room for progress states, the accept state, and
//!   one escape sentinel (N+2 ≤ 256).
//! - `%needle%` pushdown is limited to **254 bytes**. The contains DFA stores
//!   states in `u8`, so it needs room for every match-progress state plus both
//!   the accept state and the escape sentinel.
//! - `%seg1%seg2%...%segN%` pushdown is limited to **254 bytes total** across
//!   all segments. The multi-contains DFA chains segment states linearly.
//!
//! Patterns beyond those limits are still valid LIKE patterns; they simply do
//! not use FSST pushdown and must be evaluated through the fallback path.

mod anchor_scan;
mod fat_teddy;
mod flat_contains;
mod folded_contains;
mod multi_contains;
mod prefix;
mod shift_or;
mod skip;
mod suffix;
#[cfg(test)]
mod tests;

use flat_contains::FlatContainsDfa;
use folded_contains::FoldedContainsDfa;
use fsst::ESCAPE_CODE;
use fsst::Symbol;
use multi_contains::MultiContainsDfa;
use prefix::FlatPrefixDfa;
use shift_or::ShiftOrDfa;
use suffix::SuffixMatcher;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

// ---------------------------------------------------------------------------
// FsstMatcher — unified public API
// ---------------------------------------------------------------------------

/// A compiled matcher for LIKE patterns on FSST-compressed strings.
///
/// Encapsulates pattern parsing and DFA variant selection. Returns `None` from
/// [`try_new`](Self::try_new) for patterns that cannot be evaluated without
/// decompression (e.g., `_` wildcards, multiple `%` in non-standard positions,
/// or patterns that exceed the DFA's representable byte-length limits).
pub struct FsstMatcher {
    inner: MatcherInner,
}

enum MatcherInner {
    MatchAll,
    Prefix(FlatPrefixDfa),
    Suffix(SuffixMatcher),
    /// Bit-parallel Shift-Or matcher for short needles (≤ 8 bytes) in the
    /// no-SSA regime. One ALU op per code byte replaces the Teddy +
    /// verifier dispatch and wins on very short selective needles.
    ShiftOr(ShiftOrDfa),
    /// Escape-folded DFA for short needles (`<= 127` bytes). Eliminates the
    /// per-byte sentinel branch from the inner loop.
    FoldedContains(FoldedContainsDfa),
    /// Plain flat DFA for needles in `128..=254` bytes.
    Contains(FlatContainsDfa),
    MultiContains(Box<MultiContainsDfa>),
}

impl FsstMatcher {
    /// Try to build a matcher for the given LIKE pattern.
    ///
    /// Returns `Ok(None)` if the pattern shape is not supported for pushdown
    /// (e.g. `_` wildcards, multiple non-bookend `%`, `prefix%` longer than
    /// 253 bytes, or `%needle%` longer than 254 bytes).
    pub fn try_new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        pattern: &[u8],
    ) -> VortexResult<Option<Self>> {
        Self::try_new_with(symbols, symbol_lengths, pattern, false)
    }

    /// Variant of [`Self::try_new`] that opts in to ASCII case-insensitive
    /// matching (SQL `ILIKE`). Letter bytes in the needle then accept
    /// either case at every position.
    pub fn try_new_with(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        pattern: &[u8],
        case_insensitive: bool,
    ) -> VortexResult<Option<Self>> {
        let Some(like_kind) = LikeKind::parse(pattern) else {
            return Ok(None);
        };

        let ci = case_insensitive;
        let inner = match like_kind {
            LikeKind::Prefix(b"") | LikeKind::Contains(b"") | LikeKind::Suffix(b"") => {
                MatcherInner::MatchAll
            }
            LikeKind::Prefix(prefix) => {
                if prefix.len() > FlatPrefixDfa::MAX_PREFIX_LEN {
                    return Ok(None);
                }
                MatcherInner::Prefix(FlatPrefixDfa::new(symbols, symbol_lengths, prefix, ci)?)
            }
            LikeKind::Suffix(suffix) => {
                if suffix.len() > SuffixMatcher::MAX_SUFFIX_LEN {
                    return Ok(None);
                }
                MatcherInner::Suffix(SuffixMatcher::new(symbols, symbol_lengths, suffix, ci)?)
            }
            LikeKind::Contains(needle) => {
                // Prefer Shift-Or for short (≤ 8 byte) needles when no
                // symbol's expansion contains the needle (SSA). One ALU
                // op per code byte beats the Teddy + verifier dispatch
                // for selective short needles. Falls through to the
                // FoldedContains path when SSA is present (ShiftOrDfa::new
                // returns Err in that regime) or when ASCII case-folding
                // can't fully express the needle's case rules.
                if needle.len() <= shift_or::MAX_NEEDLE_LEN
                    && needle_len_safe_for_shift_or(needle, ci)
                    && let Ok(dfa) = ShiftOrDfa::new(symbols, symbol_lengths, needle, ci)
                {
                    MatcherInner::ShiftOr(dfa)
                } else if needle.len() <= FoldedContainsDfa::MAX_NEEDLE_LEN {
                    MatcherInner::FoldedContains(FoldedContainsDfa::new(
                        symbols,
                        symbol_lengths,
                        needle,
                        ci,
                    )?)
                } else if needle.len() <= FlatContainsDfa::MAX_NEEDLE_LEN {
                    MatcherInner::Contains(FlatContainsDfa::new(
                        symbols,
                        symbol_lengths,
                        needle,
                        ci,
                    )?)
                } else {
                    return Ok(None);
                }
            }
            LikeKind::MultiContains(segments) => {
                let total_len: usize = segments.iter().map(|s| s.len()).sum();
                if total_len > MultiContainsDfa::MAX_TOTAL_LEN {
                    return Ok(None);
                }
                MatcherInner::MultiContains(Box::new(MultiContainsDfa::new(
                    symbols,
                    symbol_lengths,
                    &segments,
                    ci,
                )?))
            }
        };

        Ok(Some(Self { inner }))
    }

    /// Run the matcher on a single FSST-compressed code sequence.
    #[inline]
    pub fn matches(&self, codes: &[u8]) -> bool {
        match &self.inner {
            MatcherInner::MatchAll => true,
            MatcherInner::Prefix(dfa) => dfa.matches(codes),
            MatcherInner::Suffix(dfa) => dfa.matches(codes),
            MatcherInner::ShiftOr(dfa) => dfa.matches(codes),
            MatcherInner::FoldedContains(dfa) => dfa.matches(codes),
            MatcherInner::Contains(dfa) => dfa.matches(codes),
            MatcherInner::MultiContains(dfa) => dfa.matches(codes),
        }
    }

    #[inline]
    pub(crate) fn matcher_name(&self) -> &'static str {
        match &self.inner {
            MatcherInner::MatchAll => "match_all",
            MatcherInner::Prefix(_) => "prefix",
            MatcherInner::Suffix(_) => "suffix",
            MatcherInner::ShiftOr(_) => "shift_or",
            MatcherInner::FoldedContains(_) => "folded_contains",
            MatcherInner::Contains(_) => "contains",
            MatcherInner::MultiContains(_) => "multi_contains",
        }
    }

    #[inline]
    pub(crate) fn scan_plan_name(&self) -> &'static str {
        match &self.inner {
            MatcherInner::MatchAll => "match_all",
            MatcherInner::Prefix(_) => "row_start_scan",
            MatcherInner::Suffix(_) => "row_loop",
            MatcherInner::ShiftOr(_) => "shift_or",
            MatcherInner::FoldedContains(dfa) => dfa.scan_plan_name(),
            MatcherInner::Contains(_) => "row_loop",
            MatcherInner::MultiContains(_) => "row_loop",
        }
    }

    /// Returns the underlying `FoldedContainsDfa` when the pattern is a
    /// short `%needle%` contains pattern. Exposed for benches that
    /// drive the prefilter primitives directly.
    #[cfg(any(test, feature = "_test-harness"))]
    pub fn as_folded(&self) -> Option<&FoldedContainsDfa> {
        match &self.inner {
            MatcherInner::FoldedContains(dfa) => Some(dfa),
            _ => None,
        }
    }

    /// Scan `n` strings (delimited by `offsets` over `all_bytes`) and return a
    /// `BitBuffer` whose `i`-th bit is set iff the matcher accepts the `i`-th
    /// string (XOR `negated`).
    ///
    /// Performs ONE enum dispatch per call — i.e., per `LIKE` invocation, not
    /// per string — routing to a specialized scan loop with the matcher logic
    /// monomorphized into the bit-packing loop.
    #[inline]
    pub fn scan_to_bitbuf<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        match &self.inner {
            MatcherInner::MatchAll => {
                if negated {
                    BitBuffer::new_unset(n)
                } else {
                    BitBuffer::new_set(n)
                }
            }
            MatcherInner::Prefix(dfa) => dfa.scan_to_bitbuf(n, offsets, all_bytes, negated),
            MatcherInner::Suffix(dfa) => dfa.scan_to_bitbuf(n, offsets, all_bytes, negated),
            MatcherInner::ShiftOr(dfa) => dfa.scan_to_bitbuf(n, offsets, all_bytes, negated),
            MatcherInner::FoldedContains(dfa) => dfa.scan_to_bitbuf(n, offsets, all_bytes, negated),
            MatcherInner::Contains(dfa) => dfa.scan_to_bitbuf(n, offsets, all_bytes, negated),
            MatcherInner::MultiContains(dfa) => dfa.scan_to_bitbuf(n, offsets, all_bytes, negated),
        }
    }
}

// ---------------------------------------------------------------------------
// MultiNeedleMatcher — Fat Teddy multi-needle OR matcher
// ---------------------------------------------------------------------------

/// A compiled multi-needle OR matcher for `LIKE x OR LIKE y OR ...` over
/// FSST-compressed strings. Builds a per-needle [`FoldedContainsDfa`] for
/// every needle, packs up to eight needles per Fat Teddy pass, and OR-merges
/// the results across passes.
///
/// Falls back to per-needle [`FsstMatcher`]-style scans for:
/// - Needles whose progressing c1 set includes `ESCAPE_CODE` (cross-bucket
///   FDR handling is a follow-up; see the TODO in
///   [`fat_teddy::pack_needles`]).
/// - Needles outside the [`FoldedContainsDfa`] needle-length range
///   (`> 127` bytes).
/// - Needles with patterns we can't represent as `%needle%`
///   (`prefix%`, `%suffix`, multi-contains) — Fat Teddy is `Contains`-only.
///
/// Pass [`Self::try_new_multi`] a slice of LIKE patterns; it returns
/// `None` if any pattern is unsupported for pushdown. Use
/// [`Self::scan_or_to_bitbuf`] to produce a single [`BitBuffer`] whose bit
/// `i` is set iff the `i`-th row matches *any* of the needles.
pub struct MultiNeedleMatcher {
    /// Per-needle DFAs. Indices into this vec are the bucket-needle ids
    /// used by [`fat_teddy`].
    fat_teddy_dfas: Vec<FoldedContainsDfa>,
    /// Bucket-packed needles for the Fat Teddy pass. The buckets cover
    /// `fat_teddy_dfas` minus `fat_teddy_fallback`.
    buckets: Vec<fat_teddy::Bucket>,
    /// Indices into `fat_teddy_dfas` that couldn't participate in a Fat
    /// Teddy pass (ESCAPE-anchored, no usable pair buckets, etc.) and
    /// must be evaluated via per-needle scans.
    fat_teddy_fallback: Vec<u16>,
    /// Needles that couldn't be folded-contains at all (too long,
    /// non-Contains shape) and need a generic per-needle scan via
    /// [`FsstMatcher::scan_to_bitbuf`].
    generic_matchers: Vec<FsstMatcher>,
}

impl MultiNeedleMatcher {
    /// Maximum number of needles in a single OR. The implementation is
    /// not strictly capped, but very large needle sets degrade to
    /// `ceil(N / 8)` Fat Teddy passes plus N-pass for fallbacks.
    pub const MAX_NEEDLES: usize = u16::MAX as usize;

    /// Try to build a multi-needle matcher for the given LIKE patterns.
    /// Returns `Ok(None)` if any pattern is unsupported for pushdown
    /// (mixed shapes, exceeds DFA length limits, etc.) — the caller
    /// should then fall back to ordinary decompression-based LIKE OR.
    ///
    /// All patterns share the same `case_insensitive` flag. Mixing
    /// CI/CS in a single batch is not supported.
    pub fn try_new_multi(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        patterns: &[&[u8]],
        case_insensitive: bool,
    ) -> VortexResult<Option<Self>> {
        if patterns.is_empty() || patterns.len() > Self::MAX_NEEDLES {
            return Ok(None);
        }

        // Partition each pattern into either a Fat-Teddy-eligible
        // FoldedContainsDfa or a generic per-needle FsstMatcher. Any
        // pattern that fails the parser falls through to a None result
        // for the entire batch — the caller will then take the
        // fallback path.
        let mut fat_teddy_dfas: Vec<FoldedContainsDfa> = Vec::new();
        let mut generic_matchers: Vec<FsstMatcher> = Vec::new();

        for &pattern in patterns {
            let Some(kind) = LikeKind::parse(pattern) else {
                return Ok(None);
            };
            match kind {
                LikeKind::Contains(needle)
                    if !needle.is_empty() && needle.len() <= FoldedContainsDfa::MAX_NEEDLE_LEN =>
                {
                    let dfa =
                        FoldedContainsDfa::new(symbols, symbol_lengths, needle, case_insensitive)?;
                    fat_teddy_dfas.push(dfa);
                }
                _ => {
                    // Use the existing single-pattern matcher (handles
                    // empty contains as MatchAll, plus all the other
                    // shapes that Fat Teddy doesn't cover).
                    let Some(m) = FsstMatcher::try_new_with(
                        symbols,
                        symbol_lengths,
                        pattern,
                        case_insensitive,
                    )?
                    else {
                        return Ok(None);
                    };
                    generic_matchers.push(m);
                }
            }
        }

        let (buckets, fat_teddy_fallback) = fat_teddy::pack_needles(&fat_teddy_dfas);

        Ok(Some(Self {
            fat_teddy_dfas,
            buckets,
            fat_teddy_fallback,
            generic_matchers,
        }))
    }

    /// Number of needles in this batch (Fat-Teddy-eligible plus generic
    /// fallback). Exposed for tests and diagnostics.
    pub fn needle_count(&self) -> usize {
        self.fat_teddy_dfas.len() + self.generic_matchers.len()
    }

    /// Number of buckets the Fat Teddy pass would use. Exposed for
    /// tests and benches.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Number of needles routed to the per-needle scalar fallback —
    /// either because they couldn't fit into a Fat Teddy bucket
    /// (ESCAPE-anchored, no pair buckets) or because they're not
    /// `Contains`-shape (the `generic_matchers`).
    pub fn fallback_count(&self) -> usize {
        self.fat_teddy_fallback.len() + self.generic_matchers.len()
    }

    /// Scan `n` strings (delimited by `offsets` over `all_bytes`) and
    /// return a `BitBuffer` whose `i`-th bit is set iff *any* of the
    /// configured needles match the `i`-th string (XOR `negated`).
    ///
    /// When `negated` is `true`, the bit `i` is set iff *none* of the
    /// needles match — i.e. `NOT (LIKE x OR LIKE y ...)` ≡ `NOT LIKE x
    /// AND NOT LIKE y ...`. We compute the inverse by running the
    /// positive scan and inverting at the end; this preserves the
    /// existing single-pattern `negated` semantics.
    pub fn scan_or_to_bitbuf<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        // We always run the positive OR scan and invert at the end if
        // negated. This keeps the candidate-handling loop simple and
        // matches the test convention.
        let positive = self.scan_or_positive(n, offsets, all_bytes);
        if negated { !&positive } else { positive }
    }

    /// Internal: positive OR scan. The `negated` post-processing
    /// happens in [`Self::scan_or_to_bitbuf`].
    fn scan_or_positive<T>(&self, n: usize, offsets: &[T], all_bytes: &[u8]) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        // Empty needle set: no row matches.
        if self.needle_count() == 0 {
            return BitBuffer::new_unset(n);
        }

        // Step 1: Fat Teddy pass over up to eight buckets at a time.
        // Each pass takes a chunk of `FAT_TEDDY_BUCKETS` buckets and
        // OR-merges into the running result.
        let mut acc: Option<BitBuffer> = None;
        for chunk in self.buckets.chunks(fat_teddy::FAT_TEDDY_BUCKETS) {
            let pass = fat_teddy::fat_teddy_or_scan(
                n,
                offsets,
                all_bytes,
                chunk,
                false,
                |needle_idx, row| {
                    // Bucket needle ids index into self.fat_teddy_dfas.
                    let dfa = &self.fat_teddy_dfas[usize::from(needle_idx)];
                    dfa.matches(row)
                },
            );
            acc = Some(match acc {
                Some(prev) => &prev | &pass,
                None => pass,
            });
        }

        // Step 2: Fat Teddy fallback — per-needle scans for needles
        // that couldn't participate in the Fat Teddy pass. OR these
        // in via scalar per-row scans.
        for &fb_idx in &self.fat_teddy_fallback {
            let dfa = &self.fat_teddy_dfas[usize::from(fb_idx)];
            let pass = dfa.scan_to_bitbuf(n, offsets, all_bytes, false);
            acc = Some(match acc {
                Some(prev) => &prev | &pass,
                None => pass,
            });
        }

        // Step 3: Generic per-needle scans for non-Contains shapes.
        for matcher in &self.generic_matchers {
            let pass = matcher.scan_to_bitbuf(n, offsets, all_bytes, false);
            acc = Some(match acc {
                Some(prev) => &prev | &pass,
                None => pass,
            });
        }

        acc.unwrap_or_else(|| BitBuffer::new_unset(n))
    }
}

/// Returns `true` iff the Shift-Or path can be safely used for this needle.
///
/// Shift-Or supports ASCII case-folding (uppercase ↔ lowercase letter
/// bytes). Non-ASCII bytes in the needle have no defined case folding in
/// the matcher (the FSST stream is byte-level, not Unicode-aware), so for
/// case-insensitive matching the Shift-Or `B` table is identical to the
/// case-sensitive one for non-letter bytes — i.e. ASCII-only case folding
/// is correct. This helper exists to make the routing decision explicit
/// and to leave room for future stricter checks if non-ASCII semantics
/// ever land.
#[inline]
fn needle_len_safe_for_shift_or(needle: &[u8], _ci: bool) -> bool {
    !needle.is_empty() && needle.len() <= shift_or::MAX_NEEDLE_LEN
}

/// The subset of LIKE patterns we can handle without decompression.
enum LikeKind<'a> {
    /// `prefix%`
    Prefix(&'a [u8]),
    /// `%suffix`
    Suffix(&'a [u8]),
    /// `%needle%`
    Contains(&'a [u8]),
    /// `%seg1%seg2%...%segN%`
    MultiContains(Vec<&'a [u8]>),
}

impl<'a> LikeKind<'a> {
    fn parse(pattern: &'a [u8]) -> Option<Self> {
        // `prefix%` (including just `%` where prefix is empty).
        // `_` in the prefix is the single-byte wildcard (anchored from
        // the row start, no KMP fallback ambiguity).
        if let Some(prefix) = pattern.strip_suffix(b"%")
            && !prefix.contains(&b'%')
        {
            return Some(LikeKind::Prefix(prefix));
        }

        // `%suffix` (no trailing %); `_` allowed in suffix (anchored
        // from the row end, scanned right-to-left, also wildcard-safe).
        if let Some(suffix) = pattern.strip_prefix(b"%")
            && !suffix.contains(&b'%')
        {
            return Some(LikeKind::Suffix(suffix));
        }

        // `%needle%`. We reject `_` in unanchored contains for now —
        // the symmetric KMP failure function over-approximates when
        // wildcards appear in the matched portion, producing false
        // positives. A correct unanchored wildcard matcher needs NFA
        // subset construction (or per-position sliding-window match);
        // tracked as a follow-up.
        let inner = pattern.strip_prefix(b"%")?.strip_suffix(b"%")?;
        if !inner.contains(&b'%') && !inner.contains(&b'_') {
            return Some(LikeKind::Contains(inner));
        }

        // `%seg1%seg2%...%segN%`. Same wildcard limitation: any
        // segment containing `_` falls through to the
        // decompression-based fallback.
        let segments: Vec<&[u8]> = inner
            .split(|&b| b == b'%')
            .filter(|s| !s.is_empty())
            .collect();
        if segments.len() >= 2 && segments.iter().all(|s| !s.contains(&b'_')) {
            return Some(LikeKind::MultiContains(segments));
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Scan helper
// ---------------------------------------------------------------------------

// TODO: add N-way ILP overrun scan for higher throughput on short strings.
//
// `scan_to_bitbuf_with` is the shared inner loop used by every DFA's
// `scan_to_bitbuf` method. Marked `#[inline(always)]` so that, when invoked
// from a DFA-specific `scan_to_bitbuf` with a concrete closure that calls
// that DFA's `matches`, the closure body is fully monomorphized into the
// bit-packing loop and the per-string enum dispatch present in
// `FsstMatcher::matches` is eliminated entirely.
//
// SAFETY contract for callers: `offsets` must contain `n + 1` entries that are
// monotonically non-decreasing and whose final entry does not exceed
// `all_bytes.len()`. This mirrors the invariant the upstream `varbin`
// representation already guarantees.
#[inline(always)]
pub(crate) fn scan_to_bitbuf_with<T, F>(
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    mut matches: F,
) -> BitBuffer
where
    T: vortex_array::dtype::IntegerPType,
    F: FnMut(&[u8]) -> bool,
{
    debug_assert!(offsets.len() > n);
    // SAFETY: caller guarantees `offsets.len() > n`, i.e. at least `n + 1`
    // entries.
    let mut start: usize = unsafe { *offsets.get_unchecked(0) }.as_();
    BitBuffer::collect_bool(n, |i| {
        // SAFETY: `i < n` (BitBuffer::collect_bool invariant) and
        // `offsets.len() >= n + 1` so `i + 1 < offsets.len()`.
        let end: usize = unsafe { *offsets.get_unchecked(i + 1) }.as_();
        debug_assert!(start <= end && end <= all_bytes.len());
        // SAFETY: caller guarantees `start <= end <= all_bytes.len()` via the
        // monotonicity / bounds invariants on `offsets`.
        let codes = unsafe { all_bytes.get_unchecked(start..end) };
        let result = matches(codes) != negated;
        start = end;
        result
    })
}

// ---------------------------------------------------------------------------
// DFA construction helpers
// ---------------------------------------------------------------------------

/// Returns `true` iff no literal byte of `needle` appears in any symbol's
/// expansion. Wildcard (`_`) positions are skipped — they're allowed to
/// match symbol bytes and don't constrain the prefilter.
///
/// When this holds AND the needle has no wildcards, every needle byte in
/// the decompressed stream must come from an `ESCAPE` pair, so the only
/// compressed sequence that reaches the contains DFA's accept state
/// from state 0 is exactly
/// `[ESCAPE, needle[0], ESCAPE, needle[1], …, ESCAPE, needle[L-1]]`. The
/// contains scan can then prefilter with a single `memmem` for that 2L-byte
/// pattern. For needles WITH wildcards, the same condition implies each
/// literal byte must come from an escape pair, but wildcard bytes can
/// come from anywhere — the encoded pattern is no longer unique, so
/// the memmem prefilter is disabled.
pub(super) fn needle_bytes_absent_from_all_symbols(
    symbols: &[Symbol],
    symbol_lengths: &[u8],
    needle: &[u8],
) -> bool {
    let mut needle_byte_present = [false; 256];
    for &b in needle {
        if b == WILDCARD {
            continue;
        }
        needle_byte_present[usize::from(b)] = true;
    }
    debug_assert!(symbol_lengths.len() >= symbols.len());
    for (sym, &len) in symbols.iter().zip(symbol_lengths.iter()) {
        let bytes = sym.to_u64().to_le_bytes();
        let len = usize::from(len).min(8);
        for &b in &bytes[..len] {
            if needle_byte_present[usize::from(b)] {
                return false;
            }
        }
    }
    true
}

/// Build the compressed pattern `[ESCAPE, needle[0], ESCAPE, needle[1], …,
/// ESCAPE, needle[L-1]]`. Only meaningful when
/// [`needle_bytes_absent_from_all_symbols`] is true AND the needle is
/// wildcard-free.
pub(super) fn build_escape_only_encoded_pattern(needle: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(needle.len() * 2);
    for &b in needle {
        out.push(ESCAPE_CODE);
        out.push(b);
    }
    out
}

/// `true` iff the needle has no `_` wildcard bytes.
#[inline]
pub(super) fn needle_is_literal(needle: &[u8]) -> bool {
    !needle.contains(&WILDCARD)
}

/// Builds the per-symbol transition table for FSST symbols.
///
/// For each `(state, symbol_code)` pair, simulates feeding the symbol's bytes
/// through the byte-level transition table to compute the resulting state.
///
/// Returns a flat `Vec<u8>` indexed as `[state * n_symbols + code]`.
///
/// ## Implementation note (perf)
///
/// The natural loop ordering `for state { for code { simulate } }` has a
/// dependency chain `s = byte_table[s * 256 + b]` per byte that's hard
/// to pipeline. We swap the loops to `for code { for byte { for state }
/// }` so the per-state lookups within a single byte step are
/// independent — the CPU can issue them in parallel up to the load-port
/// budget. With `n_states ≤ 8` this turns ~7 dependent loads per byte
/// into 2-cycle batched loads, ~3× faster on `%google%`-class needles
/// where the inner loop dominates `FoldedContainsDfa::new`.
///
/// Accept-state stickiness is handled inside `byte_table` itself
/// (rows for `accept` map every byte → `accept`), so the inner loop
/// doesn't need a per-cell branch.
fn build_symbol_transitions(
    symbols: &[Symbol],
    symbol_lengths: &[u8],
    byte_table: &[u8],
    n_states: u8,
    _accept_state: u8,
) -> Vec<u8> {
    let n_symbols = symbols.len();
    let n_states_usize = usize::from(n_states);
    let mut sym_trans = vec![0u8; n_states_usize * n_symbols];
    debug_assert!(byte_table.len() >= n_states_usize * 256);
    debug_assert!(symbol_lengths.len() >= n_symbols);

    let bt = byte_table.as_ptr();
    let mut v = [0u8; 256];
    for code in 0..n_symbols {
        for s in 0..n_states_usize {
            v[s] = s as u8;
        }

        let sym_bytes = symbols[code].to_u64().to_le_bytes();
        // SAFETY: `code < n_symbols ≤ symbol_lengths.len()`.
        let sym_len = usize::from(unsafe { *symbol_lengths.get_unchecked(code) });
        for &b in &sym_bytes[..sym_len.min(8)] {
            let b_us = usize::from(b);
            // Independent loads across states — pipeline them. Bounds
            // are safe by construction: v[s] is a valid state (< 256),
            // b_us < 256, byte_table is (n_states * 256) bytes long
            // and v[s] < n_states (invariant maintained by the
            // initial state init and the byte_table semantics —
            // every transition lands in a valid state).
            //
            // We unroll 4 at a time so the compiler emits independent
            // loads even without inlining.
            let mut s = 0;
            while s + 4 <= n_states_usize {
                // SAFETY: see safety comment above.
                unsafe {
                    let i0 = usize::from(v[s]) * 256 + b_us;
                    let i1 = usize::from(v[s + 1]) * 256 + b_us;
                    let i2 = usize::from(v[s + 2]) * 256 + b_us;
                    let i3 = usize::from(v[s + 3]) * 256 + b_us;
                    let v0 = *bt.add(i0);
                    let v1 = *bt.add(i1);
                    let v2 = *bt.add(i2);
                    let v3 = *bt.add(i3);
                    v[s] = v0;
                    v[s + 1] = v1;
                    v[s + 2] = v2;
                    v[s + 3] = v3;
                }
                s += 4;
            }
            while s < n_states_usize {
                // SAFETY: see safety comment above.
                unsafe {
                    v[s] = *bt.add(usize::from(v[s]) * 256 + b_us);
                }
                s += 1;
            }
        }

        // Scatter results into the per-state rows.
        for s in 0..n_states_usize {
            // SAFETY: s < n_states, code < n_symbols.
            unsafe {
                *sym_trans.get_unchecked_mut(s * n_symbols + code) = v[s];
            }
        }
    }
    sym_trans
}

/// Builds a fused 256-wide transition table from symbol transitions.
///
/// For each `(state, code_byte)`:
/// - Code bytes `0..n_symbols`: use the symbol transition
/// - `ESCAPE_CODE`: maps to `escape_value` (either a sentinel or escape state)
/// - All others: use `default` (typically 0 for contains, fail_state for prefix)
///
/// Returns a flat `Vec<u8>` indexed as `[state * 256 + code_byte]`.
fn build_fused_table(
    sym_trans: &[u8],
    n_symbols: usize,
    n_states: u8,
    escape_value_fn: impl Fn(u8) -> u8,
    default: u8,
) -> Vec<u8> {
    let mut fused = vec![default; usize::from(n_states) * 256];
    for state in 0..n_states {
        let s = usize::from(state);
        for code in 0..n_symbols {
            fused[s * 256 + code] = sym_trans[s * n_symbols + code];
        }
        fused[s * 256 + usize::from(ESCAPE_CODE)] = escape_value_fn(state);
    }
    fused
}

// ---------------------------------------------------------------------------
// KMP helpers
// ---------------------------------------------------------------------------

/// The wildcard byte in a LIKE needle. SQL `_` (`0x5F`) is interpreted
/// as "match any single byte" by [`kmp_byte_transitions`] and
/// [`kmp_failure_table`]. Without SQL `ESCAPE` support, every `_`
/// in the parsed needle is a wildcard; a literal `_` cannot be
/// expressed.
pub(super) const WILDCARD: u8 = b'_';

/// ASCII case fold to lowercase. Non-letters pass through.
#[inline]
fn ascii_to_lower(b: u8) -> u8 {
    if b.is_ascii_uppercase() { b + 32 } else { b }
}

/// Pattern-position byte equality with wildcard semantics. Returns
/// `true` if `a` or `b` is the [`WILDCARD`] byte, or both bytes are
/// equal. When `ci` is true, ASCII letter case is ignored.
#[inline]
fn pattern_eq(a: u8, b: u8, ci: bool) -> bool {
    if a == WILDCARD || b == WILDCARD {
        return true;
    }
    if ci {
        ascii_to_lower(a) == ascii_to_lower(b)
    } else {
        a == b
    }
}

/// Concrete-input byte match against a needle position. The pattern
/// position `p` is one of the needle bytes (possibly the wildcard);
/// the input byte `b` is always concrete (never a wildcard). When `ci`
/// is true, ASCII letter case is ignored.
#[inline]
#[expect(
    dead_code,
    reason = "Reserved for the future correct contains-wildcard DFA."
)]
fn pattern_matches_byte(p: u8, b: u8, ci: bool) -> bool {
    if p == WILDCARD {
        return true;
    }
    if ci {
        ascii_to_lower(p) == ascii_to_lower(b)
    } else {
        p == b
    }
}

/// For an advancing transition on byte `needle_byte`, set the table
/// row entry. With `ci` true, also set the entry for the case-flipped
/// byte so either case of the same ASCII letter advances.
#[inline]
fn set_advance(table: &mut [u8], row_start: usize, needle_byte: u8, new_state: u8, ci: bool) {
    table[row_start + usize::from(needle_byte)] = new_state;
    if ci && needle_byte.is_ascii_alphabetic() {
        let flipped = needle_byte ^ 0x20;
        table[row_start + usize::from(flipped)] = new_state;
    }
}

/// Build the `(state × byte) → state` KMP transition table.
///
/// ## Construction
///
/// Uses the standard recurrence — for each non-accept state `s`:
///   - On byte == `needle[s]` (or `needle[s]` is the wildcard): transition to `s + 1`.
///   - On any other byte: transition to whatever the *failure* row
///     would give for the same byte, i.e. `table[failure[s-1] * 256 + b]`
///     for `s > 0`, and `0` for `s = 0`.
///
/// When `needle[s]` is the [`WILDCARD`] byte (`_`), every input byte
/// advances to `s + 1` regardless of the failure row's content.
///
/// This is one 256-byte memcpy + a single override per state, instead
/// of running the KMP fallback loop at every cell.
fn kmp_byte_transitions(needle: &[u8], ci: bool) -> Vec<u8> {
    let n_states = u8::try_from(needle.len() + 1)
        .vortex_expect("kmp_byte_transitions: must have needle.len() ≤ 255");
    let accept = n_states - 1;
    let failure = kmp_failure_table(needle, ci);

    let mut table = vec![0u8; usize::from(n_states) * 256];

    // State 0: either `needle[0]` (literal) or every byte (wildcard) advances.
    if let Some(&first) = needle.first() {
        if first == WILDCARD {
            table[0..256].fill(1);
        } else {
            set_advance(&mut table, 0, first, 1, ci);
        }
    }

    // States 1..accept: each row is the failure-row plus the advance entry.
    for state in 1..accept {
        let s = usize::from(state);
        let fail_row = usize::from(failure[s - 1]) * 256;
        let state_row = s * 256;
        // Copy the failure row: for every byte not equal to needle[s],
        // the KMP fallback eventually lands at the same place the
        // failure-state would land on that byte.
        table.copy_within(fail_row..fail_row + 256, state_row);
        // Override the advancing entries.
        if needle[s] == WILDCARD {
            // Wildcard at position s: every byte advances.
            table[state_row..state_row + 256].fill(state + 1);
        } else {
            set_advance(&mut table, state_row, needle[s], state + 1, ci);
        }
    }

    // Accept state: sticky — every byte stays at accept.
    if usize::from(accept) < usize::from(n_states) {
        let accept_row = usize::from(accept) * 256;
        table[accept_row..accept_row + 256].fill(accept);
    }

    table
}

fn kmp_failure_table(needle: &[u8], ci: bool) -> Vec<u8> {
    let mut failure = vec![0u8; needle.len()];
    let mut k = 0u8;
    for i in 1..needle.len() {
        while k > 0 && !pattern_eq(needle[usize::from(k)], needle[i], ci) {
            k = failure[usize::from(k) - 1];
        }
        if pattern_eq(needle[usize::from(k)], needle[i], ci) {
            k += 1;
        }
        failure[i] = k;
    }
    failure
}
