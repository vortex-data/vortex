// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::LazyLock;

use fsst::ESCAPE_CODE;
use fsst::Symbol;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::FsstMatcher;
use super::LikeKind;
use super::flat_contains::FlatContainsDfa;
use super::prefix::FlatPrefixDfa;
use super::suffix::SuffixMatcher;
use crate::FSST;
use crate::FSSTArray;
use crate::fsst_compress;
use crate::fsst_train_compressor;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

/// Helper: make a Symbol from a byte string (up to 8 bytes, zero-padded).
fn sym(bytes: &[u8]) -> Symbol {
    let mut buf = [0u8; 8];
    buf[..bytes.len()].copy_from_slice(bytes);
    Symbol::from_slice(&buf)
}

fn escaped(bytes: &[u8]) -> Vec<u8> {
    let mut codes = Vec::with_capacity(bytes.len() * 2);
    for &b in bytes {
        codes.push(ESCAPE_CODE);
        codes.push(b);
    }
    codes
}

fn assert_borrowed_prefix(pattern: &[u8], expected: &[u8]) {
    let Some(LikeKind::Prefix(actual)) = LikeKind::parse(pattern) else {
        panic!("expected borrowed prefix pattern");
    };
    assert!(matches!(actual, Cow::Borrowed(_)));
    assert_eq!(actual.as_ref(), expected);
}

fn assert_owned_prefix(pattern: &[u8], expected: &[u8]) {
    let Some(LikeKind::Prefix(actual)) = LikeKind::parse(pattern) else {
        panic!("expected owned prefix pattern");
    };
    assert!(matches!(actual, Cow::Owned(_)));
    assert_eq!(actual.as_ref(), expected);
}

fn assert_borrowed_suffix(pattern: &[u8], expected: &[u8]) {
    let Some(LikeKind::Suffix(actual)) = LikeKind::parse(pattern) else {
        panic!("expected borrowed suffix pattern");
    };
    assert!(matches!(actual, Cow::Borrowed(_)));
    assert_eq!(actual.as_ref(), expected);
}

fn assert_owned_suffix(pattern: &[u8], expected: &[u8]) {
    let Some(LikeKind::Suffix(actual)) = LikeKind::parse(pattern) else {
        panic!("expected owned suffix pattern");
    };
    assert!(matches!(actual, Cow::Owned(_)));
    assert_eq!(actual.as_ref(), expected);
}

fn assert_borrowed_contains(pattern: &[u8], expected: &[u8]) {
    let Some(LikeKind::Contains(actual)) = LikeKind::parse(pattern) else {
        panic!("expected borrowed contains pattern");
    };
    assert!(matches!(actual, Cow::Borrowed(_)));
    assert_eq!(actual.as_ref(), expected);
}

fn assert_owned_contains(pattern: &[u8], expected: &[u8]) {
    let Some(LikeKind::Contains(actual)) = LikeKind::parse(pattern) else {
        panic!("expected owned contains pattern");
    };
    assert!(matches!(actual, Cow::Owned(_)));
    assert_eq!(actual.as_ref(), expected);
}

#[test]
fn test_like_kind_parse_plain_patterns() {
    assert_borrowed_prefix(b"http%", b"http");
    assert_borrowed_contains(b"%needle%", b"needle");
    assert_borrowed_suffix(b"%suffix", b"suffix");
    assert_borrowed_prefix(b"%", b"");
}

#[test]
fn test_like_kind_parse_escaped_patterns() {
    assert_owned_prefix(br"\%%", b"%");
    assert_owned_prefix(br"\_%", b"_");
    assert_owned_prefix(br"\\%", b"\\");
    assert_owned_prefix(br"has\%middle%", b"has%middle");
    assert_owned_contains(br"%\%%", b"%");
    assert_owned_contains(br"%\_%", b"_");
    assert_owned_contains(br"%\\%", b"\\");
    assert_owned_contains(br"%has\%middle%", b"has%middle");
    assert_owned_suffix(br"%\%", b"%");
    assert_owned_suffix(br"%\_", b"_");
    assert_owned_suffix(br"%has\%middle", b"has%middle");
}

#[test]
fn test_like_kind_parse_unsupported_patterns() {
    assert!(LikeKind::parse(b"a_c").is_none());
    assert!(LikeKind::parse(br"foo\%bar").is_none());
    // A `%` in the middle is neither prefix, contains, nor suffix.
    assert!(LikeKind::parse(b"a%b").is_none());
    assert!(LikeKind::parse(b"%a%b").is_none());
    // `_` anywhere in the tail still disqualifies a suffix.
    assert!(LikeKind::parse(b"%a_b").is_none());
}

/// No symbols — all bytes escaped. Simplest case to see the two tables.
#[test]
fn test_prefix_dfa_no_symbols() -> VortexResult<()> {
    let dfa = FlatPrefixDfa::new(&[], &[], b"ab")?;

    assert!(dfa.matches(&escaped(b"abx")));
    assert!(dfa.matches(&escaped(b"ab")));
    assert!(!dfa.matches(&escaped(b"a")));
    assert!(!dfa.matches(&escaped(b"ax")));
    assert!(!dfa.matches(&escaped(b"ba")));
    assert!(!dfa.matches(&[]));

    Ok(())
}

/// With symbols — shows how multi-byte symbols interact with prefix matching.
///
/// Symbol table: code 0 = "ht", code 1 = "tp"
/// Prefix: "http"
///
/// The string "http" can be encoded as:
///   [0, 1]           — two symbols: "ht" + "tp"
///   [ESC,h, ESC,t, ESC,t, ESC,p] — all escaped
///   [0, ESC,t, ESC,p]            — symbol "ht" + escaped "t" + escaped "p"
#[test]
fn test_prefix_dfa_with_symbols() -> VortexResult<()> {
    let symbols = [sym(b"ht"), sym(b"tp")];
    let lengths = [2u8, 2];
    let dfa = FlatPrefixDfa::new(&symbols, &lengths, b"http")?;

    // "http" via two symbols: code 0 ("ht") + code 1 ("tp") → accept
    assert!(dfa.matches(&[0, 1]));

    // "http" all escaped
    assert!(dfa.matches(&escaped(b"http")));

    // "http" mixed: symbol "ht" + escaped "tp"
    assert!(dfa.matches(&[0, ESCAPE_CODE, b't', ESCAPE_CODE, b'p']));

    // "htxx" via symbol "ht" + escaped "xx" → fail after "ht" advances to state 2,
    // then 'x' doesn't match 't'
    assert!(!dfa.matches(&[0, ESCAPE_CODE, b'x', ESCAPE_CODE, b'x']));

    // "tp" alone → symbol "tp" from state 0 feeds 't','p' through byte table:
    // state 0 wants 'h', sees 't' → fail
    assert!(!dfa.matches(&[1]));

    Ok(())
}

/// Longer prefix showing more progress states.
#[test]
fn test_prefix_dfa_longer() -> VortexResult<()> {
    // code 0 = "tp" (2 bytes), code 1 = "htt" (3 bytes), code 2 = "p:/" (3 bytes)
    let symbols = [sym(b"tp"), sym(b"htt"), sym(b"p:/")];
    let lengths = [2u8, 3, 3];
    let dfa = FlatPrefixDfa::new(&symbols, &lengths, b"http://")?;

    // "http://e" via symbols: "htt"(1) + "p:/"(2) + escaped "/" + escaped "e"
    // "htt" = states 0→1→2→3, "p:/" = states 3→4→5→6, "/" = state 6→accept
    assert!(dfa.matches(&[1, 2, ESCAPE_CODE, b'/', ESCAPE_CODE, b'e']));

    // "http:/" — 6 chars, missing the 7th '/'
    assert!(!dfa.matches(&[1, ESCAPE_CODE, b'p', ESCAPE_CODE, b':', ESCAPE_CODE, b'/',]));

    // "http://" all escaped — 7 chars, exact match
    assert!(dfa.matches(&escaped(b"http://")));

    // "tp" alone (code 0) from state 0: feeds 't','p' → state 0 wants 'h', sees 't' → fail
    assert!(!dfa.matches(&[0]));

    // "htt" + "tp" = "httpp"? No — "htt" → states 0→1→2→3, then "tp":
    // state 3 wants 'p', sees 't' → fail immediately
    assert!(!dfa.matches(&[1, 0]));

    Ok(())
}

#[test]
fn test_prefix_pushdown_len_13_with_escapes() {
    let matcher = FsstMatcher::try_new(&[], &[], b"abcdefghijklm%")
        .unwrap()
        .unwrap();

    assert!(matcher.matches(&escaped(b"abcdefghijklm")));
    assert!(!matcher.matches(&escaped(b"abcdefghijklx")));
}

#[test]
fn test_prefix_pushdown_len_14_now_handled() {
    // 14-byte prefix is now handled by FlatPrefixDfa (was rejected by shift-packed).
    assert!(
        FsstMatcher::try_new(&[], &[], b"abcdefghijklmn%")
            .unwrap()
            .is_some()
    );
}

#[test]
fn test_prefix_pushdown_long_prefix() -> VortexResult<()> {
    let prefix = "a".repeat(FlatPrefixDfa::MAX_PREFIX_LEN);
    let pattern = format!("{prefix}%");
    let matcher = FsstMatcher::try_new(&[], &[], pattern.as_bytes())?.unwrap();

    assert!(matcher.matches(&escaped(prefix.as_bytes())));

    let mut mismatch = prefix.into_bytes();
    mismatch[FlatPrefixDfa::MAX_PREFIX_LEN - 1] = b'b';
    assert!(!matcher.matches(&escaped(&mismatch)));

    Ok(())
}

#[test]
fn test_prefix_pushdown_rejects_len_254() {
    debug_assert_eq!(FlatPrefixDfa::MAX_PREFIX_LEN, 253);
    let prefix = "a".repeat(254);
    let pattern = format!("{prefix}%");
    assert!(
        FsstMatcher::try_new(&[], &[], pattern.as_bytes())
            .unwrap()
            .is_none()
    );
}

#[test]
fn test_contains_pushdown_len_254_with_escapes() {
    let needle = "a".repeat(FlatContainsDfa::MAX_NEEDLE_LEN);
    let pattern = format!("%{needle}%");
    let matcher = FsstMatcher::try_new(&[], &[], pattern.as_bytes())
        .unwrap()
        .unwrap();

    assert!(matcher.matches(&escaped(needle.as_bytes())));

    let mut mismatch = needle.into_bytes();
    mismatch[FlatContainsDfa::MAX_NEEDLE_LEN - 1] = b'b';
    assert!(!matcher.matches(&escaped(&mismatch)));
}

/// No symbols — every byte escaped, so every token the walk sees is a literal.
#[test]
fn test_suffix_matcher_no_symbols() -> VortexResult<()> {
    let matcher = SuffixMatcher::new(&[], &[], b"ab")?;

    assert!(matcher.matches(&escaped(b"ab")));
    assert!(matcher.matches(&escaped(b"xab")));
    assert!(matcher.matches(&escaped(b"abab")));
    // Only the end matters: "ab" occurs, but not last.
    assert!(!matcher.matches(&escaped(b"abx")));
    assert!(!matcher.matches(&escaped(b"a")));
    assert!(!matcher.matches(&escaped(b"ba")));
    assert!(!matcher.matches(&[]));

    Ok(())
}

/// A suffix whose bytes repeat, so the walk consumes the same byte value twice and the
/// two steps must be counted separately rather than collapsing.
#[test]
fn test_suffix_matcher_repeated_chars() -> VortexResult<()> {
    let matcher = SuffixMatcher::new(&[], &[], b"aa")?;

    assert!(matcher.matches(&escaped(b"aa")));
    assert!(matcher.matches(&escaped(b"aaa")));
    assert!(matcher.matches(&escaped(b"baa")));
    assert!(!matcher.matches(&escaped(b"aab")));
    assert!(!matcher.matches(&escaped(b"a")));

    Ok(())
}

/// With symbols — the suffix can be covered by one symbol, or straddle a symbol and a
/// literal.
///
/// Symbol table: code 0 = "ab", code 1 = "ba"
/// Suffix: "ab"
#[test]
fn test_suffix_matcher_with_symbols() -> VortexResult<()> {
    let symbols = [sym(b"ab"), sym(b"ba")];
    let lengths = [2u8, 2];
    let matcher = SuffixMatcher::new(&symbols, &lengths, b"ab")?;

    // "ab" as one symbol.
    assert!(matcher.matches(&[0]));
    // "abab": the last symbol alone carries the whole suffix.
    assert!(matcher.matches(&[0, 0]));
    // "abba" ends in "ba".
    assert!(!matcher.matches(&[0, 1]));
    // "ba" + escaped 'b' → "bab": the suffix straddles a symbol and a literal.
    assert!(matcher.matches(&[1, ESCAPE_CODE, b'b']));
    // "ba" + escaped "ab" → "baab", matched from two literals.
    assert!(matcher.matches(&[1, ESCAPE_CODE, b'a', ESCAPE_CODE, b'b']));
    // "ba" alone is a partial match only.
    assert!(!matcher.matches(&[1]));
    // "ab" + escaped 'a' → "aba".
    assert!(!matcher.matches(&[0, ESCAPE_CODE, b'a']));

    Ok(())
}

/// Backward walking has to tell a symbol code from an escaped literal that happens to
/// share its byte value. Only the parity of the `ESCAPE_CODE` run to the left does that,
/// and a literal `0xFF` is what makes the run longer than one.
#[test]
fn test_suffix_matcher_escape_run_parity() -> VortexResult<()> {
    let symbols = [sym(b"ab")];
    let lengths = [2u8];
    let matcher = SuffixMatcher::new(&symbols, &lengths, b"ab")?;

    // One escape: the trailing 0 is the literal `0x00`, so the string is "\0", not "ab".
    // Reading it as a code instead would decode "ab" and match.
    assert!(!matcher.matches(&[ESCAPE_CODE, 0]));
    // Two escapes: they pair off into a literal `0xFF`, leaving 0 to be symbol "ab".
    assert!(matcher.matches(&[ESCAPE_CODE, ESCAPE_CODE, 0]));
    // The run terminates at the first byte that is not `ESCAPE_CODE`: "ab" + literal
    // `0x00`, which does not end in "ab".
    assert!(!matcher.matches(&[0, ESCAPE_CODE, 0]));
    // Same run, one escape longer: "ab" + literal `0xFF` + symbol "ab".
    assert!(matcher.matches(&[0, ESCAPE_CODE, ESCAPE_CODE, 0]));
    // A dangling escape marker is a truncated stream, not a match.
    assert!(!matcher.matches(&[0, ESCAPE_CODE]));
    // Runs longer than two: a scan that only looked back one or two bytes would read the
    // trailing 0 as symbol "ab" here and match. Decoded: `0xFF` then `0x00`.
    assert!(!matcher.matches(&[ESCAPE_CODE, ESCAPE_CODE, ESCAPE_CODE, 0]));
    // Four escapes pair off into two literal `0xFF`s, so the 0 is symbol "ab" again.
    assert!(matcher.matches(&[ESCAPE_CODE, ESCAPE_CODE, ESCAPE_CODE, ESCAPE_CODE, 0]));
    // A truncated stream found *after* a token has already matched: the walk still owes a
    // byte and the only thing left is a dangling marker.
    assert!(!matcher.matches(&[ESCAPE_CODE, ESCAPE_CODE, ESCAPE_CODE, b'b']));

    Ok(())
}

/// A literal `0xFF` compared against the suffix itself, not merely skipped over. The
/// kernel accepts binary patterns, so a suffix can contain `0xFF`.
#[test]
fn test_suffix_matcher_literal_escape_byte_in_suffix() -> VortexResult<()> {
    let matcher = SuffixMatcher::new(&[sym(b"ab")], &[2u8], &[b'a', ESCAPE_CODE])?;

    // Symbol "ab" then escaped `0xFF` → "ab\xFF", whose last two bytes are 'b', 0xFF.
    assert!(!matcher.matches(&[0, ESCAPE_CODE, ESCAPE_CODE]));
    // Escaped 'a' then escaped `0xFF` → "a\xFF".
    assert!(matcher.matches(&[ESCAPE_CODE, b'a', ESCAPE_CODE, ESCAPE_CODE]));
    // Escaped `0xFF` alone is one byte short of the suffix.
    assert!(!matcher.matches(&[ESCAPE_CODE, ESCAPE_CODE]));

    Ok(())
}

/// A final symbol longer than the suffix must be compared on its *tail*. Aligning on its
/// head instead would accept `"...abcx" LIKE '%abc'`.
#[test]
fn test_suffix_matcher_symbol_longer_than_suffix() -> VortexResult<()> {
    let symbols = [sym(b"abcx"), sym(b"xabc")];
    let lengths = [4u8, 4];
    let matcher = SuffixMatcher::new(&symbols, &lengths, b"abc")?;

    // "xabc" ends with the suffix; its leading 'x' is simply outside it.
    assert!(matcher.matches(&[1]));
    // "abcx" contains the suffix but does not end with it.
    assert!(!matcher.matches(&[0]));
    // Same distinction when an earlier symbol precedes it.
    assert!(matcher.matches(&[0, 1]));
    assert!(!matcher.matches(&[1, 0]));

    Ok(())
}

/// A code byte past the end of the symbol table is only producible by a corrupt file, and
/// must answer "no match" rather than panic — the prefix and contains DFAs absorb the same
/// byte in their table padding.
#[test]
fn test_suffix_matcher_code_beyond_symbol_table() -> VortexResult<()> {
    let matcher = SuffixMatcher::new(&[sym(b"ab")], &[2u8], b"ab")?;

    assert!(!matcher.matches(&[5]));
    assert!(!matcher.matches(&[ESCAPE_CODE, ESCAPE_CODE, 5]));
    // Reachable past the first token too, where the walk still owes suffix bytes.
    assert!(!matcher.matches(&[5, ESCAPE_CODE, b'b']));
    // The largest code that is not `ESCAPE_CODE`.
    assert!(!matcher.matches(&[254]));

    Ok(())
}

#[test]
fn test_suffix_pushdown_len_254_with_escapes() {
    // Heterogeneous, so that every step of the 254-byte walk reads a different byte and an
    // off-by-one in the suffix index cannot pass.
    let suffix: String = (0..SuffixMatcher::MAX_SUFFIX_LEN)
        .map(|i| char::from(b'a' + u8::try_from(i % 26).unwrap()))
        .collect();
    let pattern = format!("%{suffix}");
    let symbols: Vec<Symbol> = vec![];
    let matcher = FsstMatcher::try_new(&symbols, &[], pattern.as_bytes())
        .unwrap()
        .expect("suffix of MAX_SUFFIX_LEN should be pushed down");
    assert!(matcher.matches(&escaped(suffix.as_bytes())));
    assert!(matcher.matches(&escaped(format!("prefix{suffix}").as_bytes())));
    // Mismatch in the interior, so the walk has to run before it can fail.
    let mut interior = suffix.clone().into_bytes();
    interior[SuffixMatcher::MAX_SUFFIX_LEN / 2] = b'!';
    assert!(!matcher.matches(&escaped(&interior)));
    // Mismatch at the far end of the walk.
    let mut first = suffix.clone().into_bytes();
    first[0] = b'!';
    assert!(!matcher.matches(&escaped(&first)));
    assert!(!matcher.matches(&escaped(format!("{suffix}b").as_bytes())));
}

#[test]
fn test_suffix_pushdown_rejects_len_255() {
    let suffix = "a".repeat(SuffixMatcher::MAX_SUFFIX_LEN + 1);
    let pattern = format!("%{suffix}");
    let symbols: Vec<Symbol> = vec![];
    assert!(
        FsstMatcher::try_new(&symbols, &[], pattern.as_bytes())
            .unwrap()
            .is_none()
    );
}

#[test]
fn test_contains_pushdown_rejects_len_255() {
    let needle = "a".repeat(FlatContainsDfa::MAX_NEEDLE_LEN + 1);
    let pattern = format!("%{needle}%");
    assert!(
        FsstMatcher::try_new(&[], &[], pattern.as_bytes())
            .unwrap()
            .is_none()
    );
}

// ---------------------------------------------------------------------------
// End-to-end edge cases: FSST compress → LIKE → compare booleans
// ---------------------------------------------------------------------------

fn make_fsst_str(strings: &[Option<&str>]) -> FSSTArray {
    let array = VarBinArray::from_iter(
        strings.iter().copied(),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let compressor = fsst_train_compressor(&array, &mut ctx).unwrap();
    fsst_compress(&array, &compressor, &mut ctx).unwrap()
}

/// Evaluates LIKE over an FSST array, asserting first that the kernel pushed the pattern
/// down.
///
/// Without that assertion these cases prove nothing about this module: the fallback
/// decompresses and returns the same booleans, so every one of them would still pass with
/// pushdown disabled. Every pattern in the table below is a shape the module claims to
/// handle, so `None` here is a regression whichever shape it is.
fn run_like(array: FSSTArray, pattern_arr: ArrayRef) -> VortexResult<BoolArray> {
    let mut ctx = SESSION.create_execution_ctx();
    assert!(
        <FSST as LikeKernel>::like(
            array.as_view(),
            &pattern_arr,
            LikeOptions::default(),
            &mut ctx
        )?
        .is_some(),
        "pattern should be pushed down, not evaluated by decompressing"
    );

    let arr: ArrayRef = array.into_array();
    let result = Like::try_new(arr, pattern_arr, LikeOptions::default())?
        .into_array()
        .execute::<Canonical>(&mut ctx)?;
    Ok(result.into_bool())
}

#[rstest]
// Empty strings
#[case(&[""], "aaaa%", &[false])]
#[case(&[""], "%aaaa%", &[false])]
#[case(&[""], "%", &[true])]
#[case(&[""], "%%", &[true])]
#[case(&["", "", ""], "%", &[true, true, true])]
#[case(&["", "abc", ""], "%%", &[true, true, true])]
// Single-char patterns
#[case(&["a", "b", ""], "a%", &[true, false, false])]
#[case(&["a", "b", ""], "%a%", &[true, false, false])]
// Needle longer than every input string
#[case(&["ab", "abc", ""], "%abcd%", &[false, false, false])]
#[case(&["ab", "abc", ""], "abcd%", &[false, false, false])]
// Exact match (prefix pattern = entire string + %)
#[case(&["abc", "abcd", "ab"], "abc%", &[true, true, false])]
#[case(&["abc", "abcd", "ab"], "%abc%", &[true, true, false])]
// Repeated characters — KMP overlap
#[case(&["aa", "aaa", "aaaa", "aba"], "%aaa%", &[false, true, true, false])]
#[case(&["aab", "aaab", "a"], "aaa%", &[false, true, false])]
// Needle at different positions
#[case(&["xxabcyy", "abcyy", "xxabc", "abc", "xabx"], "%abc%", &[true, true, true, true, false])]
// All identical strings
#[case(&["aaa", "aaa", "aaa"], "%aaa%", &[true, true, true])]
#[case(&["aaa", "aaa", "aaa"], "bbb%", &[false, false, false])]
// Single element arrays
#[case(&["hello"], "hello%", &[true])]
#[case(&["hello"], "hellx%", &[false])]
#[case(&["hello"], "%ello%", &[true])]
#[case(&["hello"], "%ellx%", &[false])]
// Overlapping KMP pattern "abab"
#[case(&["ababab", "abab", "aba", "xababx"], "%abab%", &[true, true, false, true])]
// Prefix that shares chars with rest of string
#[case(&["abab", "abba", "abcd"], "ab%", &[true, true, true])]
#[case(&["abab", "abba", "abcd", "ba"], "ab%", &[true, true, true, false])]
// The string "aabaabaabaab" requires multi-level KMP fallback at the 'a' after "aabaabaab"
#[case(&["aabaabaabaab", "aabaabaax", "xaabaabaab"], "%aabaabaab%", &[true, false, true])]
#[case(&["café latte", "naïve approach", "café noir"], "café%", &[true, false, true])]
#[case(&["日本語テスト", "日本語データ", "英語テスト"], "%日本語%", &[true, true, false])]
// 10-byte needle, contains: match at start, middle, end, exact, and near-miss
#[case(
    &["abcdefghijxxx", "xxxabcdefghij", "xxabcdefghijxx", "abcdefghij", "abcdefghxx"],
    "%abcdefghij%",
    &[true, true, true, true, false]
)]
// 10-byte prefix: same needle but anchored at the start of the string
#[case(
    &["abcdefghijxxx", "abcdefghij", "xabcdefghij", "abcdefghxx"],
    "abcdefghij%",
    &[true, true, false, false]
)]
// 9-byte needle with KMP-relevant overlap ("abcabcabc"):
// failure table = [0,0,0,1,2,3,4,5,6], so a partial match of "abcabcab"
// followed by a mismatch must fall back to state 5 ("abcab"), not restart.
// This exercises multi-level KMP backtracking across symbol boundaries.
#[case(
    &["xxabcabcabcxx", "abcabcabc", "abcabcabx", "abcabcxx"],
    "%abcabcabc%",
    &[true, true, false, false]
)]
// ---- suffix (`%suffix`) end-to-end ----
#[case(&["abc", "xabc", "abcx", "bc", ""], "%abc", &[true, true, false, false, false])]
#[case(&[""], "%a", &[false])]
// The needle occurs, but not at the end.
#[case(&["abcabc", "abcabcx"], "%abc", &[true, false])]
// Repeated bytes: the walk owes two steps of the same byte value.
#[case(&["aa", "aaa", "aab", "a"], "%aa", &[true, true, false, false])]
// Suffix equal to the whole string, and one byte longer than it.
#[case(&["hello", "hell"], "%hello", &[true, false])]
// Overlapping suffix, where a shorter tail of it also occurs earlier.
#[case(&["abab", "ababab", "ababa", "xabab"], "%abab", &[true, true, false, true])]
// Multi-byte UTF-8 tails.
#[case(&["café latte", "café", "caf"], "%café", &[false, true, false])]
#[case(&["日本語テスト", "テスト", "テストx"], "%テスト", &[true, true, false])]
// An escaped literal `%` as the suffix — previously fell back, now pushed down.
#[case(&["100%", "100", "%100"], r"%\%", &[true, false, false])]
fn test_like_edge_cases(
    #[case] strings: &[&str],
    #[case] pattern: &str,
    #[case] expected: &[bool],
) -> VortexResult<()> {
    let opts: Vec<Option<&str>> = strings.iter().map(|s| Some(*s)).collect();
    let fsst_arr = make_fsst_str(&opts);
    let result = run_like(
        fsst_arr,
        ConstantArray::new(pattern, opts.len()).into_array(),
    )?;
    let expected_arr = BoolArray::from_iter(expected.iter().copied());
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(&result, &expected_arr, &mut ctx);
    Ok(())
}
