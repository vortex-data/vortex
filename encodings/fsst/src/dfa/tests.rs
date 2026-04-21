// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::session::ArraySession;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::FsstMatcher;
use super::LikeKind;
use super::flat_contains::FlatContainsDfa;
use super::multi_contains::MultiContainsDfa;
use super::prefix::FlatPrefixDfa;
use crate::FSSTArray;
use crate::fsst_compress;
use crate::fsst_train_compressor;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

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

#[test]
fn test_like_kind_parse() {
    assert!(matches!(
        LikeKind::parse(b"http%"),
        Some(LikeKind::Prefix(b"http"))
    ));
    assert!(matches!(
        LikeKind::parse(b"%needle%"),
        Some(LikeKind::Contains(b"needle"))
    ));
    assert!(matches!(LikeKind::parse(b"%"), Some(LikeKind::Prefix(b""))));
    assert!(matches!(
        LikeKind::parse(b"%suffix"),
        Some(LikeKind::Suffix(b"suffix"))
    ));
    // Multi-contains
    assert!(matches!(
        LikeKind::parse(b"%abc%def%"),
        Some(LikeKind::MultiContains(_))
    ));
    assert!(matches!(
        LikeKind::parse(b"%a%b%c%"),
        Some(LikeKind::MultiContains(_))
    ));
    // Consecutive %% in multi-contains is fine (empty segments filtered out)
    assert!(matches!(
        LikeKind::parse(b"%abc%%def%"),
        Some(LikeKind::MultiContains(_))
    ));
    // Underscore in any segment rejects
    assert!(LikeKind::parse(b"%abc%d_f%").is_none());
    // Underscore patterns are not supported.
    assert!(LikeKind::parse(b"a_c").is_none());
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

/// Zero-branch contains variant must agree with the default variant on all
/// inputs that the default variant handles.
#[rstest]
#[case(b"abc", b"abcdef", true)]
#[case(b"abc", b"xxabcxx", true)]
#[case(b"abc", b"ab", false)]
#[case(b"abc", b"", false)]
#[case(b"hello", b"hello world", true)]
#[case(b"world", b"hello world", true)]
#[case(b"xyz", b"no match here", false)]
#[case(b"a", b"a", true)]
#[case(b"a", b"", false)]
#[case(b"abc", b"abc", true)]
fn test_contains_branchless_matches_default(
    #[case] needle: &[u8],
    #[case] haystack: &[u8],
    #[case] expected: bool,
) {
    // No FSST symbols — every byte goes through the escape path.
    let dfa = FlatContainsDfa::new(&[], &[], needle).unwrap();
    let codes = escaped(haystack);
    let default = dfa.matches(&codes);
    let branchless = dfa.matches_branchless(&codes);
    assert_eq!(
        default, expected,
        "default variant wrong: needle={:?} haystack={:?}",
        needle, haystack
    );
    assert_eq!(
        branchless, expected,
        "branchless variant wrong: needle={:?} haystack={:?}",
        needle, haystack
    );
}

/// With symbols so matches can cross symbol boundaries — ensure branchless
/// still agrees with the default path.
#[test]
fn test_contains_branchless_crosses_symbols() -> VortexResult<()> {
    // code 0 = "ab", code 1 = "cd"
    let symbols = [sym(b"ab"), sym(b"cd")];
    let lengths = [2u8, 2];
    let dfa = FlatContainsDfa::new(&symbols, &lengths, b"bc")?;

    // "abcd" via symbols: code 0 ("ab") + code 1 ("cd") — contains "bc"
    let codes = vec![0u8, 1];
    assert!(dfa.matches(&codes));
    assert!(dfa.matches_branchless(&codes));

    // "ab" only → no "bc"
    let codes = vec![0u8];
    assert!(!dfa.matches(&codes));
    assert!(!dfa.matches_branchless(&codes));

    Ok(())
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
// MultiContainsDfa unit tests
// ---------------------------------------------------------------------------

/// No symbols — all bytes escaped. Two segments.
#[test]
fn test_multi_contains_dfa_no_symbols() -> VortexResult<()> {
    let dfa = MultiContainsDfa::new(&[], &[], &[b"ab", b"cd"])?;

    assert!(dfa.matches(&escaped(b"abcd")));
    assert!(dfa.matches(&escaped(b"xxabxxcdxx")));
    assert!(dfa.matches(&escaped(b"abxcd")));
    assert!(!dfa.matches(&escaped(b"cdab"))); // wrong order
    assert!(!dfa.matches(&escaped(b"ab"))); // missing cd
    assert!(!dfa.matches(&escaped(b"cd"))); // missing ab
    assert!(!dfa.matches(&[]));

    Ok(())
}

/// Three segments, all escaped.
#[test]
fn test_multi_contains_dfa_three_segments() -> VortexResult<()> {
    let dfa = MultiContainsDfa::new(&[], &[], &[b"a", b"b", b"c"])?;

    assert!(dfa.matches(&escaped(b"abc")));
    assert!(dfa.matches(&escaped(b"xaxbxcx")));
    assert!(dfa.matches(&escaped(b"abc")));
    assert!(!dfa.matches(&escaped(b"cba"))); // wrong order
    assert!(!dfa.matches(&escaped(b"ab"))); // missing c
    assert!(!dfa.matches(&escaped(b"ac"))); // missing b between a and c

    Ok(())
}

/// With symbols — multi-byte symbols can straddle segment boundaries.
#[test]
fn test_multi_contains_dfa_with_symbols() -> VortexResult<()> {
    // code 0 = "ab", code 1 = "cd"
    let symbols = [sym(b"ab"), sym(b"cd")];
    let lengths = [2u8, 2];
    let dfa = MultiContainsDfa::new(&symbols, &lengths, &[b"ab", b"cd"])?;

    // "abcd" via symbols: code 0 ("ab") + code 1 ("cd")
    assert!(dfa.matches(&[0, 1]));
    // "abcd" all escaped
    assert!(dfa.matches(&escaped(b"abcd")));
    // "ab...cd" with gap via escape
    assert!(dfa.matches(&[0, ESCAPE_CODE, b'x', 1]));
    // Only first segment
    assert!(!dfa.matches(&[0]));
    // Wrong order
    assert!(!dfa.matches(&[1, 0]));

    Ok(())
}

/// KMP overlap within a segment: "abab" has failure [0,0,1,2].
#[test]
fn test_multi_contains_dfa_kmp_within_segment() -> VortexResult<()> {
    let dfa = MultiContainsDfa::new(&[], &[], &[b"abab", b"xy"])?;

    assert!(dfa.matches(&escaped(b"ababxy")));
    assert!(dfa.matches(&escaped(b"xababxyx")));
    // "abababxy" — KMP for "abab" matches at position 0-3, then "xy" at 6-7
    assert!(dfa.matches(&escaped(b"abababxy")));
    assert!(!dfa.matches(&escaped(b"xyabab"))); // wrong order
    assert!(!dfa.matches(&escaped(b"abab"))); // missing xy

    Ok(())
}

/// Greedy-first-match correctness: %ab%ab% on "abab".
/// Find "ab" at 0, then "ab" from position 2 onward — found at 2.
#[test]
fn test_multi_contains_dfa_greedy_correctness() -> VortexResult<()> {
    let dfa = MultiContainsDfa::new(&[], &[], &[b"ab", b"ab"])?;

    assert!(dfa.matches(&escaped(b"abab")));
    assert!(dfa.matches(&escaped(b"xababx")));
    assert!(!dfa.matches(&escaped(b"ab"))); // only one "ab"
    assert!(!dfa.matches(&escaped(b"xabx"))); // only one "ab"

    Ok(())
}

/// State-space limit: total 254 bytes across segments.
#[test]
fn test_multi_contains_dfa_max_total_len() -> VortexResult<()> {
    let seg1 = "a".repeat(127);
    let seg2 = "b".repeat(127);
    let dfa = MultiContainsDfa::new(&[], &[], &[seg1.as_bytes(), seg2.as_bytes()])?;

    let matching = format!("{seg1}{seg2}");
    assert!(dfa.matches(&escaped(matching.as_bytes())));

    let non_matching = format!("{seg2}{seg1}"); // wrong order
    assert!(!dfa.matches(&escaped(non_matching.as_bytes())));

    Ok(())
}

#[test]
fn test_multi_contains_dfa_rejects_over_max() {
    let seg1 = "a".repeat(128);
    let seg2 = "b".repeat(127);
    // total = 255 > MAX_TOTAL_LEN = 254
    assert!(MultiContainsDfa::new(&[], &[], &[seg1.as_bytes(), seg2.as_bytes()]).is_err());
}

/// FsstMatcher integration: %abc%def% should be handled.
#[test]
fn test_multi_contains_matcher_handles() {
    let matcher = FsstMatcher::try_new(&[], &[], b"%abc%def%")
        .unwrap()
        .unwrap();

    assert!(matcher.matches(&escaped(b"abcdef")));
    assert!(matcher.matches(&escaped(b"xxabcxxdefxx")));
    assert!(!matcher.matches(&escaped(b"defabc")));
    assert!(!matcher.matches(&escaped(b"abc")));
}

// ---------------------------------------------------------------------------
// End-to-end edge cases: FSST compress → LIKE → compare booleans
// ---------------------------------------------------------------------------

fn make_fsst_str(strings: &[Option<&str>]) -> FSSTArray {
    let varbin = VarBinArray::from_iter(
        strings.iter().copied(),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    fsst_compress(varbin, len, &dtype, &compressor)
}

fn run_like(array: FSSTArray, pattern_arr: ArrayRef) -> VortexResult<BoolArray> {
    let len = array.len();
    let arr: ArrayRef = array.into_array();
    let result = Like
        .try_new_array(len, LikeOptions::default(), [arr, pattern_arr])?
        .into_array()
        .execute::<Canonical>(&mut SESSION.create_execution_ctx())?;
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
// Suffix patterns (`%suffix`)
#[case(&["hello", "world", "jello"], "%ello", &[true, false, true])]
#[case(&["foobar", "bazbar", "bar", "ba"], "%bar", &[true, true, true, false])]
#[case(&["abc", "xabc", "abcd", ""], "%abc", &[true, true, false, false])]
#[case(&["BRASS", "xBRASS", "BRASSx", "brass"], "%BRASS", &[true, true, false, false])]
#[case(&["aa", "aaa", "aaaa", "ba"], "%aa", &[true, true, true, false])]
// Suffix with KMP overlap: "abab" — "xababab" ends with "abab"
#[case(&["ababab", "abab", "aba", "xabab"], "%abab", &[true, true, false, true])]
// Empty suffix matches everything
#[case(&["abc", "", "xyz"], "%", &[true, true, true])]
// Multi-contains: two segments
#[case(&["abcdef", "abxdef", "defabc", "abc"], "%abc%def%", &[true, false, false, false])]
#[case(&["xxabcxxdefxx", "abcdef", "defabc"], "%abc%def%", &[true, true, false])]
// Multi-contains: three segments (single-char each)
#[case(&["axbxc", "abc", "cba", "ab"], "%a%b%c%", &[true, true, false, false])]
// Multi-contains: greedy first match
#[case(&["abab", "ab", "aba", "xababx"], "%ab%ab%", &[true, false, false, true])]
// Multi-contains: segments don't overlap ("abcdef" has no "cd" after "abc")
#[case(&["abccd", "abcd", "abcdef"], "%abc%cd%", &[true, false, false])]
// Multi-contains: KMP overlap within segment
#[case(&["xxabcabcabcxxdefxx", "abcabcabcdef", "defabcabcabc"], "%abcabcabc%def%", &[true, true, false])]
// Multi-contains with longer segments
#[case(&["hello world goodbye", "hello goodbye", "world hello goodbye"], "%hello%goodbye%", &[true, true, true])]
// Multi-contains: segment appears but not in order
#[case(&["barfoo", "foobar", "fooxbar"], "%foo%bar%", &[false, true, true])]
// Multi-contains: same segment repeated three times ("xaxax" has only 2 'a's)
#[case(&["aaa", "aa", "axaxa", "xaxax"], "%a%a%a%", &[true, false, true, false])]
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
    assert_arrays_eq!(&result, &expected_arr);
    Ok(())
}
