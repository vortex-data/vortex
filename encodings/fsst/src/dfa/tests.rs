// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fsst::ESCAPE_CODE;
use fsst::Symbol;
use vortex_error::VortexResult;

use super::FsstMatcher;
use super::LikeKind;
use super::flat_contains::FlatContainsDfa;
use super::prefix::FlatPrefixDfa;

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
        LikeKind::parse("http%"),
        Some(LikeKind::Prefix("http"))
    ));
    assert!(matches!(
        LikeKind::parse("%needle%"),
        Some(LikeKind::Contains("needle"))
    ));
    assert!(matches!(LikeKind::parse("%"), Some(LikeKind::Prefix(""))));
    // Suffix and underscore patterns are not supported.
    assert!(LikeKind::parse("%suffix").is_none());
    assert!(LikeKind::parse("a_c").is_none());
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
    let matcher = FsstMatcher::try_new(&[], &[], "abcdefghijklm%")
        .unwrap()
        .unwrap();

    assert!(matcher.matches(&escaped(b"abcdefghijklm")));
    assert!(!matcher.matches(&escaped(b"abcdefghijklx")));
}

#[test]
fn test_prefix_pushdown_len_14_now_handled() {
    // 14-byte prefix is now handled by FlatPrefixDfa (was rejected by shift-packed).
    assert!(
        FsstMatcher::try_new(&[], &[], "abcdefghijklmn%")
            .unwrap()
            .is_some()
    );
}

#[test]
fn test_prefix_pushdown_long_prefix() -> VortexResult<()> {
    let prefix = "a".repeat(FlatPrefixDfa::MAX_PREFIX_LEN);
    let pattern = format!("{prefix}%");
    let matcher = FsstMatcher::try_new(&[], &[], &pattern)?.unwrap();

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
    assert!(FsstMatcher::try_new(&[], &[], &pattern).unwrap().is_none());
}

#[test]
fn test_contains_pushdown_len_254_with_escapes() {
    let needle = "a".repeat(FlatContainsDfa::MAX_NEEDLE_LEN);
    let pattern = format!("%{needle}%");
    let matcher = FsstMatcher::try_new(&[], &[], &pattern).unwrap().unwrap();

    assert!(matcher.matches(&escaped(needle.as_bytes())));

    let mut mismatch = needle.into_bytes();
    mismatch[FlatContainsDfa::MAX_NEEDLE_LEN - 1] = b'b';
    assert!(!matcher.matches(&escaped(&mismatch)));
}

#[test]
fn test_contains_pushdown_rejects_len_255() {
    let needle = "a".repeat(FlatContainsDfa::MAX_NEEDLE_LEN + 1);
    let pattern = format!("%{needle}%");
    assert!(FsstMatcher::try_new(&[], &[], &pattern).unwrap().is_none());
}
