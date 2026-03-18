// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fsst::ESCAPE_CODE;

use super::FsstMatcher;
use super::LikeKind;
use super::flat_contains::FlatContainsDfa;
use super::prefix::FsstPrefixDfa;

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

#[test]
fn test_prefix_pushdown_len_13_with_escapes() {
    let matcher = FsstMatcher::try_new(&[], &[], "abcdefghijklm%")
        .unwrap()
        .unwrap();

    assert!(matcher.matches(&escaped(b"abcdefghijklm")));
    assert!(!matcher.matches(&escaped(b"abcdefghijklx")));
}

#[test]
fn test_prefix_pushdown_rejects_len_14() {
    debug_assert_eq!(FsstPrefixDfa::MAX_PREFIX_LEN, 13);
    assert!(
        FsstMatcher::try_new(&[], &[], "abcdefghijklmn%")
            .unwrap()
            .is_none()
    );
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
