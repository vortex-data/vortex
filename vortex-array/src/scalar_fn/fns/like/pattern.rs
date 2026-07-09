// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compiled SQL LIKE patterns evaluated over UTF-8 byte slices.
//!
//! A [`LikePattern`] is compiled once per pattern and then evaluated per value. Patterns that
//! reduce to plain equality, prefix, suffix, or substring searches use direct byte comparisons
//! (with `memchr`'s SIMD substring search for the contains case); everything else falls back to
//! a regex translated from the LIKE pattern.
//!
//! The pattern grammar and its regex translation mirror the `arrow-string` LIKE kernels so that
//! results are identical to the previous Arrow-backed implementation: `%` matches any sequence
//! of characters (including across newlines), `_` matches exactly one character, and `\` escapes
//! the next character.

use memchr::memchr3;
use memchr::memmem::Finder;
use regex::bytes::Regex;
use regex::bytes::RegexBuilder;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

/// A LIKE pattern compiled for repeated evaluation against UTF-8 haystacks.
pub(crate) enum LikePattern {
    /// The pattern has no wildcards: plain byte equality.
    Eq(Vec<u8>),
    /// `prefix%`: byte prefix comparison.
    StartsWith(Vec<u8>),
    /// `%suffix`: byte suffix comparison.
    EndsWith(Vec<u8>),
    /// `%needle%`: SIMD substring search, along with the needle length.
    Contains(Finder<'static>, usize),
    /// Equality ignoring ASCII case (ILIKE over ASCII-only data).
    IEqAscii(Vec<u8>),
    /// Prefix comparison ignoring ASCII case (ILIKE over ASCII-only data).
    IStartsWithAscii(Vec<u8>),
    /// Suffix comparison ignoring ASCII case (ILIKE over ASCII-only data).
    IEndsWithAscii(Vec<u8>),
    /// Everything else: the LIKE pattern translated to an anchored regex.
    Regex(Regex),
}

impl LikePattern {
    /// Compile `pattern`, with `case_insensitive` selecting ILIKE semantics.
    ///
    /// `ascii_haystack` enables the ASCII-only case-insensitive fast paths and must only be
    /// `true` when every haystack this pattern will be evaluated against is pure ASCII:
    /// full case folding can match ASCII pattern characters against non-ASCII haystack
    /// characters (e.g. `k` folds to U+212A KELVIN SIGN), which the ASCII fast paths would
    /// miss.
    pub(crate) fn compile(
        pattern: &str,
        case_insensitive: bool,
        ascii_haystack: bool,
    ) -> VortexResult<Self> {
        if case_insensitive {
            if ascii_haystack && pattern.is_ascii() {
                if !contains_like_pattern(pattern) {
                    return Ok(Self::IEqAscii(pattern.as_bytes().to_vec()));
                } else if pattern.ends_with('%')
                    && !contains_like_pattern(&pattern[..pattern.len() - 1])
                {
                    return Ok(Self::IStartsWithAscii(
                        pattern.as_bytes()[..pattern.len() - 1].to_vec(),
                    ));
                } else if pattern.starts_with('%') && !contains_like_pattern(&pattern[1..]) {
                    return Ok(Self::IEndsWithAscii(pattern.as_bytes()[1..].to_vec()));
                }
            }
            return Ok(Self::Regex(regex_like(pattern, true)?));
        }

        Ok(if !contains_like_pattern(pattern) {
            Self::Eq(pattern.as_bytes().to_vec())
        } else if pattern.ends_with('%') && !contains_like_pattern(&pattern[..pattern.len() - 1]) {
            Self::StartsWith(pattern.as_bytes()[..pattern.len() - 1].to_vec())
        } else if pattern.starts_with('%') && !contains_like_pattern(&pattern[1..]) {
            Self::EndsWith(pattern.as_bytes()[1..].to_vec())
        } else if pattern.starts_with('%')
            && pattern.ends_with('%')
            && !contains_like_pattern(&pattern[1..pattern.len() - 1])
        {
            Self::Contains(
                Finder::new(&pattern.as_bytes()[1..pattern.len() - 1]).into_owned(),
                pattern.len() - 2,
            )
        } else {
            Self::Regex(regex_like(pattern, false)?)
        })
    }

    /// Evaluate this pattern against `haystack`, which must be valid UTF-8.
    #[inline]
    pub(crate) fn matches(&self, haystack: &[u8]) -> bool {
        match self {
            Self::Eq(needle) => needle == haystack,
            Self::StartsWith(needle) => {
                needle.len() <= haystack.len() && &haystack[..needle.len()] == needle.as_slice()
            }
            Self::EndsWith(needle) => {
                needle.len() <= haystack.len()
                    && &haystack[haystack.len() - needle.len()..] == needle.as_slice()
            }
            Self::Contains(finder, needle_len) => {
                haystack.len() >= *needle_len && finder.find(haystack).is_some()
            }
            Self::IEqAscii(needle) => haystack.eq_ignore_ascii_case(needle),
            Self::IStartsWithAscii(needle) => {
                needle.len() <= haystack.len()
                    && haystack[..needle.len()].eq_ignore_ascii_case(needle)
            }
            Self::IEndsWithAscii(needle) => {
                needle.len() <= haystack.len()
                    && haystack[haystack.len() - needle.len()..].eq_ignore_ascii_case(needle)
            }
            Self::Regex(regex) => regex.is_match(haystack),
        }
    }
}

/// Returns whether `pattern` contains a wildcard (`%`, `_`) or an escape (`\`).
fn contains_like_pattern(pattern: &str) -> bool {
    memchr3(b'%', b'_', b'\\', pattern.as_bytes()).is_some()
}

/// Transforms a LIKE `pattern` into an equivalent anchored regex.
///
/// This is a port of the `arrow-string` translation so that pattern semantics are unchanged:
///
/// 1. `%` => `.*` (a leading/trailing `%` truncates the anchor instead, e.g. `%foo%` => `foo`
///    rather than `^.*foo.*$`);
/// 2. `_` => `.`;
/// 3. regex meta characters are escaped so they match literally;
/// 4. `\x` matches `x` literally (a trailing `\` matches a literal backslash).
///
/// The regex runs over the haystack bytes directly (`regex::bytes`) in Unicode mode, which
/// matches identically to a `&str` regex on valid UTF-8 input.
fn regex_like(pattern: &str, case_insensitive: bool) -> VortexResult<Regex> {
    let mut result = String::with_capacity(pattern.len() * 2);
    let mut chars_iter = pattern.chars().peekable();
    match chars_iter.peek() {
        // If the pattern starts with `%`, avoid starting the regex with a slow but
        // meaningless `^.*`.
        Some('%') => {
            chars_iter.next();
        }
        _ => result.push('^'),
    };

    while let Some(c) = chars_iter.next() {
        match c {
            '\\' => {
                match chars_iter.peek() {
                    Some(&next) => {
                        if regex_syntax::is_meta_character(next) {
                            result.push('\\');
                        }
                        result.push(next);
                        // Skip the next char as it is already appended.
                        chars_iter.next();
                    }
                    None => {
                        // A trailing backslash matches a literal backslash.
                        result.push('\\');
                        result.push('\\');
                    }
                }
            }
            '%' => result.push_str(".*"),
            '_' => result.push('.'),
            c => {
                if regex_syntax::is_meta_character(c) {
                    result.push('\\');
                }
                result.push(c);
            }
        }
    }
    // Instead of ending the regex with `.*$` and making it needlessly slow, just end it.
    if result.ends_with(".*") {
        result.pop();
        result.pop();
    } else {
        result.push('$');
    }
    RegexBuilder::new(&result)
        .case_insensitive(case_insensitive)
        .dot_matches_new_line(true)
        .build()
        .map_err(|e| vortex_err!("Unable to build regex from LIKE pattern: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn like(pattern: &str) -> LikePattern {
        LikePattern::compile(pattern, false, false).unwrap()
    }

    #[test]
    fn compile_fast_paths() {
        assert!(matches!(like("foo"), LikePattern::Eq(_)));
        assert!(matches!(like("foo%"), LikePattern::StartsWith(_)));
        assert!(matches!(like("%foo"), LikePattern::EndsWith(_)));
        assert!(matches!(like("%foo%"), LikePattern::Contains(..)));
        assert!(matches!(like("%"), LikePattern::StartsWith(_)));
        assert!(matches!(like("%%"), LikePattern::Contains(..)));
        assert!(matches!(like("f%o"), LikePattern::Regex(_)));
        assert!(matches!(like("f_o"), LikePattern::Regex(_)));
        assert!(matches!(like(r"foo\%"), LikePattern::Regex(_)));
    }

    #[test]
    fn regex_translation() {
        // (pattern, expected regex) pairs from the arrow-string translation.
        let cases = [
            (r"%foobar%", r"foobar"),
            (r"foo%bar", r"^foo.*bar$"),
            (r"foo_bar", r"^foo.bar$"),
            (r"\%\_", r"^%_$"),
            (r"\a", r"^a$"),
            (r"\\%", r"^\\"),
            (r"\\a", r"^\\a$"),
            (r".", r"^\.$"),
            (r"$", r"^\$$"),
            (r"\\", r"^\\$"),
        ];
        for (pattern, expected) in cases {
            assert_eq!(regex_like(pattern, false).unwrap().to_string(), expected);
        }
    }

    #[test]
    fn matches_wildcards() {
        assert!(like("h_llo w%d").matches(b"hello world"));
        assert!(like("h_llo w%d").matches("h\u{00a3}llo wd".as_bytes()));
        assert!(!like("h_llo w%d").matches(b"hxxllo world"));
        // `%` crosses newlines.
        assert!(like("hello%world").matches(b"hello\nworld"));
        // `_` matches exactly one character, of any width.
        assert!(like("_").matches("\u{00a3}".as_bytes()));
        assert!(!like("_").matches("\u{00a3}x".as_bytes()));
    }

    #[test]
    fn matches_escapes() {
        assert!(like(r"100\%").matches(b"100%"));
        assert!(!like(r"100\%").matches(b"100x"));
        assert!(like(r"a\_b").matches(b"a_b"));
        assert!(!like(r"a\_b").matches(b"axb"));
        // A trailing backslash matches a literal backslash.
        assert!(like("trailing\\").matches(b"trailing\\"));
    }

    #[test]
    fn matches_case_insensitive() {
        let ilike = |pattern: &str| LikePattern::compile(pattern, true, false).unwrap();
        assert!(ilike("hello%").matches(b"HELLO WORLD"));
        assert!(ilike("%WORLD").matches(b"hello world"));
        // Full case folding: the ASCII pattern `k` matches U+212A KELVIN SIGN.
        assert!(ilike("k").matches("\u{212a}".as_bytes()));
        // Greek sigma case folds across all three forms.
        assert!(ilike("\u{03c3}").matches("\u{03a3}".as_bytes()));
        assert!(ilike("\u{03c3}").matches("\u{03c2}".as_bytes()));

        // The ASCII fast paths agree with the regex on ASCII haystacks.
        let ascii = |pattern: &str| LikePattern::compile(pattern, true, true).unwrap();
        assert!(matches!(ascii("abc"), LikePattern::IEqAscii(_)));
        assert!(matches!(ascii("abc%"), LikePattern::IStartsWithAscii(_)));
        assert!(matches!(ascii("%abc"), LikePattern::IEndsWithAscii(_)));
        assert!(ascii("abc").matches(b"AbC"));
        assert!(ascii("abc%").matches(b"ABCDEF"));
        assert!(ascii("%def").matches(b"ABCDEF"));
        assert!(!ascii("abc").matches(b"AbCd"));
    }
}
