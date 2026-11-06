// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Text normalization shared by the full-text and trigram indexes and the `text_match` function.
//!
//! The index and the predicate it answers must agree exactly on tokenization, otherwise an exact
//! index answer would differ from evaluating the predicate on the data. Both sides call into this
//! module and nothing else.

/// Split `text` into lowercase tokens: maximal runs of Unicode alphanumeric characters.
pub fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
}

/// Pack three bytes into a trigram key.
pub fn trigram_key(bytes: [u8; 3]) -> u32 {
    (u32::from(bytes[0]) << 16) | (u32::from(bytes[1]) << 8) | u32::from(bytes[2])
}

/// Every byte trigram of `bytes`, after ASCII lowercasing, in order and possibly repeated.
///
/// Lowercasing only ASCII keeps byte positions stable, and is a superset-safe normalization: a
/// case-sensitive match is also a match of the lowercased text.
pub fn trigrams(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes.windows(3).map(|w| {
        trigram_key([
            w[0].to_ascii_lowercase(),
            w[1].to_ascii_lowercase(),
            w[2].to_ascii_lowercase(),
        ])
    })
}

/// The literal runs of a SQL `LIKE` pattern, or `None` if the pattern uses an escape character,
/// which this parser does not model.
///
/// Every row matching the pattern contains each returned run as a substring.
pub fn like_literal_runs(pattern: &str) -> Option<Vec<&str>> {
    if pattern.contains('\\') {
        return None;
    }
    Some(
        pattern
            .split(['%', '_'])
            .filter(|run| !run.is_empty())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_on_non_alphanumerics() {
        let tokens: Vec<_> = tokenize("Hello, World! It's 2026-11-05.").collect();
        assert_eq!(
            tokens,
            ["hello", "world", "it", "s", "2026", "11", "05"].map(String::from)
        );
    }

    #[test]
    fn trigrams_lowercase_ascii() {
        let grams: Vec<_> = trigrams(b"AbCd").collect();
        assert_eq!(grams, vec![trigram_key(*b"abc"), trigram_key(*b"bcd")]);
    }

    #[test]
    fn like_runs() {
        assert_eq!(like_literal_runs("%foo_bar%"), Some(vec!["foo", "bar"]));
        assert_eq!(like_literal_runs("%%"), Some(vec![]));
        assert_eq!(like_literal_runs("a\\%b"), None);
    }
}
