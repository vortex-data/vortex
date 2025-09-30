// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Implements different tokenization strategies for the Split-Block Bloom Filter builtin to Vortex.
//!
//! We attempt to tokenize "words", which we define as adjacent collections of alphanumeric
//! characters.

use itertools::Itertools;

/// Tokenize the provided input for insertion into a Bloom filter.
///
/// We tokenize along roughly "word" boundaries, with non-alphanumeric chars split into individual
/// tokens.
///
/// ```rust
///
/// # use vortex_layout::layouts::dict::bloom::sbbf::tokenize;
///
/// // Simple case: ASCII sentence
/// assert_eq!(tokenize("hello world"), vec!["hello", " ", "world"]);
///
/// // Also handles Unicode
/// assert_eq!(tokenize("Liberté Égalité Fraternité"), vec!["Liberté", " ", "Égalité", " ", "Fraternité"]);
///
/// // URLs
/// assert_eq!(tokenize("www.google.com"), vec!["www", ".", "google", ".", "com"]);
/// ```
pub fn tokenize(input: &str) -> Vec<&str> {
    let mut result = vec![];

    let mut token_start = 0;

    // For
    let token_starts = input.char_indices().map(|(idx, _)| idx).collect_vec();
    let token_ends = input
        .char_indices()
        .map(|(idx, _)| idx)
        .skip(1)
        .chain([input.len()])
        .collect_vec();

    // edge case: google.
    // we want to tokenize as "google", "."
    for ((c, byte_start), byte_end) in input.chars().zip(token_starts).zip(token_ends) {
        if c.is_alphanumeric() {
            continue;
        }

        // if non-alpha, finish and then push new token.
        if token_start < byte_start {
            result.push(&input[token_start..byte_start]);
        }
        result.push(&input[byte_start..byte_end]);

        token_start = byte_end;
    }

    if token_start < input.len() {
        result.push(&input[token_start..]);
    }

    result
}

pub fn tokenize_contains(input: &str) -> Vec<&str> {
    let mut raw = tokenize(input);

    // Remove the first and last tokens, in case this is a substring query.
    if !raw.is_empty() {
        raw.remove(0);
    }

    if !raw.is_empty() {
        raw.pop();
    }

    raw
}

pub fn tokenize_starts_with(input: &str) -> Vec<&str> {
    // Starts with will skip the final token in the query string.
    let mut raw = tokenize(input);
    raw.pop();

    raw
}

pub fn tokenize_ends_with(input: &str) -> Vec<&str> {
    let mut raw = tokenize(input);
    raw.remove(0);

    raw
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_for_insert() {
        // spellchecker:off

        // Let's have a little fun
        let sentence = "thîs 👀 ANd...THÉSË 🚀😂🫡";

        assert_eq!(
            tokenize(sentence),
            vec![
                "thîs", " ", "👀", " ", "ANd", ".", ".", ".", "THÉSË", " ", "🚀", "😂", "🫡",
            ],
        );

        // spellchecker:on
    }

    #[test]
    fn test_tokenizer_for_query() {
        let phrase = "www.google.com";

        assert_eq!(tokenize(phrase), vec!["www", ".", "google", ".", "com"],);

        assert_eq!(tokenize_contains(phrase), vec![".", "google", "."],);

        assert_eq!(
            tokenize_starts_with(phrase),
            vec!["www", ".", "google", "."],
        );

        assert_eq!(tokenize_ends_with(phrase), vec![".", "google", ".", "com"],);
    }
}
