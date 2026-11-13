// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;

/// Variants of the LIKE filter that we know how to turn into a stats pruning predicate.
#[derive(Debug, PartialEq)]
pub(crate) enum LikeVariant<'a> {
    Exact(Cow<'a, str>),
    Prefix(Cow<'a, str>),
}

impl<'a> LikeVariant<'a> {
    /// Parse a LIKE pattern string into its relevant variant
    pub(crate) fn from_str(string: &'a str) -> Option<LikeVariant<'a>> {
        let mut literal = None;
        let mut chars = string.char_indices();

        while let Some((idx, c)) = chars.next() {
            match c {
                '\\' => {
                    let literal = literal.get_or_insert_with(|| string[..idx].to_string());
                    match chars.next() {
                        Some((_, escaped)) => literal.push(escaped),
                        None => literal.push('\\'),
                    }
                }
                '%' | '_' => {
                    return match literal {
                        Some(literal) => (!literal.is_empty())
                            .then_some(LikeVariant::Prefix(Cow::Owned(literal))),
                        None => {
                            (idx != 0).then_some(LikeVariant::Prefix(Cow::Borrowed(&string[..idx])))
                        }
                    };
                }
                c => {
                    if let Some(literal) = &mut literal {
                        literal.push(c);
                    }
                }
            }
        }

        Some(match literal {
            Some(literal) => LikeVariant::Exact(Cow::Owned(literal)),
            None => LikeVariant::Exact(Cow::Borrowed(string)),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::LikeVariant;

    fn assert_borrowed_exact(pattern: &str, expected: &str) {
        let Some(LikeVariant::Exact(actual)) = LikeVariant::from_str(pattern) else {
            panic!("expected borrowed exact pattern");
        };
        assert!(matches!(actual, Cow::Borrowed(_)));
        assert_eq!(actual.as_ref(), expected);
    }

    fn assert_owned_exact(pattern: &str, expected: &str) {
        let Some(LikeVariant::Exact(actual)) = LikeVariant::from_str(pattern) else {
            panic!("expected owned exact pattern");
        };
        assert!(matches!(actual, Cow::Owned(_)));
        assert_eq!(actual.as_ref(), expected);
    }

    fn assert_borrowed_prefix(pattern: &str, expected: &str) {
        let Some(LikeVariant::Prefix(actual)) = LikeVariant::from_str(pattern) else {
            panic!("expected borrowed prefix pattern");
        };
        assert!(matches!(actual, Cow::Borrowed(_)));
        assert_eq!(actual.as_ref(), expected);
    }

    fn assert_owned_prefix(pattern: &str, expected: &str) {
        let Some(LikeVariant::Prefix(actual)) = LikeVariant::from_str(pattern) else {
            panic!("expected owned prefix pattern");
        };
        assert!(matches!(actual, Cow::Owned(_)));
        assert_eq!(actual.as_ref(), expected);
    }

    #[test]
    fn test_like_variant_borrowed_patterns() {
        assert_borrowed_exact("simple", "simple");
        assert_borrowed_prefix("prefix%", "prefix");
        assert_borrowed_prefix("first%rest_stuff", "first");
    }

    #[test]
    fn test_like_variant_escaped_patterns() {
        assert_owned_prefix(r"\%%", "%");
        assert_owned_prefix(r"\_%", "_");
        assert_owned_prefix(r"\\%", "\\");
        assert_owned_exact(r"\%", "%");
        assert_owned_exact("trailing\\", "trailing\\");
    }

    #[test]
    fn test_like_variant_unsupported_patterns() {
        assert_eq!(LikeVariant::from_str("%suffix"), None);
        assert_eq!(LikeVariant::from_str(r"%\%%"), None);
        assert_eq!(LikeVariant::from_str("_pattern"), None);
    }
}
