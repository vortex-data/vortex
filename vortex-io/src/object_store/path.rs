// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion from a literal object key to an `object_store` [`Path`].

use object_store::path::Path;

/// Convert a literal object key (or filesystem path) into an object-store [`Path`].
///
/// Object stores key their objects *literally*: an object named `a~b.vortex` has the key
/// `a~b.vortex`, and `LocalFileSystem` likewise surfaces real filenames verbatim. [`Path::parse`]
/// preserves those characters, whereas [`Path::from`] percent-encodes `~`, `%`, `[`, `]`, `#`,
/// `{`, `}`, `^`, `|`, `*`, `?`, `<`, `>`, `"`, `` ` `` and `\` — turning `a~b.vortex` into the
/// key `a%7Eb.vortex`, which no real object has. Worse, the request layer then percent-encodes
/// that `%` again (`%7E` → `%257E`), so the object is doubly encoded and the store 404s.
///
/// Using `parse` keeps caller inputs, the keys returned by listings, and the keys sent on the wire
/// on a single literal representation, so a key from `list`/`head` round-trips back through a read
/// unchanged.
///
/// `parse` rejects empty, `.`, and `..` segments; for those we fall back to [`Path::from`], which
/// normalizes them (this never applies to a key a listing produced, so it cannot break a
/// round-trip).
pub fn object_path_from_literal(path: &str) -> Path {
    Path::parse(path).unwrap_or_else(|_| Path::from(path))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// Characters `Path::from` percent-encodes must survive verbatim (regression for #9420).
    /// `Path::from` would turn `~` into the key `%7E`, which the request layer then encodes again.
    #[rstest]
    #[case::tilde("dir/a~b.vortex")]
    #[case::double_tilde("dir/~~all~0.vortex")]
    #[case::percent_literal("dir/a%20b.vortex")]
    #[case::brackets("dir/a[1].vortex")]
    #[case::hash("dir/a#b.vortex")]
    #[case::braces("dir/a{x}.vortex")]
    #[case::caret("dir/a^b.vortex")]
    #[case::pipe("dir/a|b.vortex")]
    #[case::star("dir/a*b.vortex")]
    #[case::question("dir/a?b.vortex")]
    #[case::backslash("dir/a\\b.vortex")]
    #[case::angles("dir/a<b>.vortex")]
    #[case::backtick("dir/a`b.vortex")]
    #[case::quote("dir/a\"b.vortex")]
    #[case::space("dir/a b.vortex")]
    #[case::unicode("dir/é~ü.vortex")]
    #[case::plain("dir/plain.vortex")]
    fn test_literal_round_trip(#[case] key: &str) {
        assert_eq!(object_path_from_literal(key).as_ref(), key);
    }

    /// Relative and empty segments cannot be represented literally, so they fall back to the
    /// normalizing conversion rather than erroring.
    #[rstest]
    #[case::dot("a/./b", "a/%2E/b")]
    #[case::dotdot("a/../b", "a/%2E%2E/b")]
    #[case::empty_segment("a//b", "a/b")]
    fn test_normalizing_fallback(#[case] key: &str, #[case] expected: &str) {
        assert_eq!(object_path_from_literal(key).as_ref(), expected);
    }

    /// A leading slash is not a segment; both conversions drop it.
    #[test]
    fn test_leading_slash_stripped() {
        assert_eq!(
            object_path_from_literal("/a~b.vortex").as_ref(),
            "a~b.vortex"
        );
    }
}
