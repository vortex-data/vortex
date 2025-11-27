// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

/// A wrapper around a string that implements `Display` for a string while removing all unicode
/// escape sequences.
#[derive(Debug)]
pub struct StringEscape<'a>(pub &'a str);

impl Display for StringEscape<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.escape_debug())
    }
}

#[cfg(test)]
mod tests {
    use crate::aliases::string_escape::StringEscape;

    #[test]
    fn test_no_escape_string() {
        let s = StringEscape("hello");
        assert_eq!(s.to_string(), "hello");
    }

    #[test]
    fn test_heart_string() {
        let s = StringEscape("hello ♡");
        assert_eq!(s.to_string(), "hello ♡");
    }

    #[test]
    fn test_string_escape() {
        let s = StringEscape("\"\n\t\r\\\0\x1f\x7f");
        assert_eq!(s.to_string(), "\\\"\\n\\t\\r\\\\\\0\\u{1f}\\u{7f}");
    }
}
