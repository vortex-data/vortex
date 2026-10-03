// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UTF-8 validation shared by the string arrays.
//!
//! Validating each string on its own costs a call per value, which dominates reading a short
//! string column. A buffer that is valid UTF-8 as a whole has valid substrings exactly where they
//! start and end on character boundaries, so the arrays validate a data buffer once and then
//! check two bytes per string, falling back to per-string validation when the buffer as a whole
//! is not valid (bytes no string references are allowed to be anything).

/// Whether `index` falls on a UTF-8 character boundary of `bytes`: the end of the buffer, or a
/// byte that is not a continuation byte.
#[inline]
pub(crate) fn is_char_boundary(bytes: &[u8], index: usize) -> bool {
    match bytes.get(index) {
        // Continuation bytes are 0b10xx_xxxx, which as `i8` are below -0x40.
        Some(&byte) => (byte as i8) >= -0x40,
        None => index == bytes.len(),
    }
}

/// Whether validating a whole buffer is worth it for the strings that reference `referenced`
/// bytes of it: past twice that, the buffer is mostly bytes no string needs, as after slicing.
#[inline]
pub(crate) fn worth_validating_whole(buffer_len: usize, referenced: u64) -> bool {
    (buffer_len as u64) <= referenced.saturating_mul(2)
}

/// Validates a short string, taking the ASCII case without a call into the validator.
#[inline]
pub(crate) fn is_utf8_short(bytes: &[u8]) -> bool {
    bytes.is_ascii() || simdutf8::basic::from_utf8(bytes).is_ok()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::is_char_boundary;

    #[rstest]
    #[case(b"", 0, true)]
    #[case(b"ab", 1, true)]
    #[case(b"ab", 2, true)]
    #[case(b"ab", 3, false)]
    #[case("é".as_bytes(), 0, true)]
    #[case("é".as_bytes(), 1, false)]
    #[case("é".as_bytes(), 2, true)]
    fn char_boundary(#[case] bytes: &[u8], #[case] index: usize, #[case] expected: bool) {
        assert_eq!(is_char_boundary(bytes, index), expected);
    }
}
