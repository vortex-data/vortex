// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads strings as decimals, with the syntax and rounding of Arrow's `parse_decimal`.

/// Parses `string` as the unscaled value of a decimal of `precision` and `scale`.
///
/// The string is a number with an optional sign, decimal point and exponent, with whitespace
/// around it. Digits past `scale` round half away from zero. Returns `None` if `string` is not a
/// number or its value does not fit `precision`.
pub(super) fn parse_decimal(string: &str, precision: u8, scale: i8) -> Option<i128> {
    let (negative, number) = split_sign(string.as_bytes().trim_ascii());

    let (mantissa, scale) = match number.iter().position(|byte| matches!(byte, b'e' | b'E')) {
        Some(at) => {
            let exponent = parse_exponent(&number[at + 1..])?;
            (&number[..at], i64::from(scale).saturating_add(exponent))
        }
        None => (number, i64::from(scale)),
    };

    let magnitude = parse_mantissa(mantissa, scale)?;

    // Every value of an `i128` fits a precision that `10^precision` overflows.
    let fits = 10i128
        .checked_pow(u32::from(precision))
        .is_none_or(|bound| magnitude < bound);
    fits.then_some(if negative { -magnitude } else { magnitude })
}

fn split_sign(bytes: &[u8]) -> (bool, &[u8]) {
    match bytes.split_first() {
        Some((b'-', rest)) => (true, rest),
        Some((b'+', rest)) => (false, rest),
        _ => (false, bytes),
    }
}

/// Parses the digits of an exponent with an optional sign, saturating at the bounds of `i64`.
fn parse_exponent(exponent: &[u8]) -> Option<i64> {
    let (negative, digits) = split_sign(exponent);
    if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }

    let value = digits.iter().fold(0i64, |value, digit| {
        value
            .saturating_mul(10)
            .saturating_add(i64::from(digit - b'0'))
    });
    Some(if negative { -value } else { value })
}

/// Parses digits with at most one decimal point as the magnitude of a decimal of `scale`.
fn parse_mantissa(mantissa: &[u8], scale: i64) -> Option<i128> {
    let (integer, fraction) = match mantissa.iter().position(|&byte| byte == b'.') {
        Some(dot) => (&mantissa[..dot], &mantissa[dot + 1..]),
        None => (mantissa, &[][..]),
    };
    let is_digits = |digits: &[u8]| digits.iter().all(u8::is_ascii_digit);
    if !is_digits(integer) || !is_digits(fraction) || (integer.is_empty() && fraction.is_empty()) {
        return None;
    }

    // A negative scale drops the last `-scale` integer digits. A value with fewer integer digits
    // than that rounds to zero.
    let (integer_kept, fraction_kept, round) = if scale >= 0 {
        let fraction_kept = usize::try_from(scale).unwrap_or(usize::MAX);
        (integer.len(), fraction_kept.min(fraction.len()), true)
    } else {
        let kept = i64::try_from(integer.len()).ok()?.saturating_add(scale);
        match usize::try_from(kept) {
            Ok(kept) => (kept, 0, true),
            Err(_) => (0, 0, false),
        }
    };

    let mut value = integer[..integer_kept]
        .iter()
        .chain(&fraction[..fraction_kept])
        .try_fold(0i128, |value, digit| {
            value.checked_mul(10)?.checked_add(i128::from(digit - b'0'))
        })?;

    let missing = scale - i64::try_from(fraction_kept).ok()?;
    if missing > 0 && value != 0 {
        value = value.checked_mul(10i128.checked_pow(u32::try_from(missing).ok()?)?)?;
    }

    let first_dropped = integer[integer_kept..]
        .iter()
        .chain(&fraction[fraction_kept..])
        .next();
    if round && first_dropped.is_some_and(|&digit| digit >= b'5') {
        value = value.checked_add(1)?;
    }

    Some(value)
}
