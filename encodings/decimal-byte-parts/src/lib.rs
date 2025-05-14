mod decimal_byte_parts;

/// This encoding allow compression of decimals using integer compression schemes.
/// Decimals can be compressed by narrowing the signed decimal value into the smallest signed value,
/// then integer compression if that is a value `ptype`, otherwise the decimal can be split into
/// parts.
/// These parts can be individually compressed.
/// This encoding will compress large signed decimals by removing the leading zeroes (after the sign)
/// an i128 decimal could be converted into a [i64, u64] with further narrowing applied to either
/// value.
pub use decimal_byte_parts::*;
