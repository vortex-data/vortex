// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared JNI value decoding for authored and bound expression builders.

use jni::sys::jbyte;
use vortex::dtype::BigCast;
use vortex::dtype::DecimalDType;
use vortex::dtype::i256;
use vortex::error::vortex_err;
use vortex::extension::datetime::TimeUnit;
use vortex::scalar::DecimalValue;
use vortex::scalar_fn::fns::operators::Operator;

use crate::errors::JNIError;

pub(crate) fn parse_op(op: jbyte) -> Result<Operator, JNIError> {
    Ok(match op {
        0 => Operator::Eq,
        1 => Operator::NotEq,
        2 => Operator::Gt,
        3 => Operator::Gte,
        4 => Operator::Lt,
        5 => Operator::Lte,
        6 => Operator::And,
        7 => Operator::Or,
        8 => Operator::Add,
        9 => Operator::Sub,
        10 => Operator::Mul,
        11 => Operator::Div,
        other => throw_runtime!("unknown binary operator code: {other}"),
    })
}

/// Parse a Vortex [`TimeUnit`] from the wire-encoded byte tag.
pub(crate) fn parse_time_unit(tag: jbyte) -> Result<TimeUnit, JNIError> {
    TimeUnit::try_from(tag as u8).map_err(JNIError::from)
}

/// Decode Java `BigInteger.toByteArray()` into the narrowest decimal backing value.
pub(crate) fn decimal_value_from_be_bytes(
    bytes: &[u8],
    dtype: &DecimalDType,
) -> Result<DecimalValue, JNIError> {
    if bytes.is_empty() {
        throw_runtime!("decimal unscaled value must have at least one byte");
    }
    let value = i256_from_twos_complement_be(bytes);
    let required_bits = dtype.required_bit_width();
    if required_bits <= 8 {
        let v =
            BigCast::from(value).ok_or_else(|| vortex_err!("decimal value does not fit in i8"))?;
        Ok(DecimalValue::I8(v))
    } else if required_bits <= 16 {
        let v =
            BigCast::from(value).ok_or_else(|| vortex_err!("decimal value does not fit in i16"))?;
        Ok(DecimalValue::I16(v))
    } else if required_bits <= 32 {
        let v =
            BigCast::from(value).ok_or_else(|| vortex_err!("decimal value does not fit in i32"))?;
        Ok(DecimalValue::I32(v))
    } else if required_bits <= 64 {
        let v =
            BigCast::from(value).ok_or_else(|| vortex_err!("decimal value does not fit in i64"))?;
        Ok(DecimalValue::I64(v))
    } else if required_bits <= 128 {
        let v = value
            .maybe_i128()
            .ok_or_else(|| vortex_err!("decimal value does not fit in i128"))?;
        Ok(DecimalValue::I128(v))
    } else {
        Ok(DecimalValue::I256(value))
    }
}

fn i256_from_twos_complement_be(bytes: &[u8]) -> i256 {
    let mut le = [0u8; 32];
    let len = bytes.len().min(32);
    for (i, b) in bytes.iter().rev().take(len).enumerate() {
        le[i] = *b;
    }
    if !bytes.is_empty() && (bytes[0] & 0x80) != 0 {
        for byte in &mut le[len..] {
            *byte = 0xff;
        }
    }
    i256::from_le_bytes(le)
}
