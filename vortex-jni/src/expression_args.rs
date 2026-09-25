// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared JNI value decoding for authored and bound expression builders.

use std::sync::Arc;

use jni::sys::jboolean;
use jni::sys::jbyte;
use vortex::dtype::BigCast;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::extension::ExtDType;
use vortex::dtype::i256;
use vortex::error::vortex_err;
use vortex::extension::datetime::TimeUnit;
use vortex::extension::uuid::Uuid;
use vortex::extension::uuid::UuidMetadata;
use vortex::scalar::DecimalValue;
use vortex::scalar::Scalar;
use vortex::scalar_fn::fns::between::StrictComparison;
use vortex::scalar_fn::fns::merge::DuplicateHandling;
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

/// Parse a merge strategy from its Java wire tag.
pub(crate) fn parse_duplicate_handling(tag: jbyte) -> Result<DuplicateHandling, JNIError> {
    Ok(match tag {
        0 => DuplicateHandling::RightMost,
        1 => DuplicateHandling::Error,
        other => throw_runtime!("unknown duplicate handling code: {other}"),
    })
}

pub(crate) fn strict_from_bool(value: jboolean) -> StrictComparison {
    if value {
        StrictComparison::Strict
    } else {
        StrictComparison::NonStrict
    }
}

/// Number of bytes in a UUID's big-endian representation.
const UUID_BYTE_LEN: usize = 16;

/// Build the version-agnostic UUID extension dtype used by Java literals.
pub(crate) fn uuid_dtype(nullability: Nullability) -> Result<DType, JNIError> {
    let list_size = u32::try_from(UUID_BYTE_LEN)
        .map_err(|_| vortex_err!("UUID byte length {UUID_BYTE_LEN} does not fit in u32"))?;
    let storage_dtype = DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
        list_size,
        nullability,
    );
    let ext = ExtDType::<Uuid>::try_new(UuidMetadata::default(), storage_dtype)?;
    Ok(DType::Extension(ext.erased()))
}

/// Build a non-null UUID scalar from its 16-byte big-endian representation.
pub(crate) fn uuid_scalar(bytes: &[u8]) -> Result<Scalar, JNIError> {
    if bytes.len() != UUID_BYTE_LEN {
        throw_runtime!(
            "UUID literal must be exactly {UUID_BYTE_LEN} bytes, got {}",
            bytes.len()
        );
    }
    let children: Vec<Scalar> = bytes
        .iter()
        .map(|&b| Scalar::primitive(b, Nullability::NonNullable))
        .collect();
    let storage = Scalar::fixed_size_list(
        DType::Primitive(PType::U8, Nullability::NonNullable),
        children,
        Nullability::NonNullable,
    );
    Ok(Scalar::try_new(
        uuid_dtype(Nullability::NonNullable)?,
        storage.into_value(),
    )?)
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
