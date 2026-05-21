// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Protobuf serialization and deserialization for scalars.

use num_traits::ToBytes;
use num_traits::ToPrimitive;
use prost::Message;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_proto::scalar as pb;
use vortex_proto::scalar::ListValue;
use vortex_proto::scalar::scalar_value::Kind;
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::dtype::PType;
use crate::dtype::half::f16;
use crate::dtype::i256;
use crate::scalar::DecimalValue;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Serialize INTO proto.
////////////////////////////////////////////////////////////////////////////////////////////////////

impl From<&Scalar> for pb::Scalar {
    fn from(value: &Scalar) -> Self {
        pb::Scalar {
            dtype: Some(
                (value.dtype())
                    .try_into()
                    .vortex_expect("Failed to convert DType to proto"),
            ),
            value: Some(Box::new(ScalarValue::to_proto(value.value()))),
        }
    }
}

impl ScalarValue {
    /// Ideally, we would not have this function and instead implement this `From` implementation:
    ///
    /// ```ignore
    /// impl From<Option<&ScalarValue>> for pb::ScalarValue { ... }
    /// ```
    ///
    /// However, we are not allowed to do this because of the Orphan rule (`Option` and
    /// `pb::ScalarValue` are not types defined in this crate). So we must make this a method on
    /// `vortex_array::scalar::ScalarValue` directly.
    pub fn to_proto(this: Option<&Self>) -> pb::ScalarValue {
        match this {
            None => pb::ScalarValue {
                kind: Some(Kind::NullValue(0)),
            },
            Some(this) => pb::ScalarValue::from(this),
        }
    }

    /// Serialize an optional [`ScalarValue`] to protobuf bytes (handles null values).
    pub fn to_proto_bytes<B: Default + bytes::BufMut>(value: Option<&ScalarValue>) -> B {
        let proto = Self::to_proto(value);
        let mut buf = B::default();
        proto
            .encode(&mut buf)
            .vortex_expect("Failed to encode scalar value");
        buf
    }
}

impl From<&ScalarValue> for pb::ScalarValue {
    fn from(value: &ScalarValue) -> Self {
        match value {
            ScalarValue::Bool(v) => pb::ScalarValue {
                kind: Some(Kind::BoolValue(*v)),
            },
            ScalarValue::Primitive(v) => pb::ScalarValue::from(v),
            ScalarValue::Decimal(v) => {
                let inner_value = match v {
                    DecimalValue::I8(v) => v.to_le_bytes().to_vec(),
                    DecimalValue::I16(v) => v.to_le_bytes().to_vec(),
                    DecimalValue::I32(v) => v.to_le_bytes().to_vec(),
                    DecimalValue::I64(v) => v.to_le_bytes().to_vec(),
                    DecimalValue::I128(v128) => v128.to_le_bytes().to_vec(),
                    DecimalValue::I256(v256) => v256.to_le_bytes().to_vec(),
                };

                pb::ScalarValue {
                    kind: Some(Kind::BytesValue(inner_value)),
                }
            }
            ScalarValue::Utf8(v) => pb::ScalarValue {
                kind: Some(Kind::StringValue(v.to_string())),
            },
            ScalarValue::Binary(v) => pb::ScalarValue {
                kind: Some(Kind::BytesValue(v.to_vec())),
            },
            ScalarValue::Tuple(v) => {
                let mut values = Vec::with_capacity(v.len());
                for elem in v.iter() {
                    values.push(ScalarValue::to_proto(elem.as_ref()));
                }
                pb::ScalarValue {
                    kind: Some(Kind::ListValue(ListValue { values })),
                }
            }
            ScalarValue::Variant(v) => pb::ScalarValue {
                kind: Some(Kind::VariantValue(Box::new(pb::Scalar::from(v.as_ref())))),
            },
        }
    }
}

impl From<&PValue> for pb::ScalarValue {
    fn from(value: &PValue) -> Self {
        match value {
            PValue::I8(v) => pb::ScalarValue {
                kind: Some(Kind::Int64Value(*v as i64)),
            },
            PValue::I16(v) => pb::ScalarValue {
                kind: Some(Kind::Int64Value(*v as i64)),
            },
            PValue::I32(v) => pb::ScalarValue {
                kind: Some(Kind::Int64Value(*v as i64)),
            },
            PValue::I64(v) => pb::ScalarValue {
                kind: Some(Kind::Int64Value(*v)),
            },
            PValue::U8(v) => pb::ScalarValue {
                kind: Some(Kind::Uint64Value(*v as u64)),
            },
            PValue::U16(v) => pb::ScalarValue {
                kind: Some(Kind::Uint64Value(*v as u64)),
            },
            PValue::U32(v) => pb::ScalarValue {
                kind: Some(Kind::Uint64Value(*v as u64)),
            },
            PValue::U64(v) => pb::ScalarValue {
                kind: Some(Kind::Uint64Value(*v)),
            },
            PValue::F16(v) => pb::ScalarValue {
                kind: Some(Kind::F16Value(v.to_bits() as u64)),
            },
            PValue::F32(v) => pb::ScalarValue {
                kind: Some(Kind::F32Value(*v)),
            },
            PValue::F64(v) => pb::ScalarValue {
                kind: Some(Kind::F64Value(*v)),
            },
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Serialize FROM proto.
////////////////////////////////////////////////////////////////////////////////////////////////////

impl Scalar {
    /// Creates a [`Scalar`] from a [protobuf `ScalarValue`](pb::ScalarValue) representation.
    ///
    /// Note that we need to provide a [`DType`] since protobuf serialization only supports 64-bit
    /// integers, and serializing _into_ protobuf loses that type information.
    ///
    /// # Errors
    ///
    /// Returns an error if type validation fails.
    pub fn from_proto_value(
        value: &pb::ScalarValue,
        dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let scalar_value = ScalarValue::from_proto(value, dtype, session)?;

        Scalar::try_new(dtype.clone(), scalar_value)
    }

    /// Creates a [`Scalar`] from its [protobuf](pb::Scalar) representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the protobuf is missing required fields or if type validation fails.
    pub fn from_proto(value: &pb::Scalar, session: &VortexSession) -> VortexResult<Self> {
        let dtype = DType::from_proto(
            value
                .dtype
                .as_ref()
                .ok_or_else(|| vortex_err!(Serde: "Scalar missing dtype"))?,
            session,
        )?;

        let pb_scalar_value: &pb::ScalarValue = value
            .value
            .as_ref()
            .ok_or_else(|| vortex_err!(Serde: "Scalar missing value"))?;

        let value: Option<ScalarValue> = ScalarValue::from_proto(pb_scalar_value, &dtype, session)?;

        Scalar::try_new(dtype, value)
    }
}

impl ScalarValue {
    /// Deserialize a [`ScalarValue`] from protobuf bytes.
    ///
    /// Note that we need to provide a [`DType`] since protobuf serialization only supports 64-bit
    /// integers, and serializing _into_ protobuf loses that type information.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding or type validation fails.
    pub fn from_proto_bytes(
        bytes: &[u8],
        dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Option<Self>> {
        let proto = pb::ScalarValue::decode(bytes)?;
        Self::from_proto(&proto, dtype, session)
    }

    /// Creates a [`ScalarValue`] from its [protobuf](pb::ScalarValue) representation.
    ///
    /// Note that we need to provide a [`DType`] since protobuf serialization only supports 64-bit
    /// integers, and serializing _into_ protobuf loses that type information.
    ///
    /// # Errors
    ///
    /// Returns an error if the protobuf value cannot be converted to the given [`DType`].
    pub fn from_proto(
        value: &pb::ScalarValue,
        dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Option<Self>> {
        let kind = value
            .kind
            .as_ref()
            .ok_or_else(|| vortex_err!(Serde: "Scalar value missing kind"))?;

        // `DType::Extension` store their serialized values using the storage `DType`.
        let dtype = match dtype {
            DType::Extension(ext) => ext.storage_dtype(),
            _ => dtype,
        };

        Ok(match kind {
            Kind::NullValue(_) => None,
            Kind::BoolValue(v) => Some(bool_from_proto(*v, dtype)?),
            Kind::Int64Value(v) => Some(int64_from_proto(*v, dtype)?),
            Kind::Uint64Value(v) => Some(uint64_from_proto(*v, dtype)?),
            Kind::F16Value(v) => Some(f16_from_proto(*v, dtype)?),
            Kind::F32Value(v) => Some(f32_from_proto(*v, dtype)?),
            Kind::F64Value(v) => Some(f64_from_proto(*v, dtype)?),
            Kind::StringValue(s) => Some(string_from_proto(s, dtype)?),
            Kind::BytesValue(b) => Some(bytes_from_proto(b, dtype)?),
            Kind::ListValue(v) => Some(list_from_proto(v, dtype, session)?),
            Kind::VariantValue(v) => match dtype {
                DType::Variant(_) => Some(ScalarValue::Variant(Box::new(Scalar::from_proto(
                    v, session,
                )?))),
                _ => vortex_bail!(Serde: "expected non-Variant scalar proto for dtype {dtype}"),
            },
        })
    }
}

/// Deserialize a [`ScalarValue::Bool`] from a protobuf `BoolValue`.
fn bool_from_proto(v: bool, dtype: &DType) -> VortexResult<ScalarValue> {
    vortex_ensure!(
        dtype.is_boolean(),
        Serde: "expected Bool dtype for BoolValue, got {dtype}"
    );

    Ok(ScalarValue::Bool(v))
}

/// Deserialize a [`ScalarValue::Primitive`] from a protobuf `Int64Value`.
///
/// Protobuf consolidates all signed integers into `i64`, so we narrow back to the original
/// type using the provided [`DType`].
fn int64_from_proto(v: i64, dtype: &DType) -> VortexResult<ScalarValue> {
    vortex_ensure!(
        dtype.is_primitive(),
        Serde: "expected Primitive dtype for Int64Value, got {dtype}"
    );

    let pvalue = match dtype.as_ptype() {
        PType::I8 => v.to_i8().map(PValue::I8),
        PType::I16 => v.to_i16().map(PValue::I16),
        PType::I32 => v.to_i32().map(PValue::I32),
        PType::I64 => Some(PValue::I64(v)),
        // It was previously possible for unsigned types to get their stats serialised as signed,
        // so we allow casting back to unsigned for backwards compatibility.
        PType::U8 => v.to_u8().map(PValue::U8),
        PType::U16 => v.to_u16().map(PValue::U16),
        PType::U32 => v.to_u32().map(PValue::U32),
        PType::U64 => v.to_u64().map(PValue::U64),
        ftype @ (PType::F16 | PType::F32 | PType::F64) => vortex_bail!(
            Serde: "expected signed integer ptype for serialized Int64Value, got float {ftype}"
        ),
    }
    .ok_or_else(|| vortex_err!(Serde: "Int64 value {v} out of range for dtype {dtype}"))?;

    Ok(ScalarValue::Primitive(pvalue))
}

/// Deserialize a [`ScalarValue::Primitive`] from a protobuf `Uint64Value`.
///
/// Protobuf consolidates all unsigned integers into `u64`, so we narrow back to the original
/// type using the provided [`DType`]. Also handles the backwards-compatible case where `f16`
/// values were serialized as `u64` (via `f16::to_bits() as u64`).
fn uint64_from_proto(v: u64, dtype: &DType) -> VortexResult<ScalarValue> {
    vortex_ensure!(
        dtype.is_primitive(),
        Serde: "expected Primitive dtype for Uint64Value, got {dtype}"
    );

    let pvalue = match dtype.as_ptype() {
        PType::U8 => v.to_u8().map(PValue::U8),
        PType::U16 => v.to_u16().map(PValue::U16),
        PType::U32 => v.to_u32().map(PValue::U32),
        PType::U64 => Some(PValue::U64(v)),
        // It was previously possible for signed types to get their stats serialised as unsigned,
        // so we allow casting back to signed for backwards compatibility.
        PType::I8 => v.to_i8().map(PValue::I8),
        PType::I16 => v.to_i16().map(PValue::I16),
        PType::I32 => v.to_i32().map(PValue::I32),
        PType::I64 => v.to_i64().map(PValue::I64),
        // f16 values used to be serialized as u64, so we need to be able to read an f16 from a u64.
        PType::F16 => v.to_u16().map(f16::from_bits).map(PValue::F16),
        ftype @ (PType::F32 | PType::F64) => vortex_bail!(
            Serde: "expected unsigned integer ptype for serialized Uint64Value, got {ftype}"
        ),
    }
    .ok_or_else(|| vortex_err!(Serde: "Uint64 value {v} out of range for dtype {dtype}"))?;

    Ok(ScalarValue::Primitive(pvalue))
}

/// Deserialize a [`ScalarValue::Primitive`] from a protobuf `F16Value`.
fn f16_from_proto(v: u64, dtype: &DType) -> VortexResult<ScalarValue> {
    vortex_ensure!(
        matches!(dtype, DType::Primitive(PType::F16, _)),
        Serde: "expected F16 dtype for F16Value, got {dtype}"
    );

    let bits = u16::try_from(v)
        .map_err(|_| vortex_err!(Serde: "f16 bitwise representation has more than 16 bits: {v}"))?;

    Ok(ScalarValue::Primitive(PValue::F16(f16::from_bits(bits))))
}

/// Deserialize a [`ScalarValue::Primitive`] from a protobuf `F32Value`.
fn f32_from_proto(v: f32, dtype: &DType) -> VortexResult<ScalarValue> {
    vortex_ensure!(
        matches!(dtype, DType::Primitive(PType::F32, _)),
        Serde: "expected F32 dtype for F32Value, got {dtype}"
    );

    Ok(ScalarValue::Primitive(PValue::F32(v)))
}

/// Deserialize a [`ScalarValue::Primitive`] from a protobuf `F64Value`.
fn f64_from_proto(v: f64, dtype: &DType) -> VortexResult<ScalarValue> {
    vortex_ensure!(
        matches!(dtype, DType::Primitive(PType::F64, _)),
        Serde: "expected F64 dtype for F64Value, got {dtype}"
    );

    Ok(ScalarValue::Primitive(PValue::F64(v)))
}

/// Deserialize a [`ScalarValue::Utf8`] or [`ScalarValue::Binary`] from a protobuf
/// `StringValue`.
fn string_from_proto(s: &str, dtype: &DType) -> VortexResult<ScalarValue> {
    match dtype {
        DType::Utf8(_) => Ok(ScalarValue::Utf8(BufferString::from(s))),
        DType::Binary(_) => Ok(ScalarValue::Binary(ByteBuffer::copy_from(s.as_bytes()))),
        _ => vortex_bail!(
            Serde: "expected Utf8 or Binary dtype for StringValue, got {dtype}"
        ),
    }
}

/// Deserialize a [`ScalarValue`] from a protobuf bytes and a `DType`.
///
/// Handles [`Utf8`](ScalarValue::Utf8), [`Binary`](ScalarValue::Binary), and
/// [`Decimal`](ScalarValue::Decimal) dtypes.
fn bytes_from_proto(bytes: &[u8], dtype: &DType) -> VortexResult<ScalarValue> {
    match dtype {
        DType::Utf8(_) => Ok(ScalarValue::Utf8(BufferString::try_from(bytes)?)),
        DType::Binary(_) => Ok(ScalarValue::Binary(ByteBuffer::copy_from(bytes))),
        // TODO(connor): This is incorrect, we need to verify this matches the inner decimal_dtype.
        DType::Decimal(..) => Ok(ScalarValue::Decimal(match bytes.len() {
            1 => DecimalValue::I8(bytes[0] as i8),
            2 => DecimalValue::I16(i16::from_le_bytes(
                bytes
                    .try_into()
                    .ok()
                    .vortex_expect("Buffer has invalid number of bytes"),
            )),
            4 => DecimalValue::I32(i32::from_le_bytes(
                bytes
                    .try_into()
                    .ok()
                    .vortex_expect("Buffer has invalid number of bytes"),
            )),
            8 => DecimalValue::I64(i64::from_le_bytes(
                bytes
                    .try_into()
                    .ok()
                    .vortex_expect("Buffer has invalid number of bytes"),
            )),
            16 => DecimalValue::I128(i128::from_le_bytes(
                bytes
                    .try_into()
                    .ok()
                    .vortex_expect("Buffer has invalid number of bytes"),
            )),
            32 => DecimalValue::I256(i256::from_le_bytes(
                bytes
                    .try_into()
                    .ok()
                    .vortex_expect("Buffer has invalid number of bytes"),
            )),
            l => vortex_bail!(Serde: "invalid decimal byte length: {l}"),
        })),
        _ => vortex_bail!(
            Serde: "expected Utf8, Binary, or Decimal dtype for BytesValue, got {dtype}"
        ),
    }
}

/// Deserialize a [`ScalarValue::Tuple`] from a protobuf `ListValue`.
fn list_from_proto(
    v: &ListValue,
    dtype: &DType,
    session: &VortexSession,
) -> VortexResult<ScalarValue> {
    let element_dtype = dtype
        .as_list_element_opt()
        .ok_or_else(|| vortex_err!(Serde: "expected List dtype for ListValue, got {dtype}"))?;

    let mut values = Vec::with_capacity(v.values.len());
    for elem in v.values.iter() {
        values.push(ScalarValue::from_proto(
            elem,
            element_dtype.as_ref(),
            session,
        )?);
    }

    Ok(ScalarValue::Tuple(values))
}

#[cfg(test)]
mod tests {
    use std::f32;
    use std::f64;
    use std::sync::Arc;

    use vortex_buffer::BufferString;
    use vortex_error::vortex_panic;
    use vortex_proto::scalar as pb;
    use vortex_session::VortexSession;

    use super::*;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::half::f16;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

    fn session() -> VortexSession {
        VortexSession::empty()
    }

    fn round_trip(scalar: Scalar) {
        assert_eq!(
            scalar,
            Scalar::from_proto(&pb::Scalar::from(&scalar), &session()).unwrap(),
        );
    }

    #[test]
    fn test_null() {
        round_trip(Scalar::null(DType::Null));
    }

    #[test]
    fn test_bool() {
        round_trip(Scalar::new(
            DType::Bool(Nullability::Nullable),
            Some(ScalarValue::Bool(true)),
        ));
    }

    #[test]
    fn test_primitive() {
        round_trip(Scalar::new(
            DType::Primitive(PType::I32, Nullability::Nullable),
            Some(ScalarValue::Primitive(42i32.into())),
        ));
    }

    #[test]
    fn test_buffer() {
        round_trip(Scalar::new(
            DType::Binary(Nullability::Nullable),
            Some(ScalarValue::Binary(vec![1, 2, 3].into())),
        ));
    }

    #[test]
    fn test_buffer_string() {
        round_trip(Scalar::new(
            DType::Utf8(Nullability::Nullable),
            Some(ScalarValue::Utf8(BufferString::from("hello".to_string()))),
        ));
    }

    #[test]
    fn test_list() {
        round_trip(Scalar::new(
            DType::List(
                Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
                Nullability::Nullable,
            ),
            Some(ScalarValue::Tuple(vec![
                Some(ScalarValue::Primitive(42i32.into())),
                Some(ScalarValue::Primitive(43i32.into())),
            ])),
        ));
    }

    #[test]
    fn test_f16() {
        round_trip(Scalar::primitive(
            f16::from_f32(0.42),
            Nullability::Nullable,
        ));
    }

    #[test]
    fn test_i8() {
        round_trip(Scalar::new(
            DType::Primitive(PType::I8, Nullability::Nullable),
            Some(ScalarValue::Primitive(i8::MIN.into())),
        ));

        round_trip(Scalar::new(
            DType::Primitive(PType::I8, Nullability::Nullable),
            Some(ScalarValue::Primitive(0i8.into())),
        ));

        round_trip(Scalar::new(
            DType::Primitive(PType::I8, Nullability::Nullable),
            Some(ScalarValue::Primitive(i8::MAX.into())),
        ));
    }

    #[test]
    fn test_decimal_i32_roundtrip() {
        // A typical decimal with moderate precision and scale.
        round_trip(Scalar::decimal(
            DecimalValue::I32(123_456),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        ));
    }

    #[test]
    fn test_decimal_i128_roundtrip() {
        // A large decimal value that requires i128 storage.
        round_trip(Scalar::decimal(
            DecimalValue::I128(99_999_999_999_999_999_999),
            DecimalDType::new(38, 6),
            Nullability::Nullable,
        ));
    }

    #[test]
    fn test_decimal_null_roundtrip() {
        round_trip(Scalar::null(DType::Decimal(
            DecimalDType::new(10, 2),
            Nullability::Nullable,
        )));
    }

    #[test]
    fn test_scalar_value_serde_roundtrip_binary() {
        round_trip(Scalar::binary(
            ByteBuffer::copy_from(b"hello"),
            Nullability::NonNullable,
        ));
    }

    #[test]
    fn test_scalar_value_serde_roundtrip_utf8() {
        round_trip(Scalar::utf8("hello", Nullability::NonNullable));
    }

    #[test]
    fn test_variant_scalar_roundtrip() {
        let nums = Scalar::list(
            Arc::new(DType::Variant(Nullability::NonNullable)),
            vec![
                Scalar::variant(Scalar::primitive(-7_i16, Nullability::NonNullable)),
                Scalar::variant(Scalar::primitive(42_u32, Nullability::NonNullable)),
                Scalar::variant(Scalar::decimal(
                    DecimalValue::I128(123_456_789),
                    DecimalDType::new(18, 0),
                    Nullability::NonNullable,
                )),
            ],
            Nullability::NonNullable,
        );

        let nested = Scalar::list(
            Arc::new(DType::Variant(Nullability::NonNullable)),
            vec![
                Scalar::variant(Scalar::from(true)),
                Scalar::variant(nums),
                Scalar::variant(Scalar::binary(
                    ByteBuffer::copy_from(b"abc"),
                    Nullability::NonNullable,
                )),
                Scalar::variant(Scalar::null(DType::Null)),
            ],
            Nullability::NonNullable,
        );

        round_trip(Scalar::variant(nested));
    }

    #[test]
    fn test_variant_scalar_proto_preserves_scalar_null_vs_variant_null() {
        let scalar_null = Scalar::null(DType::Variant(Nullability::Nullable));
        let variant_null = Scalar::variant(Scalar::null(DType::Null));

        let scalar_null_pb = pb::Scalar::from(&scalar_null);
        let variant_null_pb = pb::Scalar::from(&variant_null);

        assert_ne!(scalar_null_pb, variant_null_pb);
        assert_eq!(
            Scalar::from_proto(&scalar_null_pb, &session()).unwrap(),
            scalar_null,
        );
        assert_eq!(
            Scalar::from_proto(&variant_null_pb, &session()).unwrap(),
            variant_null,
        );
    }

    #[test]
    fn test_backcompat_f16_serialized_as_u64() {
        // Backwards compatibility test for the legacy f16 serialization format.
        //
        // Previously, f16 ScalarValues were serialized as `Uint64Value(v.to_bits() as u64)` because
        // the proto schema only had 64-bit integer types, and f16's underlying representation is
        // u16 which got widened to u64.
        //
        // The current implementation uses a dedicated `F16Value` proto field, but we must still be
        // able to deserialize the old format. This test verifies that:
        //
        // 1. A `Uint64Value` containing f16 bits can be read as a U64 primitive (the raw bits).
        // 2. When wrapped in a Scalar with F16 dtype, the value is correctly interpreted as f16.
        //
        // This ensures data written with the old serialization format remains readable.

        // Simulate the old serialization: f16(0.42) stored as Uint64Value with its bit pattern.
        let f16_value = f16::from_f32(0.42);
        let f16_bits_as_u64 = f16_value.to_bits() as u64; // 14008

        let pb_scalar_value = pb::ScalarValue {
            kind: Some(Kind::Uint64Value(f16_bits_as_u64)),
        };

        // Step 1: Verify the normal U64 scalar.
        let scalar_value = ScalarValue::from_proto(
            &pb_scalar_value,
            &DType::Primitive(PType::U64, Nullability::NonNullable),
            &session(),
        )
        .unwrap();
        assert_eq!(
            scalar_value.as_ref().map(|v| v.as_primitive()),
            Some(&PValue::U64(14008u64)),
        );

        // Step 2: Verify that when we use F16 dtype, the Uint64Value is correctly interpreted.
        let scalar_value_f16 = ScalarValue::from_proto(
            &pb_scalar_value,
            &DType::Primitive(PType::F16, Nullability::Nullable),
            &session(),
        )
        .unwrap();

        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::Nullable),
            scalar_value_f16,
        );

        assert_eq!(
            scalar.as_primitive().pvalue().unwrap(),
            PValue::F16(f16::from_f32(0.42)),
            "Uint64Value should be correctly interpreted as f16 when dtype is F16"
        );
    }

    #[test]
    fn test_scalar_value_direct_roundtrip_f16() {
        // Test that ScalarValue with f16 roundtrips correctly without going through Scalar.
        let f16_values = vec![
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(-1.0),
            f16::from_f32(0.42),
            f16::from_f32(5.722046e-6),
            f16::from_f32(f32::consts::PI),
            f16::INFINITY,
            f16::NEG_INFINITY,
            f16::NAN,
        ];

        for f16_val in f16_values {
            let scalar_value = ScalarValue::Primitive(PValue::F16(f16_val));
            let pb_value = ScalarValue::to_proto(Some(&scalar_value));
            let read_back = ScalarValue::from_proto(
                &pb_value,
                &DType::Primitive(PType::F16, Nullability::NonNullable),
                &session(),
            )
            .unwrap();

            match (&scalar_value, read_back.as_ref()) {
                (
                    ScalarValue::Primitive(PValue::F16(original)),
                    Some(ScalarValue::Primitive(PValue::F16(roundtripped))),
                ) => {
                    if original.is_nan() && roundtripped.is_nan() {
                        // NaN values are equal for our purposes.
                        continue;
                    }
                    assert_eq!(
                        original, roundtripped,
                        "F16 value {original:?} did not roundtrip correctly"
                    );
                }
                _ => {
                    vortex_panic!(
                        "Expected f16 primitive values, got {scalar_value:?} and {read_back:?}"
                    )
                }
            }
        }
    }

    #[test]
    fn test_scalar_value_direct_roundtrip_preserves_values() {
        // Test that ScalarValue roundtripping preserves values (but not necessarily exact types).
        // Note: Proto encoding consolidates integer types (u8/u16/u32 → u64, i8/i16/i32 → i64).

        // Test cases that should roundtrip exactly.
        let exact_roundtrip_cases: Vec<(&str, Option<ScalarValue>, DType)> = vec![
            ("null", None, DType::Null),
            (
                "bool_true",
                Some(ScalarValue::Bool(true)),
                DType::Bool(Nullability::Nullable),
            ),
            (
                "bool_false",
                Some(ScalarValue::Bool(false)),
                DType::Bool(Nullability::Nullable),
            ),
            (
                "u64",
                Some(ScalarValue::Primitive(PValue::U64(18446744073709551615))),
                DType::Primitive(PType::U64, Nullability::Nullable),
            ),
            (
                "i64",
                Some(ScalarValue::Primitive(PValue::I64(-9223372036854775808))),
                DType::Primitive(PType::I64, Nullability::Nullable),
            ),
            (
                "f32",
                Some(ScalarValue::Primitive(PValue::F32(f32::consts::E))),
                DType::Primitive(PType::F32, Nullability::Nullable),
            ),
            (
                "f64",
                Some(ScalarValue::Primitive(PValue::F64(f64::consts::PI))),
                DType::Primitive(PType::F64, Nullability::Nullable),
            ),
            (
                "string",
                Some(ScalarValue::Utf8(BufferString::from("test"))),
                DType::Utf8(Nullability::Nullable),
            ),
            (
                "bytes",
                Some(ScalarValue::Binary(vec![1, 2, 3, 4, 5].into())),
                DType::Binary(Nullability::Nullable),
            ),
        ];

        for (name, value, dtype) in exact_roundtrip_cases {
            let pb_value = ScalarValue::to_proto(value.as_ref());
            let read_back = ScalarValue::from_proto(&pb_value, &dtype, &session()).unwrap();

            let original_debug = format!("{value:?}");
            let roundtrip_debug = format!("{read_back:?}");
            assert_eq!(
                original_debug, roundtrip_debug,
                "ScalarValue {name} did not roundtrip exactly"
            );
        }

        // Test cases where type changes but value is preserved.
        // Unsigned integers consolidate to U64.
        let unsigned_cases = vec![
            (
                "u8",
                ScalarValue::Primitive(PValue::U8(255)),
                DType::Primitive(PType::U8, Nullability::Nullable),
                255u64,
            ),
            (
                "u16",
                ScalarValue::Primitive(PValue::U16(65535)),
                DType::Primitive(PType::U16, Nullability::Nullable),
                65535u64,
            ),
            (
                "u32",
                ScalarValue::Primitive(PValue::U32(4294967295)),
                DType::Primitive(PType::U32, Nullability::Nullable),
                4294967295u64,
            ),
        ];

        for (name, value, dtype, expected) in unsigned_cases {
            let pb_value = ScalarValue::to_proto(Some(&value));
            let read_back = ScalarValue::from_proto(&pb_value, &dtype, &session()).unwrap();

            match read_back.as_ref() {
                Some(ScalarValue::Primitive(pv)) => {
                    let v = match pv {
                        PValue::U8(v) => *v as u64,
                        PValue::U16(v) => *v as u64,
                        PValue::U32(v) => *v as u64,
                        PValue::U64(v) => *v,
                        _ => vortex_panic!("Unexpected primitive type for {name}: {pv:?}"),
                    };
                    assert_eq!(
                        v, expected,
                        "ScalarValue {name} value not preserved: expected {expected}, got {v}"
                    );
                }
                _ => vortex_panic!("Unexpected type after roundtrip for {name}: {read_back:?}"),
            }
        }

        // Signed integers consolidate to I64.
        let signed_cases = vec![
            (
                "i8",
                ScalarValue::Primitive(PValue::I8(-128)),
                DType::Primitive(PType::I8, Nullability::Nullable),
                -128i64,
            ),
            (
                "i16",
                ScalarValue::Primitive(PValue::I16(-32768)),
                DType::Primitive(PType::I16, Nullability::Nullable),
                -32768i64,
            ),
            (
                "i32",
                ScalarValue::Primitive(PValue::I32(-2147483648)),
                DType::Primitive(PType::I32, Nullability::Nullable),
                -2147483648i64,
            ),
        ];

        for (name, value, dtype, expected) in signed_cases {
            let pb_value = ScalarValue::to_proto(Some(&value));
            let read_back = ScalarValue::from_proto(&pb_value, &dtype, &session()).unwrap();

            match read_back.as_ref() {
                Some(ScalarValue::Primitive(pv)) => {
                    let v = match pv {
                        PValue::I8(v) => *v as i64,
                        PValue::I16(v) => *v as i64,
                        PValue::I32(v) => *v as i64,
                        PValue::I64(v) => *v,
                        _ => vortex_panic!("Unexpected primitive type for {name}: {pv:?}"),
                    };
                    assert_eq!(
                        v, expected,
                        "ScalarValue {name} value not preserved: expected {expected}, got {v}"
                    );
                }
                _ => vortex_panic!("Unexpected type after roundtrip for {name}: {read_back:?}"),
            }
        }
    }

    // Backwards compatibility: signed integer stats could previously be serialized as unsigned.
    // Therefore, we allow casting between signed and unsigned integers of the same bit width.
    #[test]
    fn test_backcompat_signed_integer_deserialized_as_unsigned() {
        let v = ScalarValue::Primitive(PValue::I64(0));
        assert_eq!(
            Scalar::from_proto_value(
                &pb::ScalarValue::from(&v),
                &DType::Primitive(PType::U64, Nullability::Nullable),
                &session()
            )
            .unwrap(),
            Scalar::primitive(0u64, Nullability::Nullable)
        );
    }

    // Backwards compatibility: unsigned integer stats could previously be serialized as signed.
    // Therefore, we allow casting between signed and unsigned integers of the same bit width.
    #[test]
    fn test_backcompat_unsigned_integer_deserialized_as_signed() {
        let v = ScalarValue::Primitive(PValue::U64(0));
        assert_eq!(
            Scalar::from_proto_value(
                &pb::ScalarValue::from(&v),
                &DType::Primitive(PType::I64, Nullability::Nullable),
                &session()
            )
            .unwrap(),
            Scalar::primitive(0i64, Nullability::Nullable)
        );
    }
}
