// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Protobuf serialization and deserialization for scalars.

use num_traits::ToBytes;
use num_traits::ToPrimitive;
use prost::Message;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_dtype::PType;
use vortex_dtype::half::f16;
use vortex_dtype::i256;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_proto::scalar as pb;
use vortex_proto::scalar::ListValue;
use vortex_proto::scalar::scalar_value::Kind;
use vortex_session::VortexSession;

use crate::DecimalValue;
use crate::PValue;
use crate::Scalar;
use crate::ScalarValue;

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
            value: Some(ScalarValue::to_proto(value.value())),
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
    /// `vortex_scalar::ScalarValue` directly.
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
            ScalarValue::List(v) => {
                let mut values = Vec::with_capacity(v.len());
                for elem in v.iter() {
                    values.push(ScalarValue::to_proto(elem.as_ref()));
                }
                pb::ScalarValue {
                    kind: Some(Kind::ListValue(ListValue { values })),
                }
            }
            ScalarValue::Extension(ext_scalar_value_ref) => {
                Self::from(ext_scalar_value_ref.storage_value())
            }
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
                .ok_or_else(|| vortex_err!(InvalidSerde: "Scalar missing dtype"))?,
            session,
        )?;

        let pb_scalar_value: &pb::ScalarValue = value
            .value
            .as_ref()
            .ok_or_else(|| vortex_err!(InvalidSerde: "Scalar missing value"))?;

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
            .ok_or_else(|| vortex_err!(InvalidSerde: "Scalar value missing kind"))?;

        // If we want to deserialize an extension type, we need the extension registry + special
        // logic.
        if let Some(ext_dtype) = dtype.as_extension_opt() {
            let storage_value = Self::from_proto(value, ext_dtype.storage_dtype(), session)?;

            return storage_value
                .map(|sv| ScalarValue::extension_value(ext_dtype, sv, session))
                .transpose();
        };

        Ok(Some(match kind {
            Kind::NullValue(_) => return Ok(None),
            Kind::BoolValue(v) => bool_from_proto(*v, dtype)?,
            Kind::Int64Value(v) => int64_from_proto(*v, dtype)?,
            Kind::Uint64Value(v) => uint64_from_proto(*v, dtype)?,
            Kind::F16Value(v) => f16_from_proto(*v, dtype)?,
            Kind::F32Value(v) => f32_from_proto(*v, dtype)?,
            Kind::F64Value(v) => f64_from_proto(*v, dtype)?,
            Kind::StringValue(s) => string_from_proto(s, dtype)?,
            Kind::BytesValue(b) => bytes_from_proto(b, dtype)?,
            Kind::ListValue(v) => list_from_proto(v, dtype, session)?,
        }))
    }
}

/// Deserialize a [`ScalarValue::Bool`] from a protobuf `BoolValue`.
fn bool_from_proto(v: bool, dtype: &DType) -> VortexResult<ScalarValue> {
    vortex_ensure!(
        dtype.is_boolean(),
        InvalidSerde: "expected Bool dtype for BoolValue, got {dtype}"
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
        InvalidSerde: "expected Primitive dtype for Int64Value, got {dtype}"
    );

    let pvalue = match dtype.as_ptype() {
        PType::I8 => v.to_i8().map(PValue::I8),
        PType::I16 => v.to_i16().map(PValue::I16),
        PType::I32 => v.to_i32().map(PValue::I32),
        PType::I64 => Some(PValue::I64(v)),
        ptype => vortex_bail!(
            InvalidSerde: "expected signed integer ptype for Int64Value, got {ptype}"
        ),
    }
    .ok_or_else(|| vortex_err!(InvalidSerde: "Int64 value {v} out of range for dtype {dtype}"))?;

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
        InvalidSerde: "expected Primitive dtype for Uint64Value, got {dtype}"
    );

    let pvalue = match dtype.as_ptype() {
        PType::U8 => v.to_u8().map(PValue::U8),
        PType::U16 => v.to_u16().map(PValue::U16),
        PType::U32 => v.to_u32().map(PValue::U32),
        PType::U64 => Some(PValue::U64(v)),
        // Backwards compatibility: f16 values were previously serialized as u64.
        PType::F16 => v.to_u16().map(f16::from_bits).map(PValue::F16),
        ptype => vortex_bail!(
            InvalidSerde: "expected unsigned integer ptype for Uint64Value, got {ptype}"
        ),
    }
    .ok_or_else(|| vortex_err!(InvalidSerde: "Uint64 value {v} out of range for dtype {dtype}"))?;

    Ok(ScalarValue::Primitive(pvalue))
}

/// Deserialize a [`ScalarValue::Primitive`] from a protobuf `F16Value`.
fn f16_from_proto(v: u64, dtype: &DType) -> VortexResult<ScalarValue> {
    vortex_ensure!(
        matches!(dtype, DType::Primitive(PType::F16, _)),
        InvalidSerde: "expected F16 dtype for F16Value, got {dtype}"
    );

    let bits = u16::try_from(v).map_err(
        |_| vortex_err!(InvalidSerde: "f16 bitwise representation has more than 16 bits: {v}"),
    )?;

    Ok(ScalarValue::Primitive(PValue::F16(f16::from_bits(bits))))
}

/// Deserialize a [`ScalarValue::Primitive`] from a protobuf `F32Value`.
fn f32_from_proto(v: f32, dtype: &DType) -> VortexResult<ScalarValue> {
    vortex_ensure!(
        matches!(dtype, DType::Primitive(PType::F32, _)),
        InvalidSerde: "expected F32 dtype for F32Value, got {dtype}"
    );

    Ok(ScalarValue::Primitive(PValue::F32(v)))
}

/// Deserialize a [`ScalarValue::Primitive`] from a protobuf `F64Value`.
fn f64_from_proto(v: f64, dtype: &DType) -> VortexResult<ScalarValue> {
    vortex_ensure!(
        matches!(dtype, DType::Primitive(PType::F64, _)),
        InvalidSerde: "expected F64 dtype for F64Value, got {dtype}"
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
            InvalidSerde: "expected Utf8 or Binary dtype for StringValue, got {dtype}"
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
        // TODO(connor): This is incorrect, we need to verify this matches the `dtype`.
        DType::Decimal(..) => Ok(ScalarValue::Decimal(match bytes.len() {
            1 => DecimalValue::I8(bytes[0] as i8),
            2 => DecimalValue::I16(i16::from_le_bytes(bytes.try_into()?)),
            4 => DecimalValue::I32(i32::from_le_bytes(bytes.try_into()?)),
            8 => DecimalValue::I64(i64::from_le_bytes(bytes.try_into()?)),
            16 => DecimalValue::I128(i128::from_le_bytes(bytes.try_into()?)),
            32 => DecimalValue::I256(i256::from_le_bytes(bytes.try_into()?)),
            l => vortex_bail!(InvalidSerde: "invalid decimal byte length: {l}"),
        })),
        _ => vortex_bail!(
            InvalidSerde: "expected Utf8, Binary, or Decimal dtype for BytesValue, got {dtype}"
        ),
    }
}

/// Deserialize a [`ScalarValue::List`] from a protobuf `ListValue`.
fn list_from_proto(
    v: &ListValue,
    dtype: &DType,
    session: &VortexSession,
) -> VortexResult<ScalarValue> {
    let element_dtype = dtype.as_list_element_opt().ok_or_else(
        || vortex_err!(InvalidSerde: "expected List dtype for ListValue, got {dtype}"),
    )?;

    let mut values = Vec::with_capacity(v.values.len());
    for elem in v.values.iter() {
        values.push(ScalarValue::from_proto(
            elem,
            element_dtype.as_ref(),
            session,
        )?);
    }

    Ok(ScalarValue::List(values))
}
