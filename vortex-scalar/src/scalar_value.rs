// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::sync::Arc;

use bytes::BufMut;
use itertools::Itertools;
use prost::Message;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_dtype::NativeDType;
use vortex_dtype::i256;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::scalar as pb;

use crate::Scalar;
use crate::decimal::DecimalValue;
use crate::pvalue::PValue;

/// Represents the internal data of a scalar value. Must be interpreted by wrapping up with a
/// [`vortex_dtype::DType`] to make a [`super::Scalar`].
///
/// Note that these values can be deserialized from JSON or other formats. So a [`PValue`] may not
/// have the correct width for what the [`vortex_dtype::DType`] expects. Primitive values should therefore be
/// read using [`super::PrimitiveScalar`] which will handle the conversion.
#[derive(Debug, Clone)]
pub struct ScalarValue(pub(crate) InnerScalarValue);

/// It is common to represent a nullable type `T` as an `Option<T>`, so we implement a blanket
/// implementation for all `Option<T>` to simply be a nullable `T`.
impl<T> From<Option<T>> for ScalarValue
where
    T: NativeDType,
    ScalarValue: From<T>,
{
    fn from(value: Option<T>) -> Self {
        value
            .map(ScalarValue::from)
            .unwrap_or_else(|| ScalarValue(InnerScalarValue::Null))
    }
}

impl<T> From<Vec<T>> for ScalarValue
where
    T: NativeDType,
    Scalar: From<T>,
{
    /// Converts a vector into a `ScalarValue` (specifically a `ListScalar`).
    fn from(value: Vec<T>) -> Self {
        ScalarValue(InnerScalarValue::List(
            value
                .into_iter()
                .map(|x| {
                    let scalar: Scalar = T::into(x);
                    scalar.into_value()
                })
                .collect::<Arc<[ScalarValue]>>(),
        ))
    }
}

#[derive(Debug, Clone)]
pub(crate) enum InnerScalarValue {
    Null,
    Bool(bool),
    Primitive(PValue),
    Decimal(DecimalValue),
    Buffer(Arc<ByteBuffer>),
    BufferString(Arc<BufferString>),
    List(Arc<[ScalarValue]>),
}

impl ScalarValue {
    /// Serializes the scalar value to Protocol Buffers format.
    pub fn to_protobytes<B: BufMut + Default>(&self) -> B {
        let pb_scalar = pb::ScalarValue::from(self);

        let mut buf = B::default();
        pb_scalar
            .encode(&mut buf)
            .vortex_expect("protobuf encoding should succeed");
        buf
    }

    /// Deserializes a scalar value from Protocol Buffers format.
    pub fn from_protobytes(buf: &[u8]) -> VortexResult<Self> {
        ScalarValue::try_from(&pb::ScalarValue::decode(buf)?)
    }
}

fn to_hex(slice: &[u8]) -> String {
    slice
        .iter()
        .format_with("", |f, b| b(&format_args!("{f:02x}")))
        .to_string()
}

impl Display for ScalarValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for InnerScalarValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(b) => write!(f, "{b}"),
            Self::Primitive(pvalue) => write!(f, "{pvalue}"),
            Self::Decimal(value) => write!(f, "{value}"),
            Self::Buffer(buf) => {
                if buf.len() > 10 {
                    write!(
                        f,
                        "{}..{}",
                        to_hex(&buf[0..5]),
                        to_hex(&buf[buf.len() - 5..buf.len()]),
                    )
                } else {
                    write!(f, "{}", to_hex(buf))
                }
            }
            Self::BufferString(bufstr) => {
                let bufstr = bufstr.as_str();
                let str_len = bufstr.chars().count();

                if str_len > 10 {
                    let prefix = String::from_iter(bufstr.chars().take(5));
                    let suffix = String::from_iter(bufstr.chars().skip(str_len - 5));

                    write!(f, "\"{prefix}..{suffix}\"")
                } else {
                    write!(f, "\"{bufstr}\"")
                }
            }
            Self::List(elems) => {
                write!(f, "[{}]", elems.iter().format(","))
            }
            Self::Null => write!(f, "null"),
        }
    }
}

impl ScalarValue {
    /// Creates a null scalar value.
    pub const fn null() -> Self {
        ScalarValue(InnerScalarValue::Null)
    }

    /// Returns true if this is a null value.
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns scalar as a null value
    #[inline]
    pub(crate) fn as_null(&self) -> VortexResult<()> {
        self.0.as_null()
    }

    /// Returns scalar as a boolean value
    #[inline]
    pub(crate) fn as_bool(&self) -> VortexResult<Option<bool>> {
        self.0.as_bool()
    }

    /// Return scalar as a primitive value. PValues don't match dtypes but will be castable to the scalars dtype
    #[inline]
    pub(crate) fn as_pvalue(&self) -> VortexResult<Option<PValue>> {
        self.0.as_pvalue()
    }

    /// Returns scalar as a decimal value
    #[inline]
    pub(crate) fn as_decimal(&self) -> VortexResult<Option<DecimalValue>> {
        self.0.as_decimal()
    }

    /// Returns scalar as a binary buffer
    #[inline]
    pub(crate) fn as_buffer(&self) -> VortexResult<Option<Arc<ByteBuffer>>> {
        self.0.as_buffer()
    }

    /// Returns scalar as a string buffer
    #[inline]
    pub(crate) fn as_buffer_string(&self) -> VortexResult<Option<Arc<BufferString>>> {
        self.0.as_buffer_string()
    }

    /// Returns scalar as a list value
    #[inline]
    pub(crate) fn as_list(&self) -> VortexResult<Option<&Arc<[ScalarValue]>>> {
        self.0.as_list()
    }
}

impl InnerScalarValue {
    #[inline]
    pub(crate) fn is_null(&self) -> bool {
        matches!(self, InnerScalarValue::Null)
    }

    #[inline]
    pub(crate) fn as_null(&self) -> VortexResult<()> {
        if matches!(self, InnerScalarValue::Null) {
            Ok(())
        } else {
            Err(vortex_err!("Expected a Null scalar, found {self}"))
        }
    }

    #[inline]
    pub(crate) fn as_bool(&self) -> VortexResult<Option<bool>> {
        match self {
            InnerScalarValue::Null => Ok(None),
            InnerScalarValue::Bool(b) => Ok(Some(*b)),
            other => Err(vortex_err!("Expected a bool scalar, found {other}",)),
        }
    }

    /// FIXME(ngates): PValues are such a footgun... we should probably remove this.
    ///  But the other accessors can sometimes be useful? e.g. as_buffer. But maybe we just force
    ///  the user to switch over Utf8 and Binary and use the correct Scalar wrapper?
    #[inline]
    pub(crate) fn as_pvalue(&self) -> VortexResult<Option<PValue>> {
        match self {
            InnerScalarValue::Null => Ok(None),
            InnerScalarValue::Primitive(pvalue) => Ok(Some(*pvalue)),
            other => Err(vortex_err!("Expected a primitive scalar, found {other}")),
        }
    }

    #[inline]
    pub(crate) fn as_decimal(&self) -> VortexResult<Option<DecimalValue>> {
        match self {
            InnerScalarValue::Null => Ok(None),
            InnerScalarValue::Decimal(v) => Ok(Some(*v)),
            InnerScalarValue::Buffer(b) => Ok(Some(match b.len() {
                1 => DecimalValue::I8(b[0] as i8),
                2 => DecimalValue::I16(i16::from_le_bytes(b.as_slice().try_into()?)),
                4 => DecimalValue::I32(i32::from_le_bytes(b.as_slice().try_into()?)),
                8 => DecimalValue::I64(i64::from_le_bytes(b.as_slice().try_into()?)),
                16 => DecimalValue::I128(i128::from_le_bytes(b.as_slice().try_into()?)),
                32 => DecimalValue::I256(i256::from_le_bytes(b.as_slice().try_into()?)),
                l => vortex_bail!("Buffer is not a decimal value length {l}"),
            })),
            _ => vortex_bail!("Expected a decimal scalar, found {:?}", self),
        }
    }

    #[inline]
    pub(crate) fn as_buffer(&self) -> VortexResult<Option<Arc<ByteBuffer>>> {
        match &self {
            InnerScalarValue::Null => Ok(None),
            InnerScalarValue::Buffer(b) => Ok(Some(b.clone())),
            InnerScalarValue::BufferString(b) => {
                Ok(Some(Arc::new(b.as_ref().clone().into_inner())))
            }
            _ => Err(vortex_err!("Expected a binary scalar, found {:?}", self)),
        }
    }

    #[inline]
    pub(crate) fn as_buffer_string(&self) -> VortexResult<Option<Arc<BufferString>>> {
        match &self {
            InnerScalarValue::Null => Ok(None),
            InnerScalarValue::Buffer(b) => {
                Ok(Some(Arc::new(BufferString::try_from(b.as_ref().clone())?)))
            }
            InnerScalarValue::BufferString(b) => Ok(Some(b.clone())),
            _ => Err(vortex_err!("Expected a string scalar, found {:?}", self)),
        }
    }

    #[inline]
    pub(crate) fn as_list(&self) -> VortexResult<Option<&Arc<[ScalarValue]>>> {
        match &self {
            InnerScalarValue::Null => Ok(None),
            InnerScalarValue::List(l) => Ok(Some(l)),
            _ => Err(vortex_err!("Expected a list scalar, found {:?}", self)),
        }
    }
}
