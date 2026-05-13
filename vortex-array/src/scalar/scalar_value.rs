// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core [`ScalarValue`] type definition.

use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::vortex_panic;

use crate::dtype::DType;
use crate::scalar::DecimalValue;
use crate::scalar::PValue;
use crate::scalar::Scalar;

/// The value stored in a [`Scalar`][crate::scalar::Scalar].
///
/// This enum represents the possible non-null values that can be stored in a scalar. When the
/// scalar is null, the value is represented as `None` in the `Option<ScalarValue>` field.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScalarValue {
    /// A boolean value.
    Bool(bool),
    /// A primitive numeric value.
    Primitive(PValue),
    /// A decimal value.
    Decimal(DecimalValue),
    /// A UTF-8 encoded string value.
    Utf8(BufferString),
    /// A binary (byte array) value.
    Binary(ByteBuffer),
    /// A tuple of potentially null scalar values.
    ///
    /// Used as the underlying representation for list, fixed-size list, and struct scalars.
    Tuple(Vec<Option<ScalarValue>>),
    /// A row-specific scalar wrapped by `DType::Variant`.
    Variant(Box<Scalar>),
}

impl ScalarValue {
    /// Returns the zero / identity value for the given [`DType`].
    pub(super) fn zero_value(dtype: &DType) -> Self {
        match dtype {
            DType::Null => vortex_panic!("Null dtype has no zero value"),
            DType::Bool(_) => Self::Bool(false),
            DType::Primitive(ptype, _) => Self::Primitive(PValue::zero(ptype)),
            DType::Decimal(dt, ..) => Self::Decimal(DecimalValue::zero(dt)),
            DType::Utf8(_) => Self::Utf8(BufferString::empty()),
            DType::Binary(_) => Self::Binary(ByteBuffer::empty()),
            DType::List(..) => Self::Tuple(vec![]),
            DType::FixedSizeList(edt, size, _) => {
                let elements = (0..*size).map(|_| Some(Self::zero_value(edt))).collect();
                Self::Tuple(elements)
            }
            DType::Struct(fields, _) => {
                let field_values = fields
                    .fields()
                    .map(|f| Some(Self::zero_value(&f)))
                    .collect();
                Self::Tuple(field_values)
            }
            DType::Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
            DType::Extension(ext_dtype) => {
                // Since we have no way to define a "zero" extension value (since we have no idea
                // what the semantics of the extension is), a best effort attempt is to just use the
                // zero storage value and try to make an extension scalar from that.
                Self::zero_value(ext_dtype.storage_dtype())
            }
            DType::Variant(_) => Self::Variant(Box::new(Scalar::null(DType::Null))),
        }
    }

    /// A similar function to [`ScalarValue::zero_value`], but for nullable [`DType`]s, this returns
    /// `None` instead.
    ///
    /// For non-nullable and nested types that may need null values in their children (as of right
    /// now, that is _only_ `FixedSizeList` and `Struct`), this function will provide `None` as the
    /// default child values (whereas [`ScalarValue::zero_value`] would provide `Some(_)`).
    pub(super) fn default_value(dtype: &DType) -> Option<Self> {
        if dtype.is_nullable() {
            return None;
        }

        Some(match dtype {
            DType::Null => vortex_panic!("Null dtype has no zero value"),
            DType::Bool(_) => Self::Bool(false),
            DType::Primitive(ptype, _) => Self::Primitive(PValue::zero(ptype)),
            DType::Decimal(dt, ..) => Self::Decimal(DecimalValue::zero(dt)),
            DType::Utf8(_) => Self::Utf8(BufferString::empty()),
            DType::Binary(_) => Self::Binary(ByteBuffer::empty()),
            DType::List(..) => Self::Tuple(vec![]),
            DType::FixedSizeList(edt, size, _) => {
                let elements = (0..*size).map(|_| Self::default_value(edt)).collect();
                Self::Tuple(elements)
            }
            DType::Struct(fields, _) => {
                let field_values = fields.fields().map(|f| Self::default_value(&f)).collect();
                Self::Tuple(field_values)
            }
            DType::Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
            DType::Extension(ext_dtype) => {
                // Since we have no way to define a "default" extension value (since we have no idea
                // what the semantics of the extension is), a best effort attempt is to just use the
                // default storage value and try to make an extension scalar from that.
                Self::default_value(ext_dtype.storage_dtype())?
            }
            DType::Variant(_) => Self::Variant(Box::new(Scalar::null(DType::Null))),
        })
    }
}

impl Display for ScalarValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarValue::Bool(b) => write!(f, "{b}"),
            ScalarValue::Primitive(p) => write!(f, "{p}"),
            ScalarValue::Decimal(d) => write!(f, "{d}"),
            ScalarValue::Utf8(s) => {
                let bufstr = s.as_str();
                let str_len = bufstr.chars().count();

                if str_len > 10 {
                    let prefix = String::from_iter(bufstr.chars().take(5));
                    let suffix = String::from_iter(bufstr.chars().skip(str_len - 5));

                    write!(f, "\"{prefix}..{suffix}\"")
                } else {
                    write!(f, "\"{bufstr}\"")
                }
            }
            ScalarValue::Binary(b) => {
                if b.len() > 10 {
                    write!(
                        f,
                        "{}..{}",
                        to_hex(&b[0..5]),
                        to_hex(&b[b.len() - 5..b.len()]),
                    )
                } else {
                    write!(f, "{}", to_hex(b))
                }
            }
            ScalarValue::Tuple(elements) => {
                write!(f, "[")?;
                for (i, element) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    match element {
                        None => write!(f, "null")?,
                        Some(e) => write!(f, "{}", e)?,
                    }
                }
                write!(f, "]")
            }
            ScalarValue::Variant(value) => write!(f, "{value}"),
        }
    }
}

/// Formats a byte slice as a hexadecimal string.
fn to_hex(slice: &[u8]) -> String {
    slice
        .iter()
        .format_with("", |f, b| b(&format_args!("{f:02x}")))
        .to_string()
}
