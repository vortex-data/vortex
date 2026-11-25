// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core [`ScalarValue`] type definition.

use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::dtype::DType;
use crate::scalar::DecimalValue;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar::UnionValue;

/// The value stored in a [`Scalar`].
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
    /// Used as the underlying representation for list, fixed-size list, map, and struct scalars.
    Tuple(Vec<Option<ScalarValue>>),
    /// A present union value carrying its selected type ID and raw child value.
    Union(UnionValue),
    /// A row-specific scalar wrapped by `DType::Variant`.
    Variant(Box<Scalar>),
}

impl ScalarValue {
    /// Returns the zero / identity value for the given [`DType`].
    pub(super) fn zero_value(dtype: &DType) -> Self {
        Self::try_zero_value(dtype)
            .unwrap_or_else(|| vortex_panic!("{dtype} has no non-null zero value"))
    }

    /// Returns the non-null zero value for `dtype`, or [`None`] if no such value exists.
    pub(super) fn try_zero_value(dtype: &DType) -> Option<Self> {
        Some(match dtype {
            DType::Null => return None,
            DType::Bool(_) => Self::Bool(false),
            DType::Primitive(ptype, _) => Self::Primitive(PValue::zero(ptype)),
            DType::Decimal(dt, ..) => Self::Decimal(DecimalValue::zero(dt)),
            DType::Utf8(_) => Self::Utf8(BufferString::empty()),
            DType::Binary(_) => Self::Binary(ByteBuffer::empty()),
            DType::List(..) => Self::Tuple(vec![]),
            DType::Map(..) => Self::Tuple(vec![]),
            DType::FixedSizeList(edt, size, _) => {
                let elements = (0..*size)
                    .map(|_| Self::try_zero_value(edt).map(Some))
                    .collect::<Option<Vec<_>>>()?;
                Self::Tuple(elements)
            }
            DType::Struct(fields, _) => {
                let field_values = fields
                    .fields()
                    .map(|f| Self::try_zero_value(&f).map(Some))
                    .collect::<Option<Vec<_>>>()?;
                Self::Tuple(field_values)
            }
            DType::Union(variants, _) => {
                let child_dtype = variants
                    .variant_by_index(0)
                    .vortex_expect("union must have at least one variant");
                let child_value = Self::try_zero_value(&child_dtype)?;

                Self::Union(UnionValue::new(
                    variants.child_index_to_tag(0),
                    Some(child_value),
                ))
            }
            DType::Variant(_) => Self::Variant(Box::new(Scalar::null(DType::Null))),
            DType::Extension(ext_dtype) => {
                // Since we have no way to define a "zero" extension value (since we have no idea
                // what the semantics of the extension is), a best effort attempt is to just use the
                // zero storage value and try to make an extension scalar from that.
                Self::try_zero_value(ext_dtype.storage_dtype())?
            }
        })
    }

    /// Returns a valid default value for `dtype`, or [`None`] if the dtype has no default.
    ///
    /// The outer [`Option`] distinguishes a dtype with no default from a nullable dtype, whose valid
    /// default is represented by the inner [`None`].
    pub(super) fn try_default_value(dtype: &DType) -> Option<Option<Self>> {
        if dtype.is_nullable() {
            return Some(None);
        }

        let value = match dtype {
            DType::Null => return Some(None),
            DType::Bool(_) => Self::Bool(false),
            DType::Primitive(ptype, _) => Self::Primitive(PValue::zero(ptype)),
            DType::Decimal(dt, ..) => Self::Decimal(DecimalValue::zero(dt)),
            DType::Utf8(_) => Self::Utf8(BufferString::empty()),
            DType::Binary(_) => Self::Binary(ByteBuffer::empty()),
            DType::List(..) => Self::Tuple(vec![]),
            DType::Map(..) => Self::Tuple(vec![]),
            DType::FixedSizeList(edt, size, _) => {
                let elements = (0..*size)
                    .map(|_| Self::try_default_value(edt))
                    .collect::<Option<Vec<_>>>()?;
                Self::Tuple(elements)
            }
            DType::Struct(fields, _) => {
                let field_values = fields
                    .fields()
                    .map(|field| Self::try_default_value(&field))
                    .collect::<Option<Vec<_>>>()?;
                Self::Tuple(field_values)
            }
            DType::Union(variants, _) => {
                let child_dtype = variants
                    .variant_by_index(0)
                    .vortex_expect("union must have at least one variant");
                let child_value = Self::try_default_value(&child_dtype)?;

                Self::Union(UnionValue::new(variants.child_index_to_tag(0), child_value))
            }
            DType::Variant(_) => Self::Variant(Box::new(Scalar::null(DType::Null))),
            DType::Extension(ext_dtype) => {
                // Since we have no way to define a "default" extension value (since we have no idea
                // what the semantics of the extension is), a best effort attempt is to just use the
                // default storage value and try to make an extension scalar from that.
                Self::try_default_value(ext_dtype.storage_dtype())??
            }
        };

        Some(Some(value))
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
            ScalarValue::Union(value) => {
                write!(f, "union@{}(", value.type_id())?;
                match value.child_value() {
                    Some(value) => write!(f, "{value}"),
                    None => write!(f, "null"),
                }?;
                write!(f, ")")
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
