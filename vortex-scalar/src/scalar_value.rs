// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core [`ScalarValue`] type definition.

use std::cmp::Ordering;
use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::vortex_panic;

use crate::DecimalValue;
// use crate::ExtScalarRef;
use crate::PValue;

/// The value stored in a [`Scalar`][crate::Scalar].
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
    /// A list of potentially null scalar values.
    List(Vec<Option<ScalarValue>>),
    // Extension(ExtScalarRef),
}

// TODO(connor): Docs can be improved (in combination with the associated `Scalar` methods).
impl ScalarValue {
    // TODO(connor): There is an inconsistency here w.r.t. `FixedSizeList` and `Struct` types, since
    // we say that the zero value for those are **not** empty lists. But here we say that a list is
    // a "zero" value if it is empty. So depending on the dtype this might just be incorrect!
    /// Returns true if the scalar represents the zero / identity value for its [`DType`].
    ///
    /// Returns false if the scalar is null.
    ///
    /// See [`Scalar::zero_value()`] for more details about "zero" values.
    ///
    /// [`Scalar::zero_value()`]: crate::Scalar::zero_value
    pub fn is_zero(&self) -> bool {
        // TODO(connor): Is it better to just do == Self::zero_value()?
        match self {
            ScalarValue::Bool(b) => !*b,
            ScalarValue::Primitive(p) => p.is_zero(),
            ScalarValue::Decimal(d) => d.is_zero(),
            ScalarValue::Utf8(s) => s.is_empty(),
            ScalarValue::Binary(b) => b.is_empty(),
            ScalarValue::List(elems) => elems.is_empty(),
        }
    }

    /// Returns the zero / identity value for the given [`DType`].
    ///
    /// # Zero Values
    ///
    /// Here is the list of zero values for each [`DType`] (when the [`DType`] is non-nullable):
    ///
    /// - `Null`: Does not have a "zero" value
    /// - `Bool`: `false`
    /// - `Primitive`: `0`
    /// - `Decimal`: `0`
    /// - `Utf8`: `""`
    /// - `Binary`: An empty buffer
    /// - `List`: An empty list
    /// - `FixedSizeList`: A list (with correct size) of zero values, which is determined by the
    ///   element [`DType`]
    /// - `Struct`: A struct where each field has a zero value, which is determined by the field
    ///   [`DType`]
    ///
    /// - `Extension`: TODO(connor): Is this right?
    ///   The zero value of the storage [`DType`]
    pub fn zero_value(dtype: &DType) -> Self {
        match dtype {
            DType::Null => vortex_panic!("Null dtype has no zero value"),
            DType::Bool(_) => Self::Bool(false),
            DType::Primitive(ptype, _) => Self::Primitive(PValue::zero(ptype)),
            DType::Decimal(dt, ..) => Self::Decimal(DecimalValue::zero(dt)),
            DType::Utf8(_) => Self::Utf8(BufferString::empty()),
            DType::Binary(_) => Self::Binary(ByteBuffer::empty()),
            DType::List(..) => Self::List(vec![]),
            DType::FixedSizeList(edt, size, _) => {
                let elements = (0..*size).map(|_| Some(Self::zero_value(edt))).collect();
                Self::List(elements)
            }
            DType::Struct(fields, _) => {
                let field_values = fields
                    .fields()
                    .map(|f| Some(Self::zero_value(&f)))
                    .collect();
                Self::List(field_values)
            }
            DType::Extension(ext_dtype) => Self::zero_value(ext_dtype.storage_dtype()), // TODO(connor): Fix this!
        }
    }

    /// A similar function to [`ScalarValue::zero_value`], but for nullable [`DType`]s, this returns
    /// `None` instead.
    ///
    /// For non-nullable and nested types that may need null values in their children (as of right
    /// now, that is _only_ `FixedSizeList` and `Struct`), this function will provide `None` as the
    /// default child values (whereas [`ScalarValue::zero_value`] would provide `Some(_)`).
    pub fn default_value(dtype: &DType) -> Option<Self> {
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
            DType::List(..) => Self::List(vec![]),
            DType::FixedSizeList(edt, size, _) => {
                let elements = (0..*size).map(|_| Self::default_value(edt)).collect();
                Self::List(elements)
            }
            DType::Struct(fields, _) => {
                let field_values = fields.fields().map(|f| Self::default_value(&f)).collect();
                Self::List(field_values)
            }
            DType::Extension(ext_dtype) => Self::default_value(ext_dtype.storage_dtype())?, // TODO(connor): Fix this!
        })
    }
}

impl ScalarValue {
    /// Returns the boolean value, panicking if the value is not a [`Bool`][ScalarValue::Bool].
    pub fn as_bool(&self) -> bool {
        match self {
            ScalarValue::Bool(b) => *b,
            _ => vortex_panic!("ScalarValue is not a Bool"),
        }
    }

    /// Returns the primitive value, panicking if the value is not a
    /// [`Primitive`][ScalarValue::Primitive].
    pub fn as_primitive(&self) -> &PValue {
        match self {
            ScalarValue::Primitive(p) => p,
            _ => vortex_panic!("ScalarValue is not a Primitive"),
        }
    }

    /// Returns the decimal value, panicking if the value is not a
    /// [`Decimal`][ScalarValue::Decimal].
    pub fn as_decimal(&self) -> &DecimalValue {
        match self {
            ScalarValue::Decimal(d) => d,
            _ => vortex_panic!("ScalarValue is not a Decimal"),
        }
    }

    /// Returns the UTF-8 string value, panicking if the value is not a [`Utf8`][ScalarValue::Utf8].
    pub fn as_utf8(&self) -> &BufferString {
        match self {
            ScalarValue::Utf8(s) => s,
            _ => vortex_panic!("ScalarValue is not a Utf8"),
        }
    }

    /// Returns the binary value, panicking if the value is not a [`Binary`][ScalarValue::Binary].
    pub fn as_binary(&self) -> &ByteBuffer {
        match self {
            ScalarValue::Binary(b) => b,
            _ => vortex_panic!("ScalarValue is not a Binary"),
        }
    }

    /// Returns the list elements, panicking if the value is not a [`List`][ScalarValue::List].
    pub fn as_list(&self) -> &[Option<ScalarValue>] {
        match self {
            ScalarValue::List(elements) => elements,
            _ => vortex_panic!("ScalarValue is not a List"),
        }
    }

    // pub fn as_extension(&self) -> &ExtScalarRef {
    //     match self {
    //         ScalarValue::Extension(e) => e,
    //         _ => vortex_panic!("ScalarValue is not an Extension"),
    //     }
    // }
}

impl PartialOrd for ScalarValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (ScalarValue::Bool(a), ScalarValue::Bool(b)) => a.partial_cmp(b),
            (ScalarValue::Primitive(a), ScalarValue::Primitive(b)) => a.partial_cmp(b),
            (ScalarValue::Decimal(a), ScalarValue::Decimal(b)) => a.partial_cmp(b),
            (ScalarValue::Utf8(a), ScalarValue::Utf8(b)) => a.partial_cmp(b),
            (ScalarValue::Binary(a), ScalarValue::Binary(b)) => a.partial_cmp(b),
            (ScalarValue::List(a), ScalarValue::List(b)) => a.partial_cmp(b),
            // (ScalarValue::Extension(a), ScalarValue::Extension(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl Display for ScalarValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarValue::Bool(b) => write!(f, "{}", b),
            ScalarValue::Primitive(p) => write!(f, "{}", p),
            ScalarValue::Decimal(d) => write!(f, "{}", d),
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
            ScalarValue::List(elements) => {
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
            } //
              // ScalarValue::Extension(e) => write!(f, "{}", e),
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
