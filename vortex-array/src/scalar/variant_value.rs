// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;

use crate::scalar::DecimalValue;
use crate::scalar::PValue;

/// Semantic value for a `DType::Variant` scalar.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VariantValue {
    Null,
    Bool(bool),
    Primitive(PValue),
    Decimal(DecimalValue),
    Utf8(BufferString),
    Binary(ByteBuffer),
    List(Vec<VariantValue>),
    Object(Vec<(BufferString, VariantValue)>),
}

impl PartialOrd for VariantValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Null, Self::Null) => Some(Ordering::Equal),
            (Self::Bool(a), Self::Bool(b)) => a.partial_cmp(b),
            (Self::Primitive(a), Self::Primitive(b)) => a.partial_cmp(b),
            (Self::Decimal(a), Self::Decimal(b)) => a.partial_cmp(b),
            (Self::Utf8(a), Self::Utf8(b)) => a.partial_cmp(b),
            (Self::Binary(a), Self::Binary(b)) => a.partial_cmp(b),
            (Self::List(a), Self::List(b)) => a.partial_cmp(b),
            (Self::Object(a), Self::Object(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl Display for VariantValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Null => write!(f, "null"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Primitive(v) => write!(f, "{v}"),
            Self::Decimal(v) => write!(f, "{v}"),
            Self::Utf8(v) => write!(f, "\"{}\"", v.as_str()),
            Self::Binary(v) => write!(f, "\"{} bytes\"", v.len()),
            Self::List(values) => {
                write!(f, "[")?;
                for (idx, value) in values.iter().enumerate() {
                    if idx > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{value}")?;
                }
                write!(f, "]")
            }
            Self::Object(fields) => {
                write!(f, "{{")?;
                for (idx, (name, value)) in fields.iter().enumerate() {
                    if idx > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", name.as_str(), value)?;
                }
                write!(f, "}}")
            }
        }
    }
}
