// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::vortex_panic;

use crate::DecimalValue;
use crate::ExtScalarRef;
use crate::PValue;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScalarValue {
    Bool(bool),
    Primitive(PValue),
    Decimal(DecimalValue),
    Utf8(BufferString),
    Binary(ByteBuffer),
    List(Vec<Option<ScalarValue>>),
    Extension(ExtScalarRef),
}

impl ScalarValue {
    pub fn as_bool(&self) -> bool {
        match self {
            ScalarValue::Bool(b) => *b,
            _ => vortex_panic!("ScalarValue is not a Bool"),
        }
    }

    pub fn as_primitive(&self) -> &PValue {
        match self {
            ScalarValue::Primitive(p) => p,
            _ => vortex_panic!("ScalarValue is not a Primitive"),
        }
    }

    pub fn as_decimal(&self) -> &DecimalValue {
        match self {
            ScalarValue::Decimal(d) => d,
            _ => vortex_panic!("ScalarValue is not a Decimal"),
        }
    }

    pub fn as_utf8(&self) -> &BufferString {
        match self {
            ScalarValue::Utf8(s) => s,
            _ => vortex_panic!("ScalarValue is not a Utf8"),
        }
    }

    pub fn as_binary(&self) -> &ByteBuffer {
        match self {
            ScalarValue::Binary(b) => b,
            _ => vortex_panic!("ScalarValue is not a Binary"),
        }
    }

    pub fn as_list(&self) -> &[Option<ScalarValue>] {
        match self {
            ScalarValue::List(elements) => elements,
            _ => vortex_panic!("ScalarValue is not a List"),
        }
    }

    pub fn as_extension(&self) -> &ExtScalarRef {
        match self {
            ScalarValue::Extension(e) => e,
            _ => vortex_panic!("ScalarValue is not an Extension"),
        }
    }
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
            (ScalarValue::Extension(a), ScalarValue::Extension(b)) => a.partial_cmp(b),
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
            }
            ScalarValue::Extension(e) => write!(f, "{}", e),
        }
    }
}

fn to_hex(slice: &[u8]) -> String {
    slice
        .iter()
        .format_with("", |f, b| b(&format_args!("{f:02x}")))
        .to_string()
}
