// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::BinaryScalar;
use crate::BoolScalar;
use crate::DecimalScalar;
use crate::DecimalValue;
use crate::ExtScalarRef;
use crate::FixedSizeListScalar;
use crate::ListScalar;
use crate::PValue;
use crate::PrimitiveScalar;
use crate::StructScalar;
use crate::Utf8Scalar;
use crate::extension::ExtensionScalar;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Scalar {
    // NOTE: the dtype and value must share the same discriminant
    dtype: DType,
    value: ScalarValue,
}

impl Scalar {
    /// Create a new Scalar with the given DType and value without checking compatibility.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the given DType and value are compatible per the rules defined
    /// in `is_compatible`.
    pub unsafe fn new_unchecked(dtype: DType, value: ScalarValue) -> Self {
        Self { dtype, value }
    }

    /// Create a new Scalar with the given DType and value.
    pub fn try_new(dtype: DType, value: ScalarValue) -> VortexResult<Self> {
        vortex_ensure!(
            is_compatible(&dtype, &value),
            "Incompatible dtype {} with value {}",
            dtype,
            value
        );
        Ok(Self { dtype, value })
    }

    /// Returns the DType of the Scalar.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns true if the Scalar is null.
    pub fn is_null(&self) -> bool {
        matches!(self.value, ScalarValue::Null)
    }

    /// Returns the scalar value.
    pub fn value(&self) -> &ScalarValue {
        &self.value
    }
}

/// Check if the given ScalarValue is compatible with the given DType.
fn is_compatible(dtype: &DType, value: &ScalarValue) -> bool {
    if matches!(value, ScalarValue::Null) && !dtype.is_nullable() {
        return false;
    }

    match dtype {
        DType::Null => matches!(value, ScalarValue::Null),
        DType::Bool(_) => matches!(value, ScalarValue::Bool(_)),
        DType::Primitive(ptype, _) => {
            if let ScalarValue::Primitive(pvalue) = value {
                pvalue.ptype() == *ptype
            } else {
                false
            }
        }
        DType::Decimal(..) => matches!(value, ScalarValue::Decimal(_)),
        DType::Utf8(_) => matches!(value, ScalarValue::Utf8(_)),
        DType::Binary(_) => matches!(value, ScalarValue::Binary(_)),
        DType::List(elem_dtype, _) => {
            if let ScalarValue::List(elements) = value {
                elements.iter().all(|element| match element {
                    ScalarValue::Null => true,
                    _ => is_compatible(elem_dtype, element),
                })
            } else {
                false
            }
        }
        DType::FixedSizeList(elem_dtype, size, _) => {
            if let ScalarValue::FixedSizeList(elements) = value {
                if elements.len() != *size as usize {
                    return false;
                }
                elements.iter().all(|element| match element {
                    ScalarValue::Null => true,
                    _ => is_compatible(elem_dtype, element),
                })
            } else {
                false
            }
        }
        DType::Struct(fields, _) => {
            if let ScalarValue::Struct(values) = value {
                if values.len() != fields.nfields() {
                    return false;
                }
                for (field, field_value) in fields.fields().zip(values.iter()) {
                    match field_value {
                        ScalarValue::Null => {}
                        _ => {
                            if !is_compatible(&field, field_value) {
                                return false;
                            }
                        }
                    }
                }
                true
            } else {
                false
            }
        }
        DType::Extension(ext_dtype) => {
            if let ScalarValue::Extension(ext_value) = value {
                ext_value.id() == ext_dtype.id()
            } else {
                false
            }
        }
    }
}

/// Scalar downcasing methods
impl Scalar {
    /// Converts the Scalar into a BoolScalar, panicking if the conversion fails.
    pub fn as_bool(&self) -> BoolScalar<'_> {
        self.as_bool_opt()
            .vortex_expect("Scalar is not a BoolScalar")
    }

    /// Attempts to convert the Scalar into a BoolScalar.
    pub fn as_bool_opt(&self) -> Option<BoolScalar> {
        let DType::Bool(n) = &self.dtype else {
            return None;
        };
        let value = match &self.value {
            ScalarValue::Null => None,
            ScalarValue::Bool(b) => Some(*b),
            _ => unreachable!(),
        };
        Some(BoolScalar {
            nullability: *n,
            value,
            _marker: Default::default(),
        })
    }

    pub fn as_primitive(&self) -> PrimitiveScalar<'_> {
        self.as_primitive_opt()
            .vortex_expect("Scalar is not a PrimitiveScalar")
    }

    pub fn as_primitive_opt(&self) -> Option<PrimitiveScalar<'_>> {
        let DType::Primitive(ptype, n) = &self.dtype else {
            return None;
        };
        let pvalue = match &self.value {
            ScalarValue::Null => None,
            ScalarValue::Primitive(p) => Some(p),
            _ => unreachable!(),
        };
        Some(PrimitiveScalar {
            ptype: *ptype,
            nullability: *n,
            pvalue,
        })
    }

    pub fn as_decimal(&self) -> DecimalScalar<'_> {
        self.as_decimal_opt()
            .vortex_expect("Scalar is not a DecimalScalar")
    }

    pub fn as_decimal_opt(&self) -> Option<DecimalScalar<'_>> {
        let DType::Decimal(dec_dtype, n) = &self.dtype else {
            return None;
        };
        let dvalue = match &self.value {
            ScalarValue::Null => None,
            ScalarValue::Decimal(d) => Some(d),
            _ => unreachable!(),
        };
        Some(DecimalScalar {
            decimal_type: dec_dtype,
            nullability: *n,
            dvalue,
        })
    }

    pub fn as_utf8(&self) -> Utf8Scalar<'_> {
        self.as_utf8_opt()
            .vortex_expect("Scalar is not a Utf8Scalar")
    }

    pub fn as_utf8_opt(&self) -> Option<Utf8Scalar<'_>> {
        let DType::Utf8(n) = &self.dtype else {
            return None;
        };
        let value = match &self.value {
            ScalarValue::Null => None,
            ScalarValue::Utf8(b) => Some(b),
            _ => unreachable!(),
        };
        Some(Utf8Scalar {
            nullability: *n,
            value,
        })
    }

    pub fn as_binary(&self) -> BinaryScalar<'_> {
        self.as_binary_opt()
            .vortex_expect("Scalar is not a BinaryScalar")
    }

    pub fn as_binary_opt(&self) -> Option<BinaryScalar<'_>> {
        let DType::Binary(n) = &self.dtype else {
            return None;
        };
        let value = match &self.value {
            ScalarValue::Null => None,
            ScalarValue::Binary(b) => Some(b),
            _ => unreachable!(),
        };
        Some(BinaryScalar {
            nullability: *n,
            value,
        })
    }

    pub fn as_list(&self) -> ListScalar<'_> {
        self.as_list_opt()
            .vortex_expect("Scalar is not a ListScalar")
    }

    pub fn as_list_opt(&self) -> Option<ListScalar<'_>> {
        let DType::List(element_dtype, n) = &self.dtype else {
            return None;
        };
        let elements = match &self.value {
            ScalarValue::Null => None,
            ScalarValue::List(e) => Some(e.as_slice()),
            _ => unreachable!(),
        };
        Some(ListScalar {
            element_dtype,
            nullability: *n,
            elements,
        })
    }

    pub fn as_fixed_size_list(&self) -> FixedSizeListScalar<'_> {
        self.as_fixed_size_list_opt()
            .vortex_expect("Scalar is not a FixedSizeListScalar")
    }

    pub fn as_fixed_size_list_opt(&self) -> Option<FixedSizeListScalar<'_>> {
        let DType::FixedSizeList(element_dtype, element_size, n) = &self.dtype else {
            return None;
        };
        let elements = match &self.value {
            ScalarValue::Null => None,
            ScalarValue::FixedSizeList(e) => Some(e.as_slice()),
            _ => unreachable!(),
        };
        Some(FixedSizeListScalar {
            element_size: *element_size,
            element_dtype,
            nullability: *n,
            elements,
        })
    }

    pub fn as_struct(&self) -> StructScalar<'_> {
        self.as_struct_opt()
            .vortex_expect("Scalar is not a StructScalar")
    }

    pub fn as_struct_opt(&self) -> Option<StructScalar<'_>> {
        let DType::Struct(fields, n) = &self.dtype else {
            return None;
        };
        let values = match &self.value {
            ScalarValue::Null => None,
            ScalarValue::Struct(s) => Some(s.as_slice()),
            _ => unreachable!(),
        };
        Some(StructScalar {
            fields,
            nullability: *n,
            values,
        })
    }

    pub fn as_extension(&self) -> ExtensionScalar<'_> {
        self.as_extension_opt()
            .vortex_expect("Scalar is not an ExtScalarRef")
    }

    pub fn as_extension_opt(&self) -> Option<ExtensionScalar<'_>> {
        let DType::Extension(ext_dtype) = &self.dtype else {
            return None;
        };
        let ext_value = match &self.value {
            ScalarValue::Null => None,
            ScalarValue::Extension(e) => Some(e),
            _ => unreachable!(),
        };
        Some(ExtensionScalar {
            ext_dtype,
            ext_value,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScalarValue {
    Null,
    Bool(bool),
    Primitive(PValue),
    Decimal(DecimalValue),
    Utf8(BufferString),
    Binary(ByteBuffer),
    List(Vec<ScalarValue>),
    FixedSizeList(Vec<ScalarValue>),
    Struct(Vec<ScalarValue>),
    Extension(ExtScalarRef),
}

impl Display for ScalarValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarValue::Null => write!(f, "null"),
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
                    write!(f, "{}", element)?;
                }
                write!(f, "]")
            }
            ScalarValue::FixedSizeList(elements) => {
                write!(f, "[")?;
                for (i, element) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", element)?;
                }
                write!(f, "]")
            }
            ScalarValue::Struct(fields) => {
                write!(f, "{{")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", field)?;
                }
                write!(f, "}}")
            }
            ScalarValue::Extension(ext) => write!(f, "{}", ext),
        }
    }
}

fn to_hex(slice: &[u8]) -> String {
    slice
        .iter()
        .format_with("", |f, b| b(&format_args!("{f:02x}")))
        .to_string()
}
