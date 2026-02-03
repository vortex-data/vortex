// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

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
    dtype: DType,
    value: Option<ScalarValue>,
}

impl Scalar {
    /// Create a new Scalar with the given DType and value without checking compatibility.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the given DType and value are compatible per the rules defined
    /// in `is_compatible`.
    pub unsafe fn new_unchecked(dtype: DType, value: Option<ScalarValue>) -> Self {
        Self { dtype, value }
    }

    /// Create a new Scalar with the given DType and value.
    pub fn try_new(dtype: DType, value: Option<ScalarValue>) -> VortexResult<Self> {
        vortex_ensure!(
            is_compatible(&dtype, value.as_ref()),
            "Incompatible dtype {} with value {}",
            dtype,
            value
        );
        Ok(Self { dtype, value })
    }

    /// Returns the parts of the Scalar.
    pub fn into_parts(self) -> (DType, Option<ScalarValue>) {
        (self.dtype, self.value)
    }

    /// Returns the DType of the Scalar.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns true if the Scalar is null.
    pub fn is_null(&self) -> bool {
        self.value.is_none()
    }

    /// Returns the scalar value.
    pub fn value(&self) -> Option<&ScalarValue> {
        self.value.as_ref()
    }

    /// Returns the scalar value, consuming the Scalar.
    pub fn into_value(self) -> Option<ScalarValue> {
        self.value
    }
}

/// Check if the given ScalarValue is compatible with the given DType.
fn is_compatible(dtype: &DType, value: Option<&ScalarValue>) -> bool {
    if value.is_none() && !dtype.is_nullable() {
        return false;
    }

    match dtype {
        DType::Null => value.is_none(),
        DType::Bool(_) => matches!(value, ScalarValue::Bool(_)),
        DType::Primitive(ptype, _) => {
            if let ScalarValue::Primitive(pvalue) = value {
                pvalue.ptype() == *ptype
            } else {
                false
            }
        }
        DType::Decimal(dec_dtype, _) => {
            if let ScalarValue::Decimal(dvalue) = value {
                dvalue
                    .fits_in_precision(*dec_dtype)
                    // FIXME(ngates): why the option?
                    .vortex_expect("Failed to check decimal precision compatibility")
            } else {
                false
            }
        }
        DType::Utf8(_) => matches!(value, ScalarValue::Utf8(_)),
        DType::Binary(_) => matches!(value, ScalarValue::Binary(_)),
        DType::List(elem_dtype, _) => {
            if let ScalarValue::List(elements) = value {
                elements
                    .iter()
                    .all(|element| is_compatible(elem_dtype.as_ref(), element.as_ref()))
            } else {
                false
            }
        }
        DType::FixedSizeList(elem_dtype, size, _) => {
            if let ScalarValue::List(elements) = value {
                if elements.len() != *size as usize {
                    return false;
                }
                elements
                    .iter()
                    .all(|element| is_compatible(elem_dtype.as_ref(), element.as_ref()))
            } else {
                false
            }
        }
        DType::Struct(fields, _) => {
            if let ScalarValue::List(values) = value {
                if values.len() != fields.nfields() {
                    return false;
                }
                for (field, field_value) in fields.fields().zip(values.iter()) {
                    if !is_compatible(&field, field_value.as_ref()) {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }
        DType::Extension(ext_dtype) => match value {
            None => ext_dtype.storage_dtype().is_nullable(),
            Some(ScalarValue::Extension(ext_scalar)) => ext_scalar.id() == ext_dtype.id(),
            _ => false,
        },
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
    pub fn as_bool_opt(&self) -> Option<BoolScalar<'_>> {
        let DType::Bool(n) = &self.dtype else {
            return None;
        };
        Some(BoolScalar {
            nullability: *n,
            value: match &self.value {
                None => None,
                Some(ScalarValue::Bool(b)) => Some(*b),
                _ => unreachable!(),
            },
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
        Some(PrimitiveScalar {
            ptype: *ptype,
            nullability: *n,
            pvalue: match &self.value {
                None => None,
                Some(ScalarValue::Primitive(p)) => Some(p),
                _ => unreachable!(),
            },
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
        Some(DecimalScalar {
            decimal_type: dec_dtype,
            nullability: *n,
            dvalue: match &self.value {
                None => None,
                Some(ScalarValue::Decimal(d)) => Some(d),
                _ => unreachable!(),
            },
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
        Some(Utf8Scalar {
            nullability: *n,
            value: match &self.value {
                None => None,
                Some(ScalarValue::Utf8(b)) => Some(b),
                _ => unreachable!(),
            },
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
        Some(BinaryScalar {
            nullability: *n,
            value: match &self.value {
                None => None,
                Some(ScalarValue::Binary(b)) => Some(b),
                _ => unreachable!(),
            },
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
        Some(ListScalar {
            element_dtype,
            nullability: *n,
            elements: match &self.value {
                None => None,
                Some(ScalarValue::List(e)) => Some(e.as_slice()),
                _ => unreachable!(),
            },
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
        Some(FixedSizeListScalar {
            list_size: *element_size,
            element_dtype,
            nullability: *n,
            elements: match &self.value {
                None => None,
                Some(ScalarValue::List(e)) => Some(e.as_slice()),
                _ => unreachable!(),
            },
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
        Some(StructScalar {
            fields,
            nullability: *n,
            values: match &self.value {
                None => None,
                Some(ScalarValue::List(s)) => Some(s.as_slice()),
                _ => unreachable!(),
            },
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
        Some(ExtensionScalar {
            ext_dtype,
            ext_scalar: match &self.value {
                None => None,
                Some(ScalarValue::Extension(e)) => Some(e),
                _ => unreachable!(),
            },
        })
    }
}

impl PartialOrd for Scalar {
    /// Compares two scalar values for ordering.
    ///
    /// # Returns
    /// - `Some(Ordering)` if both scalars have the same data type (ignoring nullability)
    /// - `None` if the scalars have different data types
    ///
    /// # Ordering Rules
    /// When types match, the ordering follows these rules:
    /// - Null values are considered less than all non-null values
    /// - Non-null values are compared according to their natural ordering
    ///
    /// # Examples
    /// ```ignore
    /// // Same types compare successfully
    /// let a = Scalar::primitive(10i32, Nullability::NonNullable);
    /// let b = Scalar::primitive(20i32, Nullability::NonNullable);
    /// assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
    ///
    /// // Different types return None
    /// let int_scalar = Scalar::primitive(10i32, Nullability::NonNullable);
    /// let str_scalar = Scalar::utf8("hello", Nullability::NonNullable);
    /// assert_eq!(int_scalar.partial_cmp(&str_scalar), None);
    ///
    /// // Nulls are less than non-nulls
    /// let null = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
    /// let value = Scalar::primitive(0i32, Nullability::Nullable);
    /// assert_eq!(null.partial_cmp(&value), Some(Ordering::Less));
    /// ```
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !self.dtype().eq_ignore_nullability(other.dtype()) {
            return None;
        }
        self.value().partial_cmp(&other.value())
    }
}

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
                    write!(f, "{}", element)?;
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
