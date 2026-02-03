// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::BinaryScalar;
use crate::BoolScalar;
use crate::DecimalScalar;
use crate::FixedSizeListScalar;
use crate::ListScalar;
use crate::PrimitiveScalar;
use crate::ScalarValue;
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
            value.map(|v| format!("{}", v)).unwrap_or_default()
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
    let Some(value) = value else {
        return dtype.is_nullable();
    };

    match dtype {
        DType::Null => false,
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
            ScalarValue::Extension(ext_scalar) => ext_scalar.id() == ext_dtype.id(),
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
