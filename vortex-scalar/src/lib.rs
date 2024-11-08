use std::cmp::Ordering;

use scalar_type::ScalarType;
use vortex_dtype::DType;

#[cfg(feature = "arbitrary")]
pub mod arbitrary;
mod arrow;
mod binary;
mod bool;
mod datafusion;
mod display;
mod extension;
mod list;
mod null;
mod primitive;
mod pvalue;
mod scalar_type;
#[cfg(feature = "serde")]
mod serde;
mod struct_;
mod utf8;
mod value;

pub use binary::*;
pub use bool::*;
pub use extension::*;
pub use list::*;
pub use primitive::*;
pub use pvalue::*;
pub use struct_::*;
pub use utf8::*;
pub use value::*;
use vortex_error::{vortex_bail, VortexResult};

/// A single logical item, composed of both a [`ScalarValue`] and a logical [`DType`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Scalar {
    pub(crate) dtype: DType,
    pub(crate) value: ScalarValue,
}

impl Scalar {
    pub fn new(dtype: DType, value: ScalarValue) -> Self {
        Self { dtype, value }
    }

    #[inline]
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    #[inline]
    pub fn value(&self) -> &ScalarValue {
        &self.value
    }

    #[inline]
    pub fn into_parts(self) -> (DType, ScalarValue) {
        (self.dtype, self.value)
    }

    #[inline]
    pub fn into_value(self) -> ScalarValue {
        self.value
    }

    pub fn is_valid(&self) -> bool {
        !self.value.is_null()
    }

    pub fn is_null(&self) -> bool {
        self.value.is_null()
    }

    pub fn null(dtype: DType) -> Self {
        assert!(dtype.is_nullable());
        Self {
            dtype,
            value: ScalarValue::Null,
        }
    }

    pub fn cast(&self, dtype: &DType) -> VortexResult<Self> {
        if self.is_null() && !dtype.is_nullable() {
            vortex_bail!("Can't cast null scalar to non-nullable type")
        }

        if self.dtype().eq_ignore_nullability(dtype) {
            return Ok(Scalar {
                dtype: dtype.clone(),
                value: self.value.clone(),
            });
        }

        match dtype {
            DType::Null => vortex_bail!("Can't cast non-null to null"),
            DType::Bool(_) => BoolScalar::try_from(self).and_then(|s| s.cast(dtype)),
            DType::Primitive(..) => PrimitiveScalar::try_from(self).and_then(|s| s.cast(dtype)),
            DType::Utf8(_) => Utf8Scalar::try_from(self).and_then(|s| s.cast(dtype)),
            DType::Binary(_) => BinaryScalar::try_from(self).and_then(|s| s.cast(dtype)),
            DType::Struct(..) => StructScalar::try_from(self).and_then(|s| s.cast(dtype)),
            DType::List(..) => ListScalar::try_from(self).and_then(|s| s.cast(dtype)),
            DType::Extension(ext_dtype) => {
                if !self.value().is_instance_of(ext_dtype.storage_dtype()) {
                    vortex_bail!(
                        "Failed to cast scalar to extension dtype with storage type {:?}, found {:?}",
                        ext_dtype.storage_dtype(),
                        self.dtype()
                    );
                }
                Ok(Scalar::extension(
                    ext_dtype.clone(),
                    self.cast(ext_dtype.storage_dtype())?.value,
                ))
            }
        }
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype && self.value == other.value
    }
}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.dtype().eq_ignore_nullability(other.dtype()) {
            self.value.partial_cmp(&other.value)
        } else {
            None
        }
    }
}

impl AsRef<Self> for Scalar {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<T> From<T> for Scalar
where
    T: ScalarType,
    ScalarValue: From<T>,
{
    fn from(value: T) -> Self {
        Scalar::new(T::dtype(), ScalarValue::from(value))
    }
}

impl<T> From<Option<T>> for Scalar
where
    T: ScalarType,
    ScalarValue: From<Option<T>>,
{
    fn from(value: Option<T>) -> Self {
        Scalar {
            dtype: T::dtype().as_nullable(),
            value: value.into(),
        }
    }
}
