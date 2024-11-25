use std::cmp::Ordering;
use std::sync::Arc;

pub use scalar_type::ScalarType;
use vortex_buffer::{Buffer, BufferString};
use vortex_dtype::half::f16;
use vortex_dtype::{DType, Nullability};
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
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

/// A single logical item, composed of both a [`ScalarValue`] and a logical [`DType`].
///
/// A [`ScalarValue`] is opaque, and should be accessed via one of the type-specific scalar wrappers
/// for example [`BoolScalar`], [`PrimitiveScalar`], etc.
///
/// Note: [`PartialEq`] and [`PartialOrd`] are implemented only for an exact match of the scalar's
/// dtype, including nullability.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Scalar {
    dtype: DType,
    value: ScalarValue,
}

impl Scalar {
    pub fn new(dtype: DType, value: ScalarValue) -> Self {
        Self { dtype, value }
    }

    #[inline]
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Only the scalar crate should access the ScalarValue directly.
    #[inline]
    pub(crate) fn value(&self) -> &ScalarValue {
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
            value: ScalarValue(InnerScalarValue::Null),
        }
    }

    pub fn null_typed<T: ScalarType>() -> Self {
        Self {
            dtype: T::dtype().as_nullable(),
            value: ScalarValue(InnerScalarValue::Null),
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
                    self.cast(ext_dtype.storage_dtype())?,
                ))
            }
        }
    }

    pub fn into_nullable(self) -> Self {
        Self {
            dtype: self.dtype.as_nullable(),
            value: self.value,
        }
    }
}

impl Scalar {
    pub fn as_bool(&self) -> BoolScalar {
        BoolScalar::try_from(self).vortex_expect("Failed to convert scalar to bool")
    }

    pub fn as_bool_opt(&self) -> Option<BoolScalar> {
        matches!(self.dtype, DType::Bool(..)).then(|| self.as_bool())
    }

    pub fn as_primitive(&self) -> PrimitiveScalar {
        PrimitiveScalar::try_from(self).vortex_expect("Failed to convert scalar to primitive")
    }

    pub fn as_primitive_opt(&self) -> Option<PrimitiveScalar> {
        matches!(self.dtype, DType::Primitive(..)).then(|| self.as_primitive())
    }

    pub fn as_utf8(&self) -> Utf8Scalar {
        Utf8Scalar::try_from(self).vortex_expect("Failed to convert scalar to utf8")
    }

    pub fn as_utf8_opt(&self) -> Option<Utf8Scalar> {
        matches!(self.dtype, DType::Utf8(..)).then(|| self.as_utf8())
    }

    pub fn as_binary(&self) -> BinaryScalar {
        BinaryScalar::try_from(self).vortex_expect("Failed to convert scalar to binary")
    }

    pub fn as_binary_opt(&self) -> Option<BinaryScalar> {
        matches!(self.dtype, DType::Binary(..)).then(|| self.as_binary())
    }

    pub fn as_struct(&self) -> StructScalar {
        StructScalar::try_from(self).vortex_expect("Failed to convert scalar to struct")
    }

    pub fn as_struct_opt(&self) -> Option<StructScalar> {
        matches!(self.dtype, DType::Struct(..)).then(|| self.as_struct())
    }

    pub fn as_list(&self) -> ListScalar {
        ListScalar::try_from(self).vortex_expect("Failed to convert scalar to list")
    }

    pub fn as_list_opt(&self) -> Option<ListScalar> {
        matches!(self.dtype, DType::List(..)).then(|| self.as_list())
    }

    pub fn as_extension(&self) -> ExtScalar {
        ExtScalar::try_from(self).vortex_expect("Failed to convert scalar to extension")
    }

    pub fn as_extension_opt(&self) -> Option<ExtScalar> {
        matches!(self.dtype, DType::Extension(..)).then(|| self.as_extension())
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype && self.value.0 == other.value.0
    }
}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.dtype().eq_ignore_nullability(other.dtype()) {
            self.value.0.partial_cmp(&other.value.0)
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

impl<T> From<Option<T>> for Scalar
where
    T: ScalarType,
    Scalar: From<T>,
{
    fn from(value: Option<T>) -> Self {
        value
            .map(Scalar::from)
            .map(|x| x.into_nullable())
            .unwrap_or_else(|| Scalar {
                dtype: T::dtype().as_nullable(),
                value: ScalarValue(InnerScalarValue::Null),
            })
    }
}

macro_rules! from_vec_for_scalar {
    ($T:ty) => {
        impl From<Vec<$T>> for Scalar {
            fn from(value: Vec<$T>) -> Self {
                Scalar {
                    dtype: DType::List(Arc::from(<$T>::dtype()), Nullability::NonNullable),
                    value: ScalarValue(InnerScalarValue::List(
                        value
                            .into_iter()
                            .map(Scalar::from)
                            .map(|x| x.value.0)
                            .collect::<Arc<[_]>>(),
                    )),
                }
            }
        }
    };
}

// no From<Vec<u8>> because it could either be a List or a Buffer
from_vec_for_scalar!(u16);
from_vec_for_scalar!(u32);
from_vec_for_scalar!(u64);
from_vec_for_scalar!(usize); // For usize only, we implicitly cast for better ergonomics.
from_vec_for_scalar!(i8);
from_vec_for_scalar!(i16);
from_vec_for_scalar!(i32);
from_vec_for_scalar!(i64);
from_vec_for_scalar!(f16);
from_vec_for_scalar!(f32);
from_vec_for_scalar!(f64);
from_vec_for_scalar!(String);
from_vec_for_scalar!(BufferString);
from_vec_for_scalar!(bytes::Bytes);
from_vec_for_scalar!(Buffer);
