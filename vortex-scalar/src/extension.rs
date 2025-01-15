use std::sync::Arc;

use vortex_dtype::{DType, ExtDType};
use vortex_error::{vortex_bail, vortex_panic, VortexError, VortexResult};

use crate::value::ScalarValue;
use crate::Scalar;

pub struct ExtScalar<'a> {
    dtype: &'a DType,
    value: &'a ScalarValue,
}

impl<'a> ExtScalar<'a> {
    pub fn try_new(dtype: &'a DType, value: &'a ScalarValue) -> VortexResult<Self> {
        if !matches!(dtype, DType::Extension(..)) {
            vortex_bail!("Expected extension scalar, found {}", dtype)
        }

        Ok(Self { dtype, value })
    }

    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the storage scalar of the extension scalar.
    pub fn storage(&self) -> Scalar {
        let storage_dtype = if let DType::Extension(ext_dtype) = self.dtype() {
            ext_dtype.storage_dtype().clone()
        } else {
            vortex_panic!("Expected extension DType: {}", self.dtype());
        };
        Scalar::new(storage_dtype, self.value.clone())
    }

    pub fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        if self.dtype().eq_ignore_nullability(dtype) {
            if self.dtype.is_nullable() && dtype.is_nullable() && self.value.is_null() {
                vortex_bail!("cannot cast null value to {}", dtype);
            }
            // ScalarValue::cast must reject casting _to_ an extension type because it does not know
            // its own type.
            return Ok(Scalar::new(dtype.clone(), self.value.clone()));
        }

        Ok(Scalar::new(dtype.clone(), self.value.cast(dtype)?))
    }
}

impl<'a> TryFrom<&'a Scalar> for ExtScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        ExtScalar::try_new(value.dtype(), &value.value)
    }
}

impl Scalar {
    pub fn extension(ext_dtype: Arc<ExtDType>, value: Scalar) -> Self {
        Self {
            dtype: DType::Extension(ext_dtype),
            value: value.value().clone(),
        }
    }
}
