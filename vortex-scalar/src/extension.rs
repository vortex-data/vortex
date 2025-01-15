use std::sync::Arc;

use vortex_dtype::{DType, ExtDType};
use vortex_error::{vortex_bail, VortexError, VortexResult};

use crate::value::ScalarValue;
use crate::Scalar;

pub struct ExtScalar<'a> {
    dtype: &'a DType,
    ext_dtype: &'a ExtDType,
    value: &'a ScalarValue,
}

impl<'a> ExtScalar<'a> {
    pub fn try_new(dtype: &'a DType, value: &'a ScalarValue) -> VortexResult<Self> {
        let DType::Extension(ext_dtype) = dtype else {
            vortex_bail!("Expected extension scalar, found {}", dtype)
        };

        Ok(Self {
            dtype,
            ext_dtype,
            value,
        })
    }

    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the storage scalar of the extension scalar.
    pub fn storage(&self) -> Scalar {
        Scalar::new(self.ext_dtype.storage_dtype().clone(), self.value.clone())
    }

    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        if self.dtype().eq_ignore_nullability(dtype)
            || self.ext_dtype.storage_dtype().eq_ignore_nullability(dtype)
        {
            return Ok(Scalar::new(dtype.clone(), self.value.clone()));
        }

        vortex_bail!("cannot cast {} to {}", self.dtype(), dtype);
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
