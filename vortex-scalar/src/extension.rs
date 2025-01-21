use std::sync::Arc;

use vortex_dtype::{DType, ExtDType};
use vortex_error::{vortex_bail, VortexError, VortexResult};

use crate::value::ScalarValue;
use crate::Scalar;

pub struct ExtScalar<'a> {
    ext_dtype: &'a ExtDType,
    value: &'a ScalarValue,
}

impl<'a> ExtScalar<'a> {
    pub fn try_new(dtype: &'a DType, value: &'a ScalarValue) -> VortexResult<Self> {
        let DType::Extension(ext_dtype) = dtype else {
            vortex_bail!("Expected extension scalar, found {}", dtype)
        };

        Ok(Self { ext_dtype, value })
    }

    /// Returns the storage scalar of the extension scalar.
    pub fn storage(&self) -> Scalar {
        Scalar::new(self.ext_dtype.storage_dtype().clone(), self.value.clone())
    }

    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        if self.value.is_null() && !dtype.is_nullable() {
            vortex_bail!(
                "cannot cast extension dtype with id {} and storage type {} to {}",
                self.ext_dtype.id(),
                self.ext_dtype.storage_dtype(),
                dtype
            );
        }

        if self.ext_dtype.storage_dtype().eq_ignore_nullability(dtype) {
            // Casting from an extension type to the underlying storage type is OK.
            return Ok(Scalar::new(dtype.clone(), self.value.clone()));
        }

        if let DType::Extension(ext_dtype) = dtype {
            if self.ext_dtype.eq_ignore_nullability(ext_dtype) {
                return Ok(Scalar::new(dtype.clone(), self.value.clone()));
            }
        }

        vortex_bail!(
            "cannot cast extension dtype with id {} and storage type {} to {}",
            self.ext_dtype.id(),
            self.ext_dtype.storage_dtype(),
            dtype
        );
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
