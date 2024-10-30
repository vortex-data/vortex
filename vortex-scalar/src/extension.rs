use std::sync::Arc;

use vortex_dtype::{DType, ExtDType};
use vortex_error::{vortex_bail, VortexError, VortexResult};

use crate::value::ScalarValue;
use crate::Scalar;

pub struct ExtScalar<'a> {
    dtype: &'a DType,
    // TODO(ngates): we may need to serialize the value's dtype too so we can pull
    //  it out as a scalar.
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

    /// Returns the stored value of the extension scalar.
    pub fn value(&self) -> &'a ScalarValue {
        self.value
    }

    pub fn cast(&self, _dtype: &DType) -> VortexResult<Scalar> {
        todo!()
    }
}

impl<'a> TryFrom<&'a Scalar> for ExtScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        ExtScalar::try_new(value.dtype(), &value.value)
    }
}

impl Scalar {
    pub fn extension(ext_dtype: Arc<ExtDType>, value: ScalarValue) -> Self {
        Self {
            dtype: DType::Extension(ext_dtype),
            value,
        }
    }
}
