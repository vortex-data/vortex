use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use vortex_dtype::datetime::{TemporalMetadata, is_temporal_ext_type};
use vortex_dtype::{DType, ExtDType};
use vortex_error::{VortexError, VortexResult, vortex_bail};

use crate::{Scalar, ScalarValue};

pub struct ExtScalar<'a> {
    ext_dtype: &'a ExtDType,
    value: &'a ScalarValue,
}

impl Display for ExtScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Specialized handling for date/time/timestamp builtin extension types.
        if is_temporal_ext_type(self.ext_dtype().id()) {
            let metadata =
                TemporalMetadata::try_from(self.ext_dtype()).map_err(|_| std::fmt::Error)?;

            let maybe_timestamp = self
                .storage()
                .as_primitive()
                .as_::<i64>()
                .and_then(|maybe_timestamp| {
                    maybe_timestamp.map(|v| metadata.to_jiff(v)).transpose()
                })
                .map_err(|_| std::fmt::Error)?;

            match maybe_timestamp {
                None => write!(f, "null"),
                Some(v) => write!(f, "{v}"),
            }
        } else {
            write!(f, "{}({})", self.ext_dtype().id(), self.storage())
        }
    }
}

impl PartialEq for ExtScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.ext_dtype.eq_ignore_nullability(other.ext_dtype) && self.storage() == other.storage()
    }
}

impl Eq for ExtScalar<'_> {}

// Ord is not implemented since it's undefined for different Extension DTypes
impl PartialOrd for ExtScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if !self.ext_dtype.eq_ignore_nullability(other.ext_dtype) {
            return None;
        }
        self.storage().partial_cmp(&other.storage())
    }
}

impl Hash for ExtScalar<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ext_dtype.hash(state);
        self.storage().hash(state);
    }
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

    pub fn ext_dtype(&self) -> &'a ExtDType {
        self.ext_dtype
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
