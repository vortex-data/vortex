use arrow_array::{Array, ArrayRef};
use arrow_cast::cast;
use arrow_schema::DataType;
use vortex_error::{VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::stats::ArrayStatistics;
use crate::{ArrayData, Canonical, IntoCanonical};

/// Encoding VTable for canonicalizing an array.
#[allow(clippy::wrong_self_convention)]
pub trait CanonicalVTable {
    fn into_canonical(&self, array: ArrayData) -> VortexResult<Canonical>;

    fn into_arrow(&self, array: ArrayData) -> VortexResult<ArrayRef>;

    fn into_arrow_with_data_type(
        &self,
        array: ArrayData,
        data_type: &DataType,
    ) -> VortexResult<ArrayRef> {
        let arrow_array = self.into_arrow(array)?;
        if arrow_array.data_type() != data_type {
            Ok(cast(&arrow_array, data_type)?)
        } else {
            Ok(arrow_array)
        }
    }
}

/// Implement the [CanonicalVTable] for all encodings with arrays implementing [IntoCanonical].
impl<E: Encoding> CanonicalVTable for E
where
    E::Array: IntoCanonical,
    E::Array: TryFrom<ArrayData, Error = VortexError>,
{
    fn into_canonical(&self, data: ArrayData) -> VortexResult<Canonical> {
        #[cfg(feature = "canonical_counter")]
        data.inc_canonical_counter();
        let canonical = E::Array::try_from(data.clone())?.into_canonical()?;
        canonical.inherit_statistics(data.statistics());
        Ok(canonical)
    }

    fn into_arrow(&self, array: ArrayData) -> VortexResult<ArrayRef> {
        E::Array::try_from(array)?.into_arrow()
    }

    fn into_arrow_with_data_type(
        &self,
        array: ArrayData,
        data_type: &DataType,
    ) -> VortexResult<ArrayRef> {
        E::Array::try_from(array)?.into_arrow_with_data_type(data_type)
    }
}
