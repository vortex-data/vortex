use arrow_array::{Array, ArrayRef};
use arrow_cast::cast;
use arrow_schema::DataType;
use vortex_error::{VortexError, VortexExpect, VortexResult};

use crate::encoding::Encoding;
use crate::{ArrayData, Canonical};

/// Encoding VTable for canonicalizing an array.
#[allow(clippy::wrong_self_convention)]
pub trait CanonicalVTable<Array> {
    fn into_canonical(&self, array: Array) -> VortexResult<Canonical>;
}

impl<E: Encoding> CanonicalVTable<ArrayData> for E
where
    E: CanonicalVTable<E::Array>,
    E::Array: TryFrom<ArrayData, Error = VortexError>,
{
    fn into_canonical(&self, data: ArrayData) -> VortexResult<Canonical> {
        let encoding = data
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        let array = E::Array::try_from(data)?;
        CanonicalVTable::into_canonical(encoding, array)
    }
}
