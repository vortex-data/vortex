use vortex_error::{VortexError, VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::encoding::Encoding;
use crate::ArrayData;

// TODO(ngates): merge this with IntoCanonical VTable and rename to into_canonical_validity.
pub trait ValidityVTable<Array> {
    /// Returns whether the `index` item is valid.
    fn is_valid(&self, array: &Array, index: usize) -> VortexResult<bool> {
        Ok(self.logical_validity(array)?.value(index))
    }

    /// Returns the number of invalid elements in the array.
    fn null_count(&self, array: &Array) -> VortexResult<usize> {
        Ok(self.logical_validity(array)?.false_count())
    }

    fn logical_validity(&self, array: &Array) -> VortexResult<Mask>;
}

impl<E: Encoding> ValidityVTable<ArrayData> for E
where
    E: ValidityVTable<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn is_valid(&self, array: &ArrayData, index: usize) -> VortexResult<bool> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");

        ValidityVTable::is_valid(encoding, array_ref, index)
    }

    fn null_count(&self, array: &ArrayData) -> VortexResult<usize> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        ValidityVTable::null_count(encoding, array_ref)
    }

    fn logical_validity(&self, array: &ArrayData) -> VortexResult<Mask> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        ValidityVTable::logical_validity(encoding, array_ref)
    }
}
