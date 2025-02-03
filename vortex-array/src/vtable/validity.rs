use vortex_error::{VortexError, VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::encoding::Encoding;
use crate::Array;

pub trait ValidityVTable<Array> {
    /// Returns whether the `index` item is valid.
    fn is_valid(&self, array: &Array, index: usize) -> VortexResult<bool>;

    /// Returns whether the array is all valid.
    ///
    /// This is usually cheaper than computing a precise `invalid_count`.
    fn all_valid(&self, array: &Array) -> VortexResult<bool>;

    /// Returns the number of invalid elements in the array.
    fn invalid_count(&self, array: &Array) -> VortexResult<usize> {
        Ok(self.validity_mask(array)?.false_count())
    }

    fn validity_mask(&self, array: &Array) -> VortexResult<Mask>;
}

impl<E: Encoding> ValidityVTable<Array> for E
where
    E: ValidityVTable<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn is_valid(&self, array: &Array, index: usize) -> VortexResult<bool> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");

        ValidityVTable::is_valid(encoding, array_ref, index)
    }

    fn all_valid(&self, array: &Array) -> VortexResult<bool> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        ValidityVTable::all_valid(encoding, array_ref)
    }

    fn invalid_count(&self, array: &Array) -> VortexResult<usize> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        ValidityVTable::invalid_count(encoding, array_ref)
    }

    fn validity_mask(&self, array: &Array) -> VortexResult<Mask> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        ValidityVTable::validity_mask(encoding, array_ref)
    }
}
