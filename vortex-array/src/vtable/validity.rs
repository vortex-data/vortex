use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::validity::Validity;
use crate::vtable::VTable;

pub trait ValidityVTable<V: VTable> {
    fn is_valid(array: &V::Array, index: usize) -> VortexResult<bool>;

    fn all_valid(array: &V::Array) -> VortexResult<bool>;

    fn all_invalid(array: &V::Array) -> VortexResult<bool>;

    fn validity_mask(array: &V::Array) -> VortexResult<Mask>;
}

/// An implementation of the [`ValidityVTable`] for arrays that hold validity as a child array.
pub struct ValidityVTableFromValidityChild;

/// Expose validity held as a child array.
pub trait ValidityChild {
    fn validity(&self) -> &Validity;
}

impl<V: VTable> ValidityVTable<V> for ValidityVTableFromValidityChild
where
    V::Array: ValidityChild,
{
    fn is_valid(array: &V::Array, index: usize) -> VortexResult<bool> {
        array.validity().is_valid(index)
    }

    fn all_valid(array: &V::Array) -> VortexResult<bool> {
        array.validity().all_valid()
    }

    fn all_invalid(array: &V::Array) -> VortexResult<bool> {
        array.validity().all_invalid()
    }

    fn validity_mask(array: &V::Array) -> VortexResult<Mask> {
        array.validity().to_mask(array.len())
    }
}
