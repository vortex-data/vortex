use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::validity::Validity;
use crate::vtable::VTable;

pub trait ValidityVTable<V: VTable> {
    fn is_valid(array: &V::Array, index: usize) -> VortexResult<bool>;

    fn all_valid(array: &V::Array) -> VortexResult<bool>;

    fn all_invalid(array: &V::Array) -> VortexResult<bool>;

    fn validity_mask(array: &V::Array) -> VortexResult<Mask>;
}

/// An implementation of the [`ValidityVTable`] for arrays that hold validity as a child array.
pub struct ValidityVTableFromValidityHelper;

/// Expose validity held as a child array.
pub trait ValidityHelper {
    fn validity(&self) -> &Validity;
}

impl<V: VTable> ValidityVTable<V> for ValidityVTableFromValidityHelper
where
    V::Array: ValidityHelper,
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

/// An implementation of the [`ValidityVTable`] for arrays that delegate validity entirely
/// to a child array.
pub struct ValidityVTableFromChild;

pub trait ValidityChild<V: VTable> {
    fn validity_child(array: &V::Array) -> &dyn Array;
}

impl<V: VTable> ValidityVTable<V> for ValidityVTableFromChild
where
    V: ValidityChild<V>,
{
    fn is_valid(array: &V::Array, index: usize) -> VortexResult<bool> {
        array.validity_child().is_valid(index)
    }

    fn all_valid(array: &V::Array) -> VortexResult<bool> {
        array.validity_child().all_valid()
    }

    fn all_invalid(array: &V::Array) -> VortexResult<bool> {
        array.validity_child().all_invalid()
    }

    fn validity_mask(array: &V::Array) -> VortexResult<Mask> {
        array.validity_child().to_mask(array.len())
    }
}
