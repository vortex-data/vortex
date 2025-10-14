// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;

use crate::validity::Validity;
use crate::vtable::VTable;
use crate::{Array, ArrayRef};

pub trait ValidityVTable<V: VTable> {
    fn is_valid(array: &V::Array, index: usize) -> bool;

    fn all_valid(array: &V::Array) -> bool;

    fn all_invalid(array: &V::Array) -> bool;

    /// Returns the number of valid elements in the array.
    ///
    /// ## Post-conditions
    /// - The count is less than or equal to the length of the array.
    fn valid_count(array: &V::Array) -> usize {
        Self::validity_mask(array).true_count()
    }

    /// Returns the number of invalid elements in the array.
    ///
    /// ## Post-conditions
    /// - The count is less than or equal to the length of the array.
    fn invalid_count(array: &V::Array) -> usize {
        Self::validity_mask(array).false_count()
    }

    fn validity_mask(array: &V::Array) -> Mask;
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
    fn is_valid(array: &V::Array, index: usize) -> bool {
        array.validity().is_valid(index)
    }

    fn all_valid(array: &V::Array) -> bool {
        array.validity().all_valid(array.len())
    }

    fn all_invalid(array: &V::Array) -> bool {
        array.validity().all_invalid(array.len())
    }

    fn validity_mask(array: &V::Array) -> Mask {
        array.validity().to_mask(array.len())
    }
}

/// An implementation of the [`ValidityVTable`] for arrays that hold an unsliced validity
/// and a slice into it.
pub struct ValidityVTableFromValiditySliceHelper;

pub trait ValiditySliceHelper {
    fn unsliced_validity_and_slice(&self) -> (&Validity, usize, usize);

    fn sliced_validity(&self) -> Validity {
        let (unsliced_validity, start, stop) = self.unsliced_validity_and_slice();
        unsliced_validity.slice(start..stop)
    }
}

impl<V: VTable> ValidityVTable<V> for ValidityVTableFromValiditySliceHelper
where
    V::Array: ValiditySliceHelper,
{
    fn is_valid(array: &V::Array, index: usize) -> bool {
        let (unsliced_validity, start, _) = array.unsliced_validity_and_slice();
        unsliced_validity.is_valid(start + index)
    }

    fn all_valid(array: &V::Array) -> bool {
        array.sliced_validity().all_valid(array.len())
    }

    fn all_invalid(array: &V::Array) -> bool {
        array.sliced_validity().all_invalid(array.len())
    }

    fn validity_mask(array: &V::Array) -> Mask {
        array.sliced_validity().to_mask(array.len())
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
    fn is_valid(array: &V::Array, index: usize) -> bool {
        V::validity_child(array).is_valid(index)
    }

    fn all_valid(array: &V::Array) -> bool {
        V::validity_child(array).all_valid()
    }

    fn all_invalid(array: &V::Array) -> bool {
        V::validity_child(array).all_invalid()
    }

    fn validity_mask(array: &V::Array) -> Mask {
        V::validity_child(array).validity_mask()
    }
}

/// An implementation of the [`ValidityVTable`] for arrays that hold an unsliced validity
/// and a slice into it.
pub struct ValidityVTableFromChildSliceHelper;

pub trait ValidityChildSliceHelper {
    fn unsliced_child_and_slice(&self) -> (&ArrayRef, usize, usize);

    fn sliced_child_array(&self) -> ArrayRef {
        let (unsliced_validity, start, stop) = self.unsliced_child_and_slice();
        unsliced_validity.slice(start..stop)
    }
}

impl<V: VTable> ValidityVTable<V> for ValidityVTableFromChildSliceHelper
where
    V::Array: ValidityChildSliceHelper,
{
    fn is_valid(array: &V::Array, index: usize) -> bool {
        let (unsliced_validity, start, _) = array.unsliced_child_and_slice();
        unsliced_validity.is_valid(start + index)
    }

    fn all_valid(array: &V::Array) -> bool {
        array.sliced_child_array().all_valid()
    }

    fn all_invalid(array: &V::Array) -> bool {
        array.sliced_child_array().all_invalid()
    }

    fn validity_mask(array: &V::Array) -> Mask {
        array.sliced_child_array().validity_mask()
    }
}
