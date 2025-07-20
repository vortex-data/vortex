// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::validity::Validity;
use crate::vtable::VTable;

/// VTable for validity operations on arrays.
///
/// This trait provides methods for querying and manipulating the validity
/// (null/non-null status) of elements in an array.
pub trait ValidityVTable<V: VTable> {
    /// Check if the element at the given index is valid (non-null).
    ///
    /// # Arguments
    ///
    /// * `array` - The array to check
    /// * `index` - The index to check
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds.
    fn is_valid(array: &V::Array, index: usize) -> VortexResult<bool>;

    /// Check if all elements in the array are valid (non-null).
    ///
    /// # Errors
    ///
    /// Returns an error if the validity cannot be determined.
    fn all_valid(array: &V::Array) -> VortexResult<bool>;

    /// Check if all elements in the array are invalid (null).
    ///
    /// # Errors
    ///
    /// Returns an error if the validity cannot be determined.
    fn all_invalid(array: &V::Array) -> VortexResult<bool>;

    /// Returns the number of valid elements in the array.
    ///
    /// ## Post-conditions
    /// - The count is less than or equal to the length of the array.
    fn valid_count(array: &V::Array) -> VortexResult<usize> {
        Ok(Self::validity_mask(array)?.true_count())
    }

    /// Returns the number of invalid elements in the array.
    ///
    /// ## Post-conditions
    /// - The count is less than or equal to the length of the array.
    fn invalid_count(array: &V::Array) -> VortexResult<usize> {
        Ok(Self::validity_mask(array)?.false_count())
    }

    /// Get a mask indicating which elements are valid.
    ///
    /// # Errors
    ///
    /// Returns an error if the validity mask cannot be computed.
    fn validity_mask(array: &V::Array) -> VortexResult<Mask>;
}

/// Implementation of [`ValidityVTable`] for arrays that store validity as a child array.
///
/// This is used for arrays that have a separate validity array to track null values.
pub struct ValidityVTableFromValidityHelper;

/// Trait for arrays that store validity as a child array.
///
/// Arrays implementing this trait have a `Validity` field that tracks
/// which elements are null or non-null.
pub trait ValidityHelper {
    /// Get a reference to the validity information for this array.
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

/// Implementation of [`ValidityVTable`] for arrays that store an unsliced validity
/// and maintain slice information.
///
/// This is used for sliced arrays that reference validity from the original array.
pub struct ValidityVTableFromValiditySliceHelper;

/// Trait for arrays that have sliced validity information.
///
/// Arrays implementing this trait maintain a reference to the original validity
/// along with slice bounds to determine validity for the sliced portion.
pub trait ValiditySliceHelper {
    /// Get the original validity and slice bounds.
    ///
    /// Returns a tuple of (original_validity, start_index, stop_index).
    fn unsliced_validity_and_slice(&self) -> (&Validity, usize, usize);

    /// Get the validity for this slice.
    ///
    /// This creates a new validity that represents only the sliced portion.
    fn sliced_validity(&self) -> VortexResult<Validity> {
        let (unsliced_validity, start, stop) = self.unsliced_validity_and_slice();
        unsliced_validity.slice(start, stop)
    }
}

impl<V: VTable> ValidityVTable<V> for ValidityVTableFromValiditySliceHelper
where
    V::Array: ValiditySliceHelper,
{
    fn is_valid(array: &V::Array, index: usize) -> VortexResult<bool> {
        let (unsliced_validity, start, _) = array.unsliced_validity_and_slice();
        unsliced_validity.is_valid(start + index)
    }

    fn all_valid(array: &V::Array) -> VortexResult<bool> {
        array.sliced_validity()?.all_valid()
    }

    fn all_invalid(array: &V::Array) -> VortexResult<bool> {
        array.sliced_validity()?.all_invalid()
    }

    fn validity_mask(array: &V::Array) -> VortexResult<Mask> {
        array.sliced_validity()?.to_mask(array.len())
    }
}

/// Implementation of [`ValidityVTable`] for arrays that delegate validity
/// entirely to a child array.
///
/// This is used for wrapper arrays that inherit validity from their child.
pub struct ValidityVTableFromChild;

/// Trait for arrays that delegate validity to a child array.
///
/// Arrays implementing this trait have a child array that determines
/// the validity of elements in the parent array.
pub trait ValidityChild<V: VTable> {
    /// Get the child array that determines validity for this array.
    fn validity_child(array: &V::Array) -> &dyn Array;
}

impl<V: VTable> ValidityVTable<V> for ValidityVTableFromChild
where
    V: ValidityChild<V>,
{
    fn is_valid(array: &V::Array, index: usize) -> VortexResult<bool> {
        V::validity_child(array).is_valid(index)
    }

    fn all_valid(array: &V::Array) -> VortexResult<bool> {
        V::validity_child(array).all_valid()
    }

    fn all_invalid(array: &V::Array) -> VortexResult<bool> {
        V::validity_child(array).all_invalid()
    }

    fn validity_mask(array: &V::Array) -> VortexResult<Mask> {
        V::validity_child(array).validity_mask()
    }
}
