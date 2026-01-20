// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::Array;
use crate::ArrayRef;
use crate::validity::Validity;
use crate::vtable::NotSupported;
use crate::vtable::VTable;

pub trait ValidityVTable<V: VTable> {
    /// Returns the [`Validity`] of the array.
    ///
    /// ## Pre-conditions
    ///
    /// - The array DType is nullable.
    fn validity(array: &V::Array) -> VortexResult<Validity>;

    fn validity_mask(array: &V::Array) -> Mask {
        Self::validity(array)
            .vortex_expect("TODO: make this fallible")
            .to_mask(array.len())
    }
}

impl<V: VTable> ValidityVTable<V> for NotSupported {
    fn validity(array: &V::Array) -> VortexResult<Validity> {
        vortex_panic!(
            "Legacy validity is not supported for {} arrays",
            array.encoding_id()
        )
    }
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
    fn validity(array: &V::Array) -> VortexResult<Validity> {
        Ok(array.validity().clone())
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
    fn validity(array: &V::Array) -> VortexResult<Validity> {
        Ok(array.sliced_validity())
    }
}

/// An implementation of the [`ValidityVTable`] for arrays that delegate validity entirely
/// to a child array.
pub struct ValidityVTableFromChild;

pub trait ValidityChild<V: VTable> {
    fn validity_child(array: &V::Array) -> &ArrayRef;
}

impl<V: VTable> ValidityVTable<V> for ValidityVTableFromChild
where
    V: ValidityChild<V>,
{
    fn validity(array: &V::Array) -> VortexResult<Validity> {
        V::validity_child(array).validity()
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
    fn validity(array: &V::Array) -> VortexResult<Validity> {
        array.sliced_child_array().validity()
    }
}
