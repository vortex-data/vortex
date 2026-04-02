// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::validity::Validity;

pub trait ValidityVTable<V: VTable> {
    /// Returns the [`Validity`] of the array.
    ///
    /// ## Pre-conditions
    ///
    /// - The array DType is nullable.
    fn validity(array: ArrayView<'_, V>) -> VortexResult<Validity>;
}

/// An implementation of the [`ValidityVTable`] for arrays that hold validity as a child array.
pub struct ValidityVTableFromValidityHelper;

/// Expose validity held as a child array.
pub trait ValidityHelper {
    fn validity(&self) -> &Validity;
}

impl<V: VTable> ValidityVTable<V> for ValidityVTableFromValidityHelper
where
    V::ArrayData: ValidityHelper,
{
    fn validity(array: ArrayView<'_, V>) -> VortexResult<Validity> {
        Ok(array.data().validity().clone())
    }
}

/// An implementation of the [`ValidityVTable`] for arrays that hold an unsliced validity
/// and a slice into it.
pub struct ValidityVTableFromValiditySliceHelper;

pub trait ValiditySliceHelper {
    fn unsliced_validity_and_slice(&self) -> (&Validity, usize, usize);

    fn sliced_validity(&self) -> VortexResult<Validity> {
        let (unsliced_validity, start, stop) = self.unsliced_validity_and_slice();
        unsliced_validity.slice(start..stop)
    }
}

impl<V: VTable> ValidityVTable<V> for ValidityVTableFromValiditySliceHelper
where
    V::ArrayData: ValiditySliceHelper,
{
    fn validity(array: ArrayView<'_, V>) -> VortexResult<Validity> {
        array.data().sliced_validity()
    }
}

/// An implementation of the [`ValidityVTable`] for arrays that delegate validity entirely
/// to a child array.
pub struct ValidityVTableFromChild;

pub trait ValidityChild<V: VTable> {
    fn validity_child(array: &V::ArrayData) -> &ArrayRef;
}

impl<V: VTable> ValidityVTable<V> for ValidityVTableFromChild
where
    V: ValidityChild<V>,
{
    fn validity(array: ArrayView<'_, V>) -> VortexResult<Validity> {
        V::validity_child(array.data()).validity()
    }
}

/// An implementation of the [`ValidityVTable`] for arrays that hold an unsliced validity
/// and a slice into it.
pub struct ValidityVTableFromChildSliceHelper;

pub trait ValidityChildSliceHelper {
    fn unsliced_child_and_slice(&self) -> (&ArrayRef, usize, usize);

    fn sliced_child_array(&self) -> VortexResult<ArrayRef> {
        let (unsliced_validity, start, stop) = self.unsliced_child_and_slice();
        unsliced_validity.slice(start..stop)
    }
}

impl<V: VTable> ValidityVTable<V> for ValidityVTableFromChildSliceHelper
where
    V::ArrayData: ValidityChildSliceHelper,
{
    fn validity(array: ArrayView<'_, V>) -> VortexResult<Validity> {
        array.data().sliced_child_array()?.validity()
    }
}
