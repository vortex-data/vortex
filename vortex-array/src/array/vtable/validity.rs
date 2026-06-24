// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::validity::Validity;

/// Validity access for nullable instances of an encoding.
///
/// Non-nullable arrays bypass this hook and report [`Validity::NonNullable`]. Nullable arrays call
/// into the encoding so it can expose either a constant validity state or a row-aligned boolean
/// child array.
pub trait ValidityVTable<V: VTable> {
    /// Returns the [`Validity`] of the array.
    ///
    /// ## Pre-conditions
    ///
    /// - The array DType is nullable.
    ///
    /// ## Post-conditions
    ///
    /// If this returns [`Validity::Array`], the child array must have the same length as `array`
    /// and non-nullable boolean dtype.
    fn validity(array: ArrayView<'_, V>) -> VortexResult<Validity>;
}

/// An implementation of the [`ValidityVTable`] for arrays that delegate validity entirely
/// to a child array.
pub struct ValidityVTableFromChild;

/// Helper trait for encodings whose validity is exactly one child slot.
pub trait ValidityChild<V: VTable> {
    /// Returns the child array that carries validity for `array`.
    fn validity_child(array: ArrayView<'_, V>) -> ArrayRef;
}

impl<V: VTable> ValidityVTable<V> for ValidityVTableFromChild
where
    V: ValidityChild<V>,
{
    fn validity(array: ArrayView<'_, V>) -> VortexResult<Validity> {
        V::validity_child(array).validity()
    }
}

/// An implementation of the [`ValidityVTable`] for arrays that hold an unsliced validity
/// and a slice into it.
pub struct ValidityVTableFromChildSliceHelper;

/// Helper for encodings that keep an unsliced validity child plus a local slice range.
pub trait ValidityChildSliceHelper {
    /// Returns `(unsliced_validity, start, stop)` for this array's logical slice.
    fn unsliced_child_and_slice(&self) -> (&ArrayRef, usize, usize);

    /// Returns a sliced validity child array for the logical range.
    fn sliced_child_array(&self) -> VortexResult<ArrayRef> {
        let (unsliced_validity, start, stop) = self.unsliced_child_and_slice();
        unsliced_validity.slice(start..stop)
    }
}

impl<V: VTable> ValidityVTable<V> for ValidityVTableFromChildSliceHelper
where
    V::TypedArrayData: ValidityChildSliceHelper,
{
    fn validity(array: ArrayView<'_, V>) -> VortexResult<Validity> {
        array.data().sliced_child_array()?.validity()
    }
}
