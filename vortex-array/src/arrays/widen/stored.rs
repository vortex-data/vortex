// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion between an integer child at a fixed dtype and the narrower array that is stored.
//!
//! Most encodings accept an integer child of any width, and they need nothing from this module.
//! Some encodings fix the dtype of an integer child. For example, the most significant part of a
//! `DecimalByteParts` array has a dtype that follows from the decimal precision. Such an encoding
//! can still store the child at a narrower width, and hold it in memory as a [`WidenArray`] at the
//! fixed dtype.
//!
//! The reader and the writer of such an encoding need the same two steps:
//! [`WidenArray::from_stored_integer`] on read and [`WidenArray::to_stored_integer`] on write.
//! This module owns both steps, so that an encoding does not repeat the requirements of a `Widen`.
//! Neither step reads a value. The encoding still owns how it records the stored dtype, and how it
//! reads a child that is stored wider than the fixed dtype.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::wrap_values;
use crate::dtype::DType;

impl WidenArray {
    /// Restores the fixed dtype of a stored integer child, without reading its values.
    ///
    /// Returns `stored` unchanged when it already has `dtype`, and a [`WidenArray`] of `dtype` over
    /// `stored` when it is narrower. A reader calls this function on the child that it decoded at
    /// the stored dtype.
    ///
    /// No scan is needed, because every value of a narrower integer type with the same signedness
    /// is also a value of `dtype`. A decimal precision does not cover every value of its storage
    /// width, so stored decimals go through [`Self::from_stored_decimal`] instead.
    ///
    /// # Errors
    ///
    /// Returns an error if `dtype` is not an integer dtype. Also returns an error if `stored` does
    /// not have `dtype` and breaks a requirement of [`Self::try_new`], which includes a `stored`
    /// that is wider than `dtype`.
    pub fn from_stored_integer(stored: ArrayRef, dtype: DType) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            dtype.is_int(),
            "a stored integer requires an integer dtype, got {dtype}"
        );

        wrap_values(stored, dtype)
    }

    /// Returns the integer array that a writer stores for `array`: the values of an integer
    /// [`WidenArray`], and `array` itself otherwise.
    ///
    /// This is the inverse of [`Self::from_stored_integer`]. A writer stores the result and records
    /// its dtype, and the reader passes the decoded child and the fixed dtype to that function.
    ///
    /// Only a `Widen` at the root of `array` counts. A caller **must** keep the `Widen` at the
    /// root, for example by taking the values of the `Widen` instead of wrapping it in a lazy take.
    /// Otherwise the result keeps the fixed dtype and still holds the `Widen`, which does not
    /// serialize, because an integer `Widen` has no wire form yet (#10733).
    pub fn to_stored_integer(array: &ArrayRef) -> ArrayRef {
        // TODO(connor)[Widen]: Find a `Widen` below a structural encoding too, and return that
        // encoding over the narrow values, for example `Dict(codes, values)` for
        // `Dict(codes, Widen(values))`. Callers keep the `Widen` at the root by hand today. That
        // stops working when optimizer rules move a `Widen` below structural encodings (#10748).
        match array.as_opt::<Widen>() {
            Some(widen) if widen.dtype().is_int() => widen.values().clone(),
            _ => array.clone(),
        }
    }
}
