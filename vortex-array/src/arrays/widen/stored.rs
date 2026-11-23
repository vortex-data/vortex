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

    /// Returns the narrower child of an integer [`WidenArray`] at the root of `array`.
    ///
    /// The writer records the child's dtype. On read, [`Self::from_stored_integer`] restores the
    /// dtype the caller's encoding expects.
    ///
    /// All other inputs are returned as-is. If an integer `Widen` sits below another encoding,
    /// it stays in the result and the writer needs permission to emit `vortex.widen` when edition
    /// checks are enabled. To keep `Widen` at the root during a take, take rows from its child and
    /// wrap the result in `Widen`.
    pub fn to_stored_integer(array: &ArrayRef) -> ArrayRef {
        // TODO(connor)[Widen]: Extract the narrower values through encodings such as `Dict`.
        // When the optimizer moves `Widen` below a `Dict`, the writer needs to keep the codes
        // and store the narrower dictionary values (#10748).
        match array.as_opt::<Widen>() {
            Some(widen) if widen.dtype().is_int() => widen.values().clone(),
            _ => array.clone(),
        }
    }
}
