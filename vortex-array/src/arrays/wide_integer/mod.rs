// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native buffers for signed 128-bit and 256-bit integer extension dtypes.
//!
//! [`WideIntegerArray`] owns one integer buffer and an optional validity child. Its execution
//! exposes that buffer as the extension dtype's fixed-size byte storage without widening or copying.

mod compute;
pub(crate) use compute::initialize;

mod vtable;

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::Array;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::EqMode;
use crate::TypedArrayRef;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::array_slots;
use crate::buffer::BufferHandle;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::integer::integer_dtype;
use crate::match_each_decimal_value_type;
use crate::validity::Validity;

/// A native-buffer encoding for the wide signed integer dtypes.
#[derive(Clone, Debug)]
pub struct WideIntegerEncoding;

/// A signed 128-bit or 256-bit integer array stored at its logical width.
pub type WideIntegerArray = Array<WideIntegerEncoding>;

/// The optional validity child of a [`WideIntegerArray`].
#[array_slots(WideIntegerEncoding)]
pub struct WideIntegerSlots {
    /// Non-nullable boolean validity with one bit per integer.
    #[slot(0)]
    pub validity: Option<ArrayRef>,
}

/// The native values of a [`WideIntegerArray`].
#[derive(Clone, Debug)]
pub struct WideIntegerData {
    pub(super) values: BufferHandle,
    pub(super) values_type: DecimalType,
}

impl WideIntegerData {
    pub(crate) fn buffer_handle(&self) -> &BufferHandle {
        &self.values
    }
}

impl fmt::Display for WideIntegerData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "values_type: {}", self.values_type)
    }
}

impl ArrayEq for WideIntegerData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.values_type == other.values_type && self.values.array_eq(&other.values, accuracy)
    }
}

impl ArrayHash for WideIntegerData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.values_type.hash(state);
        self.values.array_hash(state, accuracy);
    }
}

/// Access to a wide integer array's native values and validity.
pub trait WideIntegerArrayExt: TypedArrayRef<WideIntegerEncoding> {
    /// Returns the native integer buffer handle.
    fn buffer_handle(&self) -> &BufferHandle {
        &self.values
    }

    /// Returns the native integer width.
    fn values_type(&self) -> DecimalType {
        self.values_type
    }

    /// Returns the validity inherited from the optional boolean child.
    fn integer_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[WideIntegerSlots::VALIDITY].as_ref(),
            self.as_ref().dtype().nullability(),
        )
    }
}

impl<T: TypedArrayRef<WideIntegerEncoding>> WideIntegerArrayExt for T {}

impl WideIntegerArray {
    /// Creates a wide integer array from native values and validity.
    ///
    /// Only `i128` and `i256` are accepted. Smaller signed integer types use [`PrimitiveArray`].
    ///
    /// [`PrimitiveArray`]: crate::arrays::PrimitiveArray
    pub fn try_new<T: NativeDecimalType>(values: Buffer<T>, validity: Validity) -> VortexResult<Self> {
        Self::try_new_handle(
            BufferHandle::new_host(values.into_byte_buffer()),
            T::DECIMAL_TYPE,
            validity,
        )
    }

    /// Creates a wide integer array from a host or device buffer.
    pub fn try_new_handle(
        values: BufferHandle,
        values_type: DecimalType,
        validity: Validity,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            matches!(values_type, DecimalType::I128 | DecimalType::I256),
            "Expected a wide integer type, got {values_type}"
        );
        vortex_ensure!(
            values.len().is_multiple_of(values_type.byte_width()),
            "Expected a whole number of {values_type} values, got {} bytes",
            values.len()
        );
        let alignment = match_each_decimal_value_type!(values_type, |T| { Alignment::of::<T>() });
        vortex_ensure!(
            values.is_aligned_to(alignment),
            "Expected integer buffer alignment {alignment:?}, got {:?}",
            values.alignment()
        );

        let len = values.len() / values_type.byte_width();
        let dtype = integer_dtype(values_type, validity.nullability());
        let data = WideIntegerData { values, values_type };
        Self::try_from_parts(ArrayParts::new(WideIntegerEncoding, dtype, len, data).with_slots(
            WideIntegerSlots {
                validity: validity_to_child(&validity, len),
            }
            .into_slots(),
        ))
    }
}

#[cfg(test)]
mod tests;
