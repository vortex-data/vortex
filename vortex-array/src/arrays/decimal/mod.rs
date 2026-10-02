// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical decimals whose unscaled values live in one signed integer child.
//!
//! The decimal precision fixes the child's logical width. [`NarrowArray`] can retain smaller
//! physical values or a compressed child. Native-buffer consumers must first call
//! [`DecimalArray::materialize_values`]; canonical execution itself keeps the child encoded.
//!
//! [`NarrowArray`]: crate::arrays::NarrowArray

mod aggregate;
pub(crate) use aggregate::register_aggregate_kernels;

mod array;
pub use array::DecimalArrayExt;
pub use array::DecimalArraySlotsExt;
pub use array::DecimalData;
pub use array::DecimalDataParts;
pub use array::DecimalSlots;

pub(crate) mod compute;
pub use compute::rules::DecimalMaskedValidityRule;

mod plugin;
pub use plugin::DecimalPlugin;

mod vtable;

mod utils;
pub use utils::converted_buffer;
pub use utils::narrowed_decimal;
pub(crate) use utils::widened_buffer;

use crate::Array;

/// Canonical encoding for scaled signed integer values.
///
/// Register [`DecimalPlugin`] to read and write the historical wire representation.
#[derive(Clone, Debug)]
pub struct Decimal;

/// Canonical decimal with one signed integer child and no own buffers.
pub type DecimalArray = Array<Decimal>;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
}

#[cfg(test)]
mod tests;
