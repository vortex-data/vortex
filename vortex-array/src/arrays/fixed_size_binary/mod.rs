// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-size binary arrays, which store values of one byte width back to back.
//!
//! [`FixedSizeBinary`] implements the shared fixed-width structural compute, so take, filter and
//! slice move whole records without inspecting their bytes.

mod array;
pub use array::FixedSizeBinaryArrayExt;
pub use array::FixedSizeBinaryArraySlotsExt;
pub use array::FixedSizeBinaryData;
pub use array::FixedSizeBinaryDataParts;
pub use array::FixedSizeBinarySlots;

pub(crate) mod compute;
pub use compute::rules::FixedSizeBinaryMaskedValidityRule;

mod vtable;
pub use vtable::FixedSizeBinary;
pub use vtable::FixedSizeBinaryArray;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
    compute::rules::register_parent_rules(session);
}
