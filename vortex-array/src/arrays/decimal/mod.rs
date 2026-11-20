// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal arrays whose logical precision determines their canonical integer width.
//!
//! [`Decimal`] reads and writes the `vortex.decimal` wire form. It reads a legacy node, stored at
//! another width, as the canonical array or as a [`Widen`].
//!
//! [`Widen`]: crate::arrays::Widen

mod array;
pub use array::DecimalArrayExt;
pub use array::DecimalArraySlotsExt;
pub use array::DecimalData;
pub use array::DecimalDataParts;
pub use array::DecimalSlots;

pub(crate) mod compute;
pub use compute::rules::DecimalMaskedValidityRule;

mod vtable;
pub use vtable::Decimal;
pub use vtable::DecimalArray;

mod utils;
pub use utils::*;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
    compute::rules::register_parent_rules(session);
}
