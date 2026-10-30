// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal arrays whose logical precision determines their canonical integer width.
//!
//! The compatibility plugin can read a node at another stored width as a [`Widen`], while the
//! [`Decimal`] vtable accepts only the canonical wire width.
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
pub use vtable::DecimalPlugin;

mod utils;
pub use utils::*;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
}
