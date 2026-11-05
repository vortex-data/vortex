// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal arrays whose logical precision determines their canonical integer width.
//!
//! The [`Decimal`] vtable reads only nodes at the canonical width. [`DecimalPlugin`] also reads
//! legacy `vortex.decimal` nodes at any other width, as the canonical array or as a [`Widen`].
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
