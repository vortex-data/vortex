// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::DecimalArrayExt;
pub use array::DecimalArraySlotsExt;
pub use array::DecimalData;
pub use array::DecimalDataParts;
pub use array::DecimalSlots;

mod aggregate;

pub(crate) mod compute;
pub use compute::rules::DecimalMaskedValidityRule;

mod plugin;
pub use plugin::DecimalPlugin;

mod vtable;
pub use vtable::Decimal;
pub use vtable::DecimalArray;

mod utils;
pub use utils::*;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    aggregate::initialize(session);
    vtable::initialize(session);
}

#[cfg(test)]
mod tests;
