// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod patch;

pub use array::BoolArrayExt;
pub use array::BoolData;
pub use array::BoolDataParts;

pub(crate) mod compute;

mod vtable;
pub use compute::rules::BoolMaskedValidityRule;
pub use vtable::Bool;
pub use vtable::BoolArray;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
}

#[cfg(feature = "_test-harness")]
mod test_harness;
