// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::ListArrayExt;
pub use array::ListArraySlotsExt;
pub use array::ListData;
pub use array::ListDataParts;
pub use array::ListSlots;
pub use array::ListSlotsView;
pub use vtable::ListArray;

pub(crate) mod compute;

mod vtable;
pub use vtable::List;
pub use vtable::ListMetadata;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    compute::initialize(session);
}

#[cfg(feature = "_test-harness")]
mod test_harness;

#[cfg(test)]
mod tests;
