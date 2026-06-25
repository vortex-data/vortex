// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::ListArrayExt;
pub use array::ListData;
pub use array::ListDataParts;
pub use vtable::ListArray;

pub(crate) mod compute;

mod vtable;
pub use vtable::List;

pub(crate) fn initialize(session: &mut vortex_session::VortexSessionBuilder) {
    compute::initialize(session);
}

#[cfg(feature = "_test-harness")]
mod test_harness;

#[cfg(test)]
mod tests;
