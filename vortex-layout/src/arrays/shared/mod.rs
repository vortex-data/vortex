// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod kernel;
mod vtable;

pub use array::SharedArrayExt;
pub use array::SharedArraySlotsExt;
pub use array::SharedData;
pub use array::SharedSlots;
pub use array::SharedSlotsView;
pub use kernel::initialize;
pub(crate) use kernel::is_sparse_string_take;
pub use vtable::Shared;
pub use vtable::SharedArray;

#[cfg(test)]
mod tests;
