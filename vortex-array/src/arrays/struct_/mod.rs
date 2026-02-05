// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::StructArray;
pub use array::StructArrayParts;
pub(crate) mod compute;

mod vtable;
pub use vtable::StructVTable;

#[cfg(test)]
mod tests;
