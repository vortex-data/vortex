// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::StructArrayExt;
pub use array::StructDataParts;
pub use vtable::StructArray;
pub(crate) mod compute;

mod vtable;
pub use vtable::Struct;

#[cfg(test)]
mod tests;
