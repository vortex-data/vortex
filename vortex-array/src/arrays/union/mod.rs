// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::UnionArrayExt;
pub use array::UnionDataParts;

pub(crate) mod compute;

mod vtable;
pub use vtable::Union;
pub use vtable::UnionArray;

#[cfg(test)]
mod tests;
