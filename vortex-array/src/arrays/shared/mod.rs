// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod vtable;

pub use array::SharedArray;
pub use vtable::SharedVTable;

#[cfg(test)]
mod tests;
