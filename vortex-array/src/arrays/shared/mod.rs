// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod vtable;

pub use array::SharedData;
pub use vtable::Shared;
pub use vtable::SharedArray;

#[cfg(test)]
mod tests;
