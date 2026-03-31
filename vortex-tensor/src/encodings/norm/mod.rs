// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::NormVectorArray;

// TODO: Compute operations for NormVector.

mod vtable;
pub use vtable::NormVector;

#[cfg(test)]
mod tests;
