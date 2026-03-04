// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::ChunkedArray;

mod compute;
mod paired_chunks;

mod vtable;
pub use vtable::ChunkedVTable;

#[cfg(test)]
mod tests;
