// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::ChunkedData;
pub use vtable::ChunkedArray;

pub(crate) mod compute;
mod paired_chunks;

mod vtable;
pub use vtable::Chunked;

#[cfg(test)]
mod tests;
