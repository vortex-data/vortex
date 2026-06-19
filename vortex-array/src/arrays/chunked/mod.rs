// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::ChunkedArrayExt;
pub use array::ChunkedData;
pub use vtable::ChunkedArray;

pub(crate) mod compute;
pub(crate) mod paired_chunks;

mod vtable;
pub use vtable::Chunked;

pub(crate) fn initialize(session: &mut vortex_session::VortexSessionBuilder) {
    compute::kernel::initialize(session);
}

#[cfg(test)]
mod tests;
