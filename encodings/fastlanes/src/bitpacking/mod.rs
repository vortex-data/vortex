// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::BitPackedArrayExt;
pub use array::BitPackedArraySlotsExt;
pub use array::BitPackedData;
pub use array::BitPackedDataParts;
pub use array::BitPackedSlots;
pub use array::bitpack_compress;
pub use array::bitpack_decompress;
pub use array::unpack_iter;

pub(crate) mod compute;

mod plugin;
mod vtable;

pub(crate) use plugin::BitPackedPatchedPlugin;
pub use vtable::BitPacked;
pub use vtable::BitPackedArray;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
}
