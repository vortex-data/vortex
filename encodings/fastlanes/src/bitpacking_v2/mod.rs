// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::BitPackedV2ArrayExt;
pub use array::BitPackedV2ArraySlotsExt;
pub use array::BitPackedV2Data;
pub use array::BitPackedV2DataParts;
pub use array::BitPackedV2Slots;
pub use array::ChunkWidths;
pub use array::bitpack_compress;
pub use array::bitpack_decompress;
pub use array::chunk_packed_bytes;
pub use array::unpack_iter;

#[cfg(test)]
mod chunk_widths_tests;

mod vtable;

pub use vtable::BitPackedV2;
pub use vtable::BitPackedV2Array;
