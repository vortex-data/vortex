// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::BitPackedArray;
pub use array::BitPackedArrayParts;
pub use array::bitpack_compress;
pub use array::bitpack_decompress;
pub use array::unpack_iter;

mod compute;
mod rules;

mod vtable;
pub use vtable::BitPackedVTable;
