// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "arbitrary")]
mod arbitrary;
mod array;
#[cfg(feature = "arbitrary")]
pub use arbitrary::ArbitraryBitPackedArray;
pub use array::BitPackedArray;
pub use array::BitPackedArrayParts;
pub use array::bitpack_compress;
pub use array::bitpack_decompress;
pub use array::unpack_iter;

mod compute;

mod vtable;
pub use vtable::BitPackedVTable;
