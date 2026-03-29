// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::BitPackedArrayParts;
pub use array::BitPackedData;
pub use array::bitpack_compress;
pub use array::bitpack_decompress;
pub use array::unpack_iter;

pub(crate) mod compute;

mod vtable;
pub use vtable::BitPacked;
pub use vtable::BitPackedArray;
