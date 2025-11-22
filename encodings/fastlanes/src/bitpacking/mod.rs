// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::{BitPackedArray, bitpack_compress, bitpack_decompress, unpack_iter};

mod compute;

mod vtable;
pub use vtable::BitPackedVTable;
