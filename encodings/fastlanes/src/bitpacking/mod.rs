// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array; // Public for several helper functions.
pub use array::{BitPackedArray, bitpack_compress, unpack_iter};

mod vtable;
pub use vtable::{BitPackedEncoding, BitPackedVTable};

mod compute;
