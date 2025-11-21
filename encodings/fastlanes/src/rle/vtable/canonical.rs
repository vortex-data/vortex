// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::vtable::CanonicalVTable;

use super::RLEVTable;
use crate::RLEArray;
use crate::rle::array::rle_decompress::rle_decompress;

impl CanonicalVTable<RLEVTable> for RLEVTable {
    fn canonicalize(array: &RLEArray) -> Canonical {
        Canonical::Primitive(rle_decompress(array))
    }
}
