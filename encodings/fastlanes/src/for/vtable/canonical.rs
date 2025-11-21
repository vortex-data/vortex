// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::vtable::CanonicalVTable;

use super::FoRVTable;
use crate::FoRArray;
use crate::r#for::array::for_decompress::decompress;

impl CanonicalVTable<FoRVTable> for FoRVTable {
    fn canonicalize(array: &FoRArray) -> Canonical {
        Canonical::Primitive(decompress(array))
    }
}
