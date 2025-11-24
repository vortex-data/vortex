// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::vtable::CanonicalVTable;

use super::DeltaVTable;
use crate::DeltaArray;
use crate::delta::array::delta_decompress::delta_decompress;

impl CanonicalVTable<DeltaVTable> for DeltaVTable {
    fn canonicalize(array: &DeltaArray) -> Canonical {
        Canonical::Primitive(delta_decompress(array))
    }
}
