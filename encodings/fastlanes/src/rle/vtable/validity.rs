// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexResult;

use crate::rle::RLE;
use crate::rle::RLEArrayExt;

impl ValidityVTable<RLE> for RLE {
    fn validity(array: ArrayView<'_, RLE>) -> VortexResult<vortex_array::validity::Validity> {
        let start = array.offset();
        let stop = start + array.len();
        array.indices().slice(start..stop)?.validity()
    }
}
