// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::Union;
use crate::arrays::union::UnionArrayExt;
use crate::validity::Validity;

impl ValidityVTable<Union> for Union {
    fn validity(array: ArrayView<'_, Union>) -> VortexResult<Validity> {
        array.type_ids().validity()
    }
}
