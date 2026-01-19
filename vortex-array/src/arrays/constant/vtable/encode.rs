// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Canonical;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::compute::is_constant;
use crate::vtable::EncodeVTable;

impl EncodeVTable<ConstantVTable> for ConstantVTable {
    fn encode(
        _vtable: &ConstantVTable,
        canonical: &Canonical,
        _like: Option<&ConstantArray>,
    ) -> VortexResult<Option<ConstantArray>> {
        let canonical = canonical.as_ref();
        if is_constant(canonical)?.unwrap_or_default() {
            let scalar = canonical.scalar_at(0);
            Ok(Some(ConstantArray::new(scalar, canonical.len())))
        } else {
            Ok(None)
        }
    }
}
