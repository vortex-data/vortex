// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Canonical;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::builders::ArrayBuilder;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<PrimitiveVTable> for PrimitiveVTable {
    fn canonicalize(array: &PrimitiveArray) -> VortexResult<Canonical> {
        Ok(Canonical::Primitive(array.clone()))
    }

    fn append_to_builder(
        array: &PrimitiveArray,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        builder.extend_from_array(array.as_ref());
        Ok(())
    }
}
