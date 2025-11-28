// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Canonical;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::builders::ArrayBuilder;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<BoolVTable> for BoolVTable {
    fn canonicalize(array: &BoolArray) -> VortexResult<Canonical> {
        Ok(Canonical::Bool(array.clone()))
    }

    fn append_to_builder(array: &BoolArray, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        builder.extend_from_array(array.as_ref());
        Ok(())
    }
}
