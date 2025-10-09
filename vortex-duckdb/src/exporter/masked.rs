// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::arrays::{ConstantVTable, MaskedArray};
use vortex::error::VortexResult;

use crate::exporter::{ColumnExporter, ConversionCache, constant, new_array_exporter, validity};

pub(crate) fn new_exporter(
    array: &MaskedArray,
    cache: &ConversionCache,
) -> VortexResult<Box<dyn ColumnExporter>> {
    // Duckdb doesn't allow constant arrays with validity, we have to flatten the child
    if let Some(constant_child) = array.child().as_opt::<ConstantVTable>() {
        constant::new_exporter_with_mask(constant_child, array.validity_mask(), cache)
    } else {
        Ok(validity::new_exporter(
            array.validity_mask(),
            new_array_exporter(array.child(), cache)?,
        ))
    }
}
