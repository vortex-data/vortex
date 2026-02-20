// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;

struct ValidityExporter {
    mask: Mask,
    exporter: Box<dyn ColumnExporter>,
}

pub(crate) fn new_exporter(
    mask: Mask,
    exporter: Box<dyn ColumnExporter>,
) -> Box<dyn ColumnExporter> {
    if mask.all_true() {
        exporter
    } else {
        Box::new(ValidityExporter { mask, exporter })
    }
}

impl ColumnExporter for ValidityExporter {
    fn export(&self, offset: usize, len: usize, vector: &mut VectorRef) -> VortexResult<()> {
        assert!(
            offset + len <= self.mask.len(),
            "cannot access outside of array"
        );
        if unsafe { vector.set_validity(&self.mask, offset, len) } {
            // All values are null, so no point copying the data.
            return Ok(());
        }

        self.exporter.export(offset, len, vector)?;

        Ok(())
    }
}
