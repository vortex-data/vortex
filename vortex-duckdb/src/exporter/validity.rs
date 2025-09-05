// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::duckdb::Vector;
use crate::exporter::{ColumnExporter, VectorExt};

struct ValidityExporter {
    mask: Mask,
    exporter: Box<dyn ColumnExporter>,
}

pub(crate) fn new_exporter(
    mask: Mask,
    exporter: Box<dyn ColumnExporter>,
) -> Box<dyn ColumnExporter> {
    Box::new(ValidityExporter { mask, exporter })
}

impl ColumnExporter for ValidityExporter {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        if unsafe { vector.set_validity(&self.mask, offset, len) } {
            // All values are null, so no point copying the data.
            return Ok(());
        }

        self.exporter.export(offset, len, vector)?;

        Ok(())
    }
}
