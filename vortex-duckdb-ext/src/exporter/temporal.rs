use vortex::ToCanonical;
use vortex::arrays::TemporalArray;
use vortex::error::VortexResult;

use crate::duckdb::Vector;
use crate::exporter::{ColumnExporter, primitive};

struct TemporalExporter {
    storage_type_exporter: Box<dyn ColumnExporter>,
}

pub(crate) fn new_exporter(array: &TemporalArray) -> VortexResult<Box<dyn ColumnExporter>> {
    Ok(Box::new(TemporalExporter {
        storage_type_exporter: primitive::new_exporter(
            &array.temporal_values().clone().to_primitive()?,
        )?,
    }))
}

impl ColumnExporter for TemporalExporter {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        self.storage_type_exporter.export(offset, len, vector)
    }
}

#[cfg(test)]
mod tests {}
