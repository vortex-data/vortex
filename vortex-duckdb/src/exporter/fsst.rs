use duckdb::vtab::arrow::WritableVector;
use vortex_error::VortexResult;
use vortex_fsst::FSSTArray;

use crate::ColumnExporter;

struct FSSTExporter {}

pub(crate) fn new_exporter(array: &FSSTArray) -> VortexResult<Box<dyn ColumnExporter>> {
    Ok(Box::new(FSSTExporter {}))
}

impl ColumnExporter for FSSTExporter {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut dyn WritableVector,
    ) -> VortexResult<()> {
        todo!()
    }
}
