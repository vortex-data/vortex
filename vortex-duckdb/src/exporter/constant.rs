use duckdb::core::Value;
use duckdb::vtab::arrow::WritableVector;
use vortex_array::arrays::ConstantArray;
use vortex_error::VortexResult;

use crate::{ColumnExporter, ToDuckDBScalar};

struct ConstantExporter {
    value: Value,
}

pub(crate) fn new_exporter(array: &ConstantArray) -> VortexResult<Box<dyn ColumnExporter>> {
    Ok(Box::new(ConstantExporter {
        value: array.scalar().try_to_duckdb_scalar()?,
    }))
}

impl ColumnExporter for ConstantExporter {
    fn export(
        &self,
        _offset: usize,
        _len: usize,
        vector: &mut dyn WritableVector,
    ) -> VortexResult<()> {
        vector.flat_vector().assign_to_constant(&self.value);
        Ok(())
    }
}
