// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::error::{vortex_ensure, VortexResult};
use crate::duckdb::{LogicalType, Value, Vector};
use crate::exporter::ColumnExporter;

struct InvalidExporter {
    len: usize,
    null_value: Value,
}

pub(crate) fn new_exporter(len: usize, logical_type: &LogicalType) -> Box<dyn ColumnExporter> {
    Box::new(InvalidExporter { len, null_value: Value::null(logical_type) })
}


impl ColumnExporter for InvalidExporter {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        vortex_ensure!(offset + len <= self.len, "invalid exporter: offset + len must be less than or equal to len");

        Ok(vector.reference_value(&self.null_value))
    }
}

#[cfg(test)]
mod tests {
    use vortex::arrays::PrimitiveArray;
    use crate::cpp::duckdb_type;
    use crate::duckdb::{DataChunk, LogicalType};
    use super::*;

    #[test]
    fn all_null_array() {
        let arr = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]);
        let ltype = LogicalType::int32();

        let mut chunk = DataChunk::new([ltype.clone()]);

        let exporter = new_exporter(arr.len(), &ltype).export(0, 3, &mut chunk.get_vector(0)).unwrap();
        chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- CONSTANT INTEGER: 3 = [ NULL]
"#
        );
    }
}