// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::error::VortexResult;
use vortex::error::vortex_ensure;

use crate::duckdb::LogicalType;
use crate::duckdb::OwnedValue;
use crate::duckdb::Vector;
use crate::exporter::ColumnExporter;

struct AllInvalidExporter {
    len: usize,
    null_value: OwnedValue,
}

pub(crate) fn new_exporter(len: usize, logical_type: &LogicalType) -> Box<dyn ColumnExporter> {
    Box::new(AllInvalidExporter {
        len,
        null_value: OwnedValue::null(logical_type),
    })
}

impl ColumnExporter for AllInvalidExporter {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        vortex_ensure!(
            offset + len <= self.len,
            "invalid exporter: offset + len must be less than or equal to len"
        );

        vector.reference_value(&self.null_value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::arrays::PrimitiveArray;

    use super::*;
    use crate::duckdb::OwnedDataChunk;
    use crate::duckdb::OwnedLogicalType;

    #[test]
    fn all_null_array() {
        let arr = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]);
        let ltype = OwnedLogicalType::int32();

        let mut chunk = OwnedDataChunk::new([ltype.clone()]);

        new_exporter(arr.len(), &ltype)
            .export(0, 3, chunk.get_vector_mut(0))
            .unwrap();
        chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- CONSTANT INTEGER: 3 = [ NULL]
"#
        );
    }
}
